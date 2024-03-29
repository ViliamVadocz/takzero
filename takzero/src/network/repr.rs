use fast_tak::{
    takparse::{Color, Direction, Move, MoveKind, Piece},
    Game,
    Reserves,
};
use ordered_float::NotNan;
use tch::{Device, Tensor};

/// Get the number of possible moves for a given board size.
///
/// # Panics
///
/// Panics for unsupported board sizes.
#[inline]
#[must_use]
pub const fn possible_moves<const N: usize>() -> usize {
    match N {
        3 => 126,
        4 => 480,
        5 => 1_575,
        6 => 4_572,
        7 => 12_495,
        8 => 32_704,
        9 => 82_863,
        10 => 204_700,
        11 => 495_495,
        12 => 1_179_504,
        13 => 2_768_727,
        14 => 6_422_332,
        15 => 14_745_375,
        16 => 33_554_176,
        _ => panic!("Unknown possible move count"),
    }
}

/// Get the number of possible spread patterns for a given board size.
#[inline]
#[must_use]
pub const fn possible_patterns<const N: usize>() -> usize {
    2usize.pow(N as u32) - 2
}

/// Get an index for a move, assuming a tensor output shape
/// where each channel corresponds to a different move type,
/// and the location in each channel corresponds to the starting
/// square of the move.
#[inline]
#[must_use]
pub fn move_index<const N: usize>(m: &Move) -> usize {
    let (row, column) = {
        let s = m.square();
        (s.row() as usize, s.column() as usize)
    };
    let channel = match m.kind() {
        MoveKind::Place(Piece::Flat) => 0,
        MoveKind::Place(Piece::Wall) => 1,
        MoveKind::Place(Piece::Cap) => 2,
        MoveKind::Spread(direction, pattern) => {
            let pattern_offset = (pattern.mask() >> (8 - N)) - 1;
            let direction_offset = possible_patterns::<N>()
                * match direction {
                    Direction::Up => 0,
                    Direction::Right => 1,
                    Direction::Down => 2,
                    Direction::Left => 3,
                };
            3 + pattern_offset as usize + direction_offset
        }
    };
    channel * N * N + row * N + column
}

/// Create a mask for all the impossible moves.
/// Possible moves are false, impossible are true.
pub fn move_mask<const N: usize>(moves: &[Move], device: Device) -> Tensor {
    let mut mask = vec![true; output_size::<N>()];
    for mov in moves {
        mask[move_index::<N>(mov)] = false;
    }
    // FIXME: Can we prevent this copy?
    // `Tensor::from_blob` will not work because backing data will deallocate.
    Tensor::from_slice(&mask)
        .reshape([1, output_channels::<N>() as i64, N as i64, N as i64])
        .to(device)
}

/// Create a tensor containing the given policy.
pub fn policy_tensor<const N: usize>(policy: &[(Move, NotNan<f32>)], device: Device) -> Tensor {
    let mut data = vec![0.0; output_size::<N>()];
    for (mov, p) in policy {
        data[move_index::<N>(mov)] = p.into_inner();
    }
    // FIXME: Can we prevent this copy?
    Tensor::from_slice(&data)
        .reshape([1, output_channels::<N>() as i64, N as i64, N as i64])
        .to(device)
}

/// Get the number of channels needed to encode each move type.
/// This is used by the newer networks.
#[inline]
#[must_use]
pub const fn output_channels<const N: usize>() -> usize {
    let place_types = 3;
    let patterns = possible_patterns::<N>();
    let spreads = 4 * patterns;
    place_types + spreads
}

/// Multiple number of move channels by board size to get
/// the total size of the network output.
#[inline]
#[must_use]
pub const fn output_size<const N: usize>() -> usize {
    N * N * output_channels::<N>()
}

#[inline]
#[must_use]
pub const fn stack_size<const N: usize>() -> usize {
    let piece_type = 3;
    let carry = piece_type + (N - 1);
    let below_carry = N + 1;
    carry + below_carry
}

#[inline]
#[must_use]
pub const fn board_size<const N: usize>() -> usize {
    stack_size::<N>() * N * N
}

#[inline]
#[must_use]
pub fn input_channels<const N: usize>() -> usize {
    const RESERVES: usize = 2; // stones + caps
    const TO_MOVE: usize = 1;
    const FCD: usize = 1;
    2 * (stack_size::<N>() + RESERVES) + TO_MOVE + FCD
}

#[inline]
#[must_use]
pub fn input_size<const N: usize>() -> usize
where
    Reserves<N>: Default,
{
    input_channels::<N>() * N * N
}

#[inline]
#[must_use]
pub fn reserves_ratio<const N: usize>(reserves: Reserves<N>) -> (NotNan<f32>, NotNan<f32>)
where
    Reserves<N>: Default,
{
    let Reserves {
        stones: default_stones,
        caps: default_caps,
    } = Reserves::default();
    (
        NotNan::new(f32::from(reserves.stones) / f32::from(default_stones)).unwrap_or_default(),
        NotNan::new(f32::from(reserves.caps) / f32::from(default_caps)).unwrap_or_default(),
    )
}

/// Write 1s into the passed buffer to represent the game.
/// Assumes the buffer is of correct size and filled with zeroes.
fn game_repr<const N: usize, const HALF_KOMI: i8>(buffer: &mut [f32], game: &Game<N, HALF_KOMI>)
where
    Reserves<N>: Default,
{
    debug_assert_eq!(buffer.len(), input_size::<N>());
    debug_assert!(buffer.iter().all(|x| x.abs() <= f32::EPSILON));

    let index = |row, column, channel| N * N * channel + N * row + column;
    let offset = |color| usize::from(color != game.to_move) * stack_size::<N>();

    for (y, row) in game.board.iter().enumerate() {
        for (x, stack) in row.enumerate() {
            let channel = match stack.top() {
                Some((Piece::Flat, color)) => offset(color),
                Some((Piece::Wall, color)) => 1 + offset(color),
                Some((Piece::Cap, color)) => 2 + offset(color),
                None => continue,
            };
            buffer[index(y, x, channel)] = 1.0;
            for (i, color) in stack
                .colors()
                .reverse()
                .into_iter()
                .skip(1)
                .take(stack_size::<N>() - 3)
                .enumerate()
            {
                buffer[index(y, x, 3 + offset(color) + i)] = 1.0;
            }
        }
    }

    let (mine, other) = match game.to_move {
        Color::White => (game.white_reserves, game.black_reserves),
        Color::Black => (game.black_reserves, game.white_reserves),
    };
    let (stones, caps) = reserves_ratio(mine);
    for i in 0..N * N {
        buffer[2 * board_size::<N>() + i] = stones.into();
        buffer[2 * board_size::<N>() + N * N + i] = caps.into();
    }

    let (stones, caps) = reserves_ratio(other);
    for i in 0..N * N {
        buffer[2 * board_size::<N>() + 2 * N * N + i] = stones.into();
        buffer[2 * board_size::<N>() + 3 * N * N + i] = caps.into();
    }

    if game.to_move == Color::Black {
        for i in 0..N * N {
            buffer[2 * board_size::<N>() + 4 * N * N + i] = 1.0;
        }
    }

    let fcd = f32::from(game.board.flat_diff()) - f32::from(HALF_KOMI) / 2.0;
    let fcd_per_square = fcd / (N * N) as f32;
    for i in 0..N * N {
        buffer[2 * board_size::<N>() + 5 * N * N + i] = fcd_per_square;
    }
}

/// Create a CUDA tensor which represent the game.
pub fn game_to_tensor<const N: usize, const HALF_KOMI: i8>(
    game: &Game<N, HALF_KOMI>,
    device: Device,
) -> Tensor
where
    Reserves<N>: Default,
{
    let mut buffer = vec![0.0; input_size::<N>()];
    game_repr(&mut buffer, game);
    // FIXME: Can we prevent this copy?
    Tensor::from_slice(&buffer)
        .reshape([1, input_channels::<N>() as i64, N as i64, N as i64])
        .to(device)
}

#[cfg(test)]
mod tests {
    use fast_tak::{takparse::Tps, Game};
    use tch::Device;

    use super::{game_repr, input_size};
    use crate::{
        network::repr::{output_size, policy_tensor},
        search::{
            agent::{simple::Simple, Agent},
            env::Environment,
        },
    };

    #[test]
    fn starting_position() {
        let x = 1.0;
        let o = 0.0;
        #[rustfmt::skip]
        let handmade = vec![
            // my pieces
            o, o, o, o, o, o, o, o, o, 
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o, 
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o, 
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            // opponent pieces
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            // my reserves
            x, x, x, x, x, x, x, x, x, // stones
            o, o, o, o, o, o, o, o, o, // caps
            // opponent reserves
            x, x, x, x, x, x, x, x, x, // stones
            o, o, o, o, o, o, o, o, o, // caps
            // white to move
            o, o, o, o, o, o, o, o, o,
            // no komi, no pieces
            o, o, o, o, o, o, o, o, o,
        ];
        assert_eq!(handmade.len(), input_size::<3>());
        let mut buffer = vec![0.0; input_size::<3>()];
        game_repr(&mut buffer, &Game::<3, 0>::default());
        assert_eq!(buffer, handmade);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn complicated_position() {
        let x = 1.0;
        let o = 0.0;
        let p = 5.0 / 21.0;
        let q = 10.0 / 21.0;
        let d = -3.0 / 25.0;
        #[rustfmt::skip]
        let handmade = vec![
            // my pieces
            o, o, o, x, o,  o, x, o, o, o,  o, x, o, o, x,  x, o, x, o, o,  o, o, o, o, o, // flat
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, x, o,  o, o, o, o, o,  o, o, o, o, o, // wall
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, x, o, o, o,  o, o, o, o, o, // cap
            o, o, x, o, o,  o, o, x, o, o,  o, o, x, o, o,  o, o, o, o, o,  o, o, x, o, o, // carry
            o, o, x, o, o,  x, o, o, o, o,  o, x, o, o, o,  o, o, o, o, o,  o, o, x, o, o, // carry
            o, o, o, o, o,  x, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o, // carry
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o, // carry
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            // opponent pieces
            o, o, o, o, o,  o, o, x, x, x,  o, o, o, o, o,  o, o, o, x, o,  o, o, x, o, o, // flat
            o, o, x, o, o,  x, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, x, // wall
            o, o, o, o, o,  o, o, o, o, o,  o, o, x, o, o,  o, o, o, o, o,  o, o, o, o, o, // cap
            o, o, o, o, o,  x, o, o, o, o,  o, x, o, o, o,  o, o, o, o, o,  o, o, o, o, o, // carry
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o, // carry
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, x, o, o, // carry
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o, // carry
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,  o, o, o, o, o,
            // reserves
            p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, // stones
            o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, // caps
            // opponent reserves
            q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, // stones
            o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, // caps
            // black to move
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            // -3 fcd split over 25 squares
            d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d,
        ];
        assert_eq!(handmade.len(), input_size::<5>());
        let tps: Tps = "x2,1221,x,1S/2,2C,2,1,x/x,212,21C,2S,2/2211S,2,21,1,1/x2,221S,2,x 2 23"
            .parse()
            .unwrap();
        let game: Game<5, 4> = tps.into();
        let mut buffer = vec![0.0; input_size::<5>()];
        game_repr(&mut buffer, &game);
        assert_eq!(buffer, handmade);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn tall_stack() {
        let x = 1.0;
        let o = 0.0;
        let p = 5.0 / 10.0;
        let q = 4.0 / 10.0;
        let d = 0.5 / 9.0;
        #[rustfmt::skip]
        let handmade = vec![
            // my pieces
            o, o, o, o, o, o, o, o, o, // flat
            o, o, o, o, o, o, o, o, o, // wall
            o, o, o, o, o, o, o, o, o, // cap
            o, o, o, o, x, o, o, o, o, // carry
            o, o, o, o, o, o, o, o, o, // carry
            o, o, o, o, o, o, o, o, o, 
            o, o, o, o, x, o, o, o, o,
            o, o, o, o, x, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            // opponent pieces
            o, o, o, o, o, o, o, o, o, // flat
            o, o, o, o, x, o, o, o, o, // wall
            o, o, o, o, o, o, o, o, o, // cap
            o, o, o, o, o, o, o, o, o, // carry
            o, o, o, o, x, o, o, o, o, // carry
            o, o, o, o, x, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, o, o, o, o, o,
            o, o, o, o, x, o, o, o, o,
            // my reserves
            p, p, p, p, p, p, p, p, p, // stones
            o, o, o, o, o, o, o, o, o, // caps
            // opponent reserves
            q, q, q, q, q, q, q, q, q, // stones
            o, o, o, o, o, o, o, o, o, // caps
            // white to move
            o, o, o, o, o, o, o, o, o,
            // +0.5 fcd split over 9
            d, d, d, d, d, d, d, d, d,
        ];
        assert_eq!(handmade.len(), input_size::<3>());
        let tps: Tps = "x3/x,21212112212S,x/x3 1 12".parse().unwrap();
        let game: Game<3, -1> = tps.into();
        let mut buffer = vec![0.0; input_size::<3>()];
        game_repr(&mut buffer, &game);
        assert_eq!(buffer, handmade);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn policy() {
        let tps: Tps = "2,1,x/1S,221,x/x,2S,2 1 6".parse().unwrap();
        let game: Game<3, 0> = tps.into();

        let f = 4.0; // flats
                     // caps
        let w = 2.0; // walls
        let s = 1.0; // spreads
        let o = 0.0;

        #[rustfmt::skip]
        let handmade = vec![
            // flat placement
            f, o, o,
            o, o, f,
            o, o, f,
            // wall placement 
            w, o, o,
            o, o, w,
            o, o, w,
            // capstone placement
            o, o, o,
            o, o, o,
            o, o, o,
            // 3#+3
            o, o, o, o, s, o, o, o, o,
            // 2#+2
            o, o, o, o, s, o, o, o, o,
            // 3#+12
            o, o, o, o, o, o, o, o, o,
            // 1#+1
            o, o, o, s, s, o, o, o, o,
            // 3#+21
            o, o, o, o, o, o, o, o, o,
            // 2#+11
            o, o, o, o, o, o, o, o, o,
            // 3#>3
            o, o, o, o, s, o, o, o, o,
            // 2#>2
            o, o, o, o, s, o, o, o, o,
            // 3#>12
            o, o, o, o, o, o, o, o, o,
            // 1#>1
            o, o, o, s, s, o, o, s, o,
            // 3#>21
            o, o, o, o, o, o, o, o, o,
            // 2#>11
            o, o, o, o, o, o, o, o, o,
            // 3#-3
            o, o, o, o, o, o, o, o, o,
            // 2#-2
            o, o, o, o, o, o, o, o, o,
            // 3#-12
            o, o, o, o, o, o, o, o, o,
            // 1#-1
            o, o, o, s, o, o, o, s, o,
            // 3#-21
            o, o, o, o, o, o, o, o, o,
            // 2#-11
            o, o, o, o, o, o, o, o, o,
            // 3#<3
            o, o, o, o, o, o, o, o, o,          
            // 2#<2
            o, o, o, o, o, o, o, o, o,
            // 3#<12
            o, o, o, o, o, o, o, o, o,
            // 1#<1
            o, o, o, o, o, o, o, s, o,
            // 3#<21
            o, o, o, o, o, o, o, o, o,
            // 2#<11
            o, o, o, o, o, o, o, o, o,
        ];

        let mut actions = [Vec::new()];
        game.populate_actions(&mut actions[0]);
        let (policy, ..) = Simple
            .policy_value_uncertainty(&[game], &actions)
            .next()
            .unwrap();
        let buffer: Vec<f32> = policy_tensor::<3>(&policy, Device::Cpu)
            .reshape([output_size::<3>() as i64])
            .try_into()
            .unwrap();

        assert_eq!(buffer, handmade);
    }
}
