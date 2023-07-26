use fast_tak::{
    takparse::{Color, Piece},
    Game,
    Reserves,
};
use ordered_float::NotNan;
use tch::{Device, Kind, Tensor};

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
    let reserves = 2; // stones + caps
    let to_move = 1;
    2 * (stack_size::<N>() + reserves) + to_move
}

#[inline]
#[must_use]
pub fn input_size<const N: usize>() -> usize
where
    Reserves<N>: Default,
{
    input_channels::<N>() * N * N
}

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
}

pub fn game_to_tensor<const N: usize, const HALF_KOMI: i8>(
    game: &Game<N, HALF_KOMI>,
    device: Device,
) -> Tensor
where
    Reserves<N>: Default,
{
    let mut buffer = vec![0.0; input_size::<N>()];
    let size = [buffer.len() as i64];
    game_repr(&mut buffer, game);
    let tensor = unsafe {
        Tensor::from_blob(
            buffer.as_ptr().cast(),
            &size,
            &size,
            Kind::Float,
            Device::Cpu,
        )
    };
    tensor.to(device)
}

#[cfg(test)]
mod tests {
    use fast_tak::{takparse::Tps, Game};

    use super::{game_repr, input_size};

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
        ];
        assert_eq!(handmade.len(), input_size::<3>());
        let mut buffer = vec![0.0; input_size::<3>()];
        game_repr(&mut buffer, &Game::<3, 0>::default());
        assert_eq!(buffer, handmade);
    }

    #[test]
    fn complicated_position() {
        let x = 1.0;
        let o = 0.0;
        let p = 5.0 / 21.0;
        let q = 10.0 / 21.0;
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
        ];
        assert_eq!(handmade.len(), input_size::<5>());
        let tps: Tps = "x2,1221,x,1S/2,2C,2,1,x/x,212,21C,2S,2/2211S,2,21,1,1/x2,221S,2,x 2 23"
            .parse()
            .unwrap();
        let game: Game<5, 0> = tps.into();
        let mut buffer = vec![0.0; input_size::<5>()];
        game_repr(&mut buffer, &game);
        assert_eq!(buffer, handmade);
    }

    #[test]
    fn tall_stack() {
        let x = 1.0;
        let o = 0.0;
        let p = 5.0 / 10.0;
        let q = 4.0 / 10.0;
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
        ];
        assert_eq!(handmade.len(), input_size::<3>());
        let tps: Tps = "x3/x,21212112212S,x/x3 1 12".parse().unwrap();
        let game: Game<3, 0> = tps.into();
        let mut buffer = vec![0.0; input_size::<3>()];
        game_repr(&mut buffer, &game);
        assert_eq!(buffer, handmade);
    }
}
