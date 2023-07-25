use fast_tak::{
    takparse::{Color, Piece},
    Game,
    Reserves,
};
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
    N * N * stack_size::<N>()
}

#[inline]
#[must_use]
pub fn reserves_size<const N: usize>() -> usize
where
    Reserves<N>: Default,
{
    let Reserves { stones, caps } = Reserves::<N>::default();
    stones as usize + caps as usize
}

#[inline]
#[must_use]
pub fn input_size<const N: usize>() -> usize
where
    Reserves<N>: Default,
{
    let to_move = 1;
    2 * (board_size::<N>() + reserves_size::<N>()) + to_move
}

fn game_repr<const N: usize, const HALF_KOMI: i8>(buffer: &mut [f32], game: &Game<N, HALF_KOMI>)
where
    Reserves<N>: Default,
{
    debug_assert_eq!(buffer.len(), input_size::<N>());
    debug_assert!(buffer.iter().all(|x| x.abs() <= f32::EPSILON));

    let offset = |color| {
        if color == game.to_move {
            0
        } else {
            board_size::<N>()
        }
    };
    for (i, stack) in game.board.iter().flatten().enumerate() {
        let pos = i * stack_size::<N>();
        let index = pos
            + match stack.top() {
                Some((Piece::Flat, color)) => offset(color),
                Some((Piece::Wall, color)) => 1 + offset(color),
                Some((Piece::Cap, color)) => 2 + offset(color),
                None => continue,
            };
        buffer[index] = 1.0;
        for (ii, color) in stack // FIXME: Colors maybe in reverse order?
            .colors()
            .reverse()
            .into_iter()
            .skip(1)
            .take(stack_size::<N>() - 3)
            .enumerate()
        {
            buffer[pos + ii + 3 + offset(color)] = 1.0;
        }
    }

    let (mine, other) = match game.to_move {
        Color::White => (game.white_reserves, game.black_reserves),
        Color::Black => (game.black_reserves, game.white_reserves),
    };
    let Reserves { stones, caps } = mine;
    let max_stones = Reserves::<N>::default().stones as usize;
    for i in 0..(stones as usize) {
        buffer[2 * board_size::<N>() + i] = 1.0;
    }
    for i in 0..(caps as usize) {
        buffer[2 * board_size::<N>() + max_stones + i] = 1.0;
    }
    let Reserves { stones, caps } = other;
    for i in 0..(stones as usize) {
        buffer[2 * board_size::<N>() + reserves_size::<N>() + i] = 1.0;
    }
    for i in 0..(caps as usize) {
        buffer[2 * board_size::<N>() + reserves_size::<N>() + max_stones + i] = 1.0;
    }

    if game.to_move == Color::Black {
        buffer[input_size() - 1] = 1.0;
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
            // top      carry    below carry
            o, o, o,    o, o,    o, o, o, o, // a1
            o, o, o,    o, o,    o, o, o, o, // b1
            o, o, o,    o, o,    o, o, o, o, // c1
            o, o, o,    o, o,    o, o, o, o, // a2
            o, o, o,    o, o,    o, o, o, o, // b2
            o, o, o,    o, o,    o, o, o, o, // c2
            o, o, o,    o, o,    o, o, o, o, // a3
            o, o, o,    o, o,    o, o, o, o, // b3
            o, o, o,    o, o,    o, o, o, o, // c3
            // opponent pieces
            o, o, o,    o, o,    o, o, o, o, // a1
            o, o, o,    o, o,    o, o, o, o, // b1
            o, o, o,    o, o,    o, o, o, o, // c1
            o, o, o,    o, o,    o, o, o, o, // a2
            o, o, o,    o, o,    o, o, o, o, // b2
            o, o, o,    o, o,    o, o, o, o, // c2
            o, o, o,    o, o,    o, o, o, o, // a3
            o, o, o,    o, o,    o, o, o, o, // b3
            o, o, o,    o, o,    o, o, o, o, // c3
            // reserves
            x, x, x, x, x, x, x, x, x, x, // mine
            x, x, x, x, x, x, x, x, x, x, // opponent
            o // white to move
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
        #[rustfmt::skip]
        let handmade = vec![
            // my pieces
            // top      carry          below carry
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // a1
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // b1
            o, o, o,    x, x, o, o,    o, o, o, o, o, o, // c1
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // d1
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // e1

            o, o, o,    o, x, x, o,    o, o, o, o, o, o, // a2
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // b2
            o, o, o,    x, o, o, o,    o, o, o, o, o, o, // c2
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // d2
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // e2
            
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // a3
            x, o, o,    o, x, o, o,    o, o, o, o, o, o, // b3
            o, o, o,    x, o, o, o,    o, o, o, o, o, o, // c3
            o, x, o,    o, o, o, o,    o, o, o, o, o, o, // d3
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // e3

            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // a4
            o, o, x,    o, o, o, o,    o, o, o, o, o, o, // b4
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // c4
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // d4
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // e4

            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // a5
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // b5
            o, o, o,    x, x, o, o,    o, o, o, o, o, o, // c5
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // d5
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // e5

            // opponent pieces
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // a1
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // b1
            o, x, o,    o, o, o, o,    o, o, o, o, o, o, // c1
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // d1
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // e1

            o, x, o,    x, o, o, o,    o, o, o, o, o, o, // a2
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // b2
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // c2
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // d2
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // e2

            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // a3
            o, o, o,    x, o, o, o,    o, o, o, o, o, o, // b3
            o, o, x,    o, o, o, o,    o, o, o, o, o, o, // c3
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // d3
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // e3

            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // a4
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // b4
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // c4
            x, o, o,    o, o, o, o,    o, o, o, o, o, o, // d4
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // e4

            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // a5
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // b5
            x, o, o,    o, o, x, o,    o, o, o, o, o, o, // c5
            o, o, o,    o, o, o, o,    o, o, o, o, o, o, // d5
            o, x, o,    o, o, o, o,    o, o, o, o, o, o, // e5

            // reserves
            x, x, x, x, x, o, o, o, o, o, // my stones
            o, o, o, o, o, o, o, o, o, o, o,
            o, // my cap
            x, x, x, x, x, x, x, x, x, x, // opponent stones 
            o, o, o, o, o, o, o, o, o, o, o,
            o, // opponent cap

            x // black to move
        ];
        assert_eq!(handmade.len(), input_size::<5>());
        let game = Game::<5, 0>::from_ptn_moves(&[
            "e3", "e2", "d2", "Sd3", "d4", "c4", "Cb3", "Cb4", "c3", "c2", "c3-", "c3", "b3>",
            "b3", "a3", "b2", "a3>", "a3", "a1", "a3>", "Sb1", "a2", "Se5", "a3", "b1<", "a3-",
            "2a1+", "a4", "c5", "b5", "d5", "b5>", "Sb1", "b5", "b1>", "b5>", "d5<", "d1", "c1<",
            "c1", "b1<", "d1<", "a1>", "d1", "b1>",
        ]);
        let mut buffer = vec![0.0; input_size::<5>()];
        game_repr(&mut buffer, &game);
        assert_eq!(buffer, handmade);
    }

    #[test]
    fn tall_stack() {
        let x = 1.0;
        let o = 0.0;
        #[rustfmt::skip]
        let handmade = vec![
            // my pieces
            // top      carry    below carry
            o, o, o,    o, o,    o, o, o, o, // a1
            o, o, o,    o, o,    o, o, o, o, // b1
            o, o, o,    o, o,    o, o, o, o, // c1
            o, o, o,    o, o,    o, o, o, o, // a2
            o, o, o,    x, o,    o, x, x, o, // b2
            o, o, o,    o, o,    o, o, o, o, // c2
            o, o, o,    o, o,    o, o, o, o, // a3
            o, o, o,    o, o,    o, o, o, o, // b3
            o, o, o,    o, o,    o, o, o, o, // c3
            // opponent
            o, o, o,    o, o,    o, o, o, o, // a1
            o, o, o,    o, o,    o, o, o, o, // b1
            o, o, o,    o, o,    o, o, o, o, // c1
            o, o, o,    o, o,    o, o, o, o, // a2
            o, x, o,    o, x,    x, o, o, x, // b2
            o, o, o,    o, o,    o, o, o, o, // c2
            o, o, o,    o, o,    o, o, o, o, // a3
            o, o, o,    o, o,    o, o, o, o, // b3
            o, o, o,    o, o,    o, o, o, o, // c3
            // reserves
            x, x, x, x, x, o, o, o, o, o, // mine
            x, x, x, x, o, o, o, o, o, o, // opponent
            o // white to move
        ];
        assert_eq!(handmade.len(), input_size::<3>());
        let tps: Tps = "x3/x,21212112212S,x/x3 1 12".parse().unwrap();
        let game: Game<3, 0> = tps.into();
        let mut buffer = vec![0.0; input_size::<3>()];
        game_repr(&mut buffer, &game);
        assert_eq!(buffer, handmade);
    }
}
