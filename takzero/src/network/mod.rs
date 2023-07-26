use std::path::Path;

use fast_tak::takparse::{Direction, Move, MoveKind, Piece};
use tch::nn::VarStore;

pub mod net3;
mod residual;

pub trait Network: Default + Sized {
    fn vs(&mut self) -> &mut VarStore;

    fn save(&mut self, path: impl AsRef<Path>) -> Result<(), tch::TchError> {
        self.vs().save(path)
    }

    fn load(path: impl AsRef<Path>) -> Result<Self, tch::TchError> {
        let mut nn = Self::default();
        nn.vs().load(path)?;
        Ok(nn)
    }
}

/// Get the number of possible moves for a given board size.
const fn possible_moves<const N: usize>() -> usize {
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
const fn possible_patterns<const N: usize>() -> usize {
    2usize.pow(N as u32) - 2
}

/// Get an index for a move, assuming a tensor output shape
/// where each channel corresponds to a different move type,
/// and the location in each channel corresponds to the starting
/// square of the move.
fn move_index<const N: usize>(m: &Move) -> usize {
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

/// Get the number of channels needed to encode each move type.
/// This is used by the newer networks.
const fn output_channels<const N: usize>() -> usize {
    let place_types = 3;
    let patterns = possible_patterns::<N>();
    let spreads = 4 * patterns;
    place_types + spreads
}

/// Multiple number of move channels by board size to get
/// the total size of the network output.
const fn output_size<const N: usize>() -> usize {
    N * N * output_channels::<N>()
}
