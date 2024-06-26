use fast_tak::takparse::Move;
use rand::{seq::IteratorRandom, Rng};
use takzero::{
    network::{
        net4_lcghash::{Env, N},
        repr::{game_to_tensor, input_channels},
    },
    search::env::Environment,
};
use tch::{Device, Tensor};

use super::DEVICE;

pub fn reference_envs(
    ply: usize,
    actions: &mut Vec<Move>,
    rng: &mut impl Rng,
    batch_size: usize,
) -> (Vec<Env>, Tensor) {
    let games: Vec<_> = (0..batch_size)
        .map(|_| Env::new_opening_with_random_steps(rng, actions, ply))
        .collect();
    let tensor = Tensor::cat(
        &games
            .iter()
            .map(|g| game_to_tensor(g, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);
    (games, tensor)
}

#[allow(clippy::type_complexity, dead_code)]
pub fn reference_batches(
    unique_positions: &[Env],
    rng: &mut impl Rng,
    batch_size: usize,
) -> (
    (Vec<Env>, Tensor),
    (Vec<Env>, Tensor),
    (Vec<Env>, Tensor),
    (Vec<Env>, Tensor),
    Tensor,
) {
    let early_game = unique_positions
        .iter()
        .filter(|s| s.ply == 8)
        .cloned()
        .choose_multiple(rng, batch_size);
    let early_tensor = Tensor::concat(
        &early_game
            .iter()
            .map(|s| game_to_tensor(s, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);
    let late_game = unique_positions
        .iter()
        .filter(|s| s.ply == 60)
        .cloned()
        .choose_multiple(rng, batch_size);
    let late_tensor = Tensor::concat(
        &late_game
            .iter()
            .map(|s| game_to_tensor(s, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);

    let mut actions = vec![];
    let (random_early_batch, random_early_tensor) =
        reference_envs(8, &mut actions, rng, batch_size);
    let (random_late_batch, random_late_tensor) = reference_envs(60, &mut actions, rng, batch_size);

    let (_, impossible_early_tensor) = reference_envs(8, &mut actions, rng, batch_size);
    let impossible_early_tensor = impossible_early_tensor.index_select(
        1,
        &Tensor::from_slice(
            &[6, 7, 4, 5, 2, 3, 0, 1]
                .into_iter()
                .chain(8..input_channels::<N>())
                .map(|x| x as i64)
                .collect::<Vec<_>>(),
        )
        .to(DEVICE),
    );

    (
        (
            early_game.into_iter().map(Env::canonical).collect(),
            early_tensor,
        ),
        (
            late_game.into_iter().map(Env::canonical).collect(),
            late_tensor,
        ),
        (
            random_early_batch.into_iter().map(Env::canonical).collect(),
            random_early_tensor,
        ),
        (
            random_late_batch.into_iter().map(Env::canonical).collect(),
            random_late_tensor,
        ),
        impossible_early_tensor,
    )
}
