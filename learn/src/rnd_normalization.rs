use std::fmt::Write;

use fast_tak::takparse::{Move, Tps};
use rand::{seq::IteratorRandom, Rng};
use takzero::{
    network::{repr::game_to_tensor, Network},
    search::env::Environment,
};
use tch::{Device, Tensor};

use super::{Env, Net, HALF_KOMI, N};

fn to_reference(envs: Vec<Env>, device: Device) -> Tensor {
    Tensor::concat(
        &envs
            .into_iter()
            .map(|env| game_to_tensor::<N, HALF_KOMI>(&env, device))
            .collect::<Vec<_>>(),
        0,
    )
}

fn random_env(ply: usize, actions: &mut Vec<Move>, rng: &mut impl Rng) -> Env {
    let mut env = Env::default();
    for _ in 0..ply {
        env.populate_actions(actions);
        let Some(action) = actions.drain(..).choose(rng) else {
            break;
        };
        env.step(action);
    }
    env
}

fn reference_games_string(envs: &[Env]) -> String {
    envs.iter()
        .map(|env| Tps::from(env.clone()))
        .fold(String::new(), |mut s, tps| {
            writeln!(s, "{tps}").unwrap();
            s
        })
}

pub fn reference_games(device: Device, rng: &mut impl Rng) -> (Tensor, Tensor) {
    const EARLY_AMOUNT: usize = 256;
    const EARLY_PLY: usize = 4;
    const LATE_AMOUNT: usize = 256;
    const LATE_PLY: usize = 120;

    // Generate reference games.
    let mut actions = Vec::new();
    let early: Vec<_> = (0..EARLY_AMOUNT)
        .map(|i| random_env(EARLY_PLY + i % 2, &mut actions, rng))
        .collect();
    let late: Vec<_> = (0..LATE_AMOUNT)
        .map(|i| random_env(LATE_PLY + i % 2, &mut actions, rng))
        .collect();

    // Log the reference games.
    log::info!(
        "Early RND reference games:\n{}",
        reference_games_string(&early)
    );
    log::info!(
        "Late RND reference games:\n{}",
        reference_games_string(&late)
    );

    let early = to_reference(early, device);
    let late = to_reference(late, device);
    (early, late)
}

pub fn update_rnd(net: &mut Net, early: &Tensor, late: &Tensor) {
    let min = net.forward_rnd(early, false).min();
    let max = net.forward_rnd(late, false).max();
    net.update_rnd_normalization(&min, &max);
}
