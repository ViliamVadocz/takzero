use fast_tak::takparse::Tps;
use takzero::network::{repr::game_to_tensor, Network};
use tch::{Device, Tensor};

use super::{Env, Net, HALF_KOMI, N};

fn to_reference(tps_str: &str, device: Device) -> Tensor {
    Tensor::concat(
        &tps_str
            .lines()
            .map(|line| line.parse().expect("Reference games should be valid TPS"))
            .map(|tps: Tps| Env::from(tps))
            .map(|env| game_to_tensor::<N, HALF_KOMI>(&env, device))
            .collect::<Vec<_>>(),
        0,
    )
}

pub fn reference_games(device: Device) -> (Tensor, Tensor) {
    let early = to_reference(include_str!("./early.tps"), device);
    let late = to_reference(include_str!("./late.tps"), device);
    (early, late)
}

pub fn update_rnd(net: &mut Net, early: &Tensor, late: &Tensor) {
    let min = net.forward_rnd(early, false).min();
    let max = net.forward_rnd(late, false).max();
    net.update_rnd_normalization(&min, &max);
}
