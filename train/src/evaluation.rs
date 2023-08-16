use std::sync::atomic::Ordering;

use rand::SeedableRng;
use takzero::{
    network::Network,
    search::{agent::Agent, env::Environment},
};
use tch::Device;

use crate::BetaNet;

/// Evaluate checkpoints of the network to make sure that it is improving.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
) -> ! {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = NET::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    loop {
        todo!()
    }
}
