use crossbeam::channel::Receiver;
use rand::SeedableRng;
use takzero::{
    network::Network,
    search::{agent::Agent, env::Environment},
};

use crate::{target::Target, BetaNet};

/// Improve the network by training on batches from the re-analyze thread.
/// Save checkpoints and distribute the newest model.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    seed: u64,
    beta_net: &BetaNet,
    rx: Receiver<Box<[Target<E>]>>,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    while let Ok(batch) = rx.recv() {}
}
