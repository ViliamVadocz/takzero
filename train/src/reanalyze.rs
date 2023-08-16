use std::{collections::VecDeque, sync::atomic::Ordering};

use crossbeam::channel::{Receiver, Sender};
use rand::SeedableRng;
use takzero::{
    network::Network,
    search::{agent::Agent, env::Environment},
};
use tch::Device;

use crate::{
    target::{Replay, Target},
    BetaNet,
};

const BATCH_SIZE: usize = 256;

/// Collect new state-action replays from self-play
/// and generate batches for training.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    rx: Receiver<Replay<E>>,
    tx: Sender<Vec<Target<E>>>,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = NET::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    // TODO: Allocate with specific capacity.
    let mut target_queue: VecDeque<Target<E>> = VecDeque::new();
    let mut replay_queue = VecDeque::new();

    loop {
        // Receive until the replay channel is empty.
        // FIXME: If the self-play thread generates replays too fast
        // this can loop without generating any new batches
        while let Ok(replays) = rx.recv() {
            replay_queue.push_back(replays);
        }

        // let sample: [Replay<E>; BATCH_SIZE] = todo!();
    }
}
