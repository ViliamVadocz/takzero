use std::collections::VecDeque;

use crossbeam::channel::{Receiver, Sender};

use crate::target::{Replay, Target};

/// Collect new state-action replays from self-play
/// and generate batches for training.
pub fn run<const N: usize, const HALF_KOMI: i8>(
    rx: Receiver<Replay<N, HALF_KOMI>>,
    tx: Sender<Box<[Target<N, HALF_KOMI>]>>,
) {
    // TODO: Allocate with specific capacity.
    let mut target_queue: VecDeque<Target<N, HALF_KOMI>> = VecDeque::new();
    let mut replay_queue = VecDeque::new();

    while let Ok(replays) = rx.recv() {
        replay_queue.push_back(replays);
    }
}
