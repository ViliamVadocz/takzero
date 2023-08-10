use crossbeam::channel::Receiver;

use crate::target::Target;

/// Improve the network by training on batches from the re-analyze thread.
/// Save checkpoints and distribute the newest model.
pub fn run<const N: usize, const HALF_KOMI: i8>(rx: Receiver<Box<[Target<N, HALF_KOMI>]>>) {
    while let Ok(batch) = rx.recv() {}
}
