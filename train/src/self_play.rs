use crossbeam::channel::Sender;
use fast_tak::{takparse::Move, Game, Reserves};
use rayon::prelude::IntoParallelRefMutIterator;
use takzero::network::Network;

use crate::target::{Replay, Target};

const BATCH_SIZE: usize = 64;

/// Populate the replay buffer with new state-action pairs from self-play.
pub fn run<const N: usize, const HALF_KOMI: i8>(tx: Sender<Replay<N, HALF_KOMI>>) {
    let mut actions = [(); BATCH_SIZE].map(|()| Vec::new());
    let mut targets = [(); BATCH_SIZE].map(|()| Vec::new());
    loop {
        let net = todo!(); // TODO: Get a recent network

        self_play(&net, &mut actions, &mut targets);

        // TODO: Send targets
        tx.send(todo!());
    }
}

/// Play a batch of self-play games.
fn self_play<const N: usize, const HALF_KOMI: i8, NET: Network>(
    net: &NET,
    actions: &mut [Vec<Move>],
    targets: &mut [Vec<Target<N, HALF_KOMI>>],
) where
    Reserves<N>: Default,
{
    let mut games: [Game<N, HALF_KOMI>; BATCH_SIZE] = [(); BATCH_SIZE].map(|()| Game::default());
    let mut nodes: [Node<Game<N, HALF_KOMI>>; BATCH_SIZE] =
        [(); BATCH_SIZE].map(|()| Node::default());

    let batch = nodes
        .par_iter_mut()
        .map(|node| node.gumbel_sequential_halving())
        .collect();
    let (policy, value) = net.forward_t(batch, false);
    nodes.par_iter_mut().for_each(|node| node.complete());

    // TODO:
    // - Create games
    // - Until all are done, (replenish or not?)

    // This should be elsewhere:
    // - Do rollout until evaluation is needed
    // - Then network eval the whole batch
    // - Propagate result through tree

    // - Choose best actions
    // - Play actions
    // - Save target/replay
}
