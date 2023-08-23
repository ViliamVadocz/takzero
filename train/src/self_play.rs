use std::{array, sync::atomic::Ordering};

use arrayvec::ArrayVec;
use crossbeam::channel::Sender;
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use rayon::prelude::*;
use takzero::{
    network::Network,
    search::{
        agent::Agent,
        env::Environment,
        node::{gumbel::gumbel_sequential_halving, Node},
    },
};
use tch::Device;

use crate::{target::Replay, BetaNet, STEP};

const BATCH_SIZE: usize = 128;
const SAMPLED: usize = 16;
const SIMULATIONS: u32 = 1024;
const STEPS_BEFORE_CHECKING_NETWORK: usize = 1_000; // TODO: Think more about this number

/// Populate the replay buffer with new state-action pairs from self-play.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    mut tx: Sender<Replay<E>>,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = NET::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| E::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut replays_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());

    loop {
        self_play(
            &mut rng,
            &net,
            &mut envs,
            &mut nodes,
            &mut replays_batch,
            &mut actions,
            &mut trajectories,
            &mut tx,
        );

        //  Get the latest network
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
            println!("Updating self-play model to beta{net_index}");
        }

        if cfg!(test) {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn self_play<E: Environment, A: Agent<E>>(
    rng: &mut impl Rng,
    agent: &A,

    envs: &mut [E],
    nodes: &mut [Node<E>],
    replays_batch: &mut [Vec<Replay<E>>],
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],

    tx: &mut Sender<Replay<E>>,
) {
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, rng));
    nodes
        .par_iter_mut()
        .for_each(|node| *node = Node::default());

    for _ in 0..STEPS_BEFORE_CHECKING_NETWORK {
        let top_actions = gumbel_sequential_halving(
            nodes,
            envs,
            agent,
            SAMPLED,
            SIMULATIONS,
            actions,
            trajectories,
            Some(rng),
        );

        // Update replays.
        replays_batch
            .par_iter_mut()
            .zip(envs.par_iter())
            .zip(&top_actions)
            .for_each(|((replays, env), action)| {
                // Push start of fresh replay.
                replays.push(Replay {
                    env: env.clone(),
                    actions: ArrayVec::default(),
                });
                // Update existing replays.
                let from = replays.len().saturating_sub(STEP);
                for replay in &mut replays[from..] {
                    replay.actions.push(action.clone());
                }
            });

        // Take a step in environments and nodes.
        nodes
            .par_iter_mut()
            .zip(envs.par_iter_mut())
            .zip(top_actions)
            .for_each(|((node, env), action)| {
                node.descend(&action);
                env.step(action);
            });

        // Refresh finished environments and nodes.
        replays_batch
            .iter_mut()
            .zip(nodes.iter_mut())
            .zip(envs.iter_mut())
            .zip(actions.iter_mut())
            .filter_map(|(((replays, node), env), actions)| {
                env.terminal().map(|_| {
                    new_opening(env, actions, rng);
                    *node = Node::default();
                    replays.drain(..)
                })
            })
            .flatten()
            .try_for_each(|replay| tx.send(replay))
            .unwrap();
    }

    // Salvage replays from unfinished games.
    for replays in replays_batch {
        let len = replays.len().saturating_sub(STEP);
        replays
            .drain(..)
            .take(len)
            .try_for_each(|replay| tx.send(replay))
            .unwrap();
    }
}

fn new_opening<E: Environment>(env: &mut E, actions: &mut Vec<E::Action>, rng: &mut impl Rng) {
    *env = E::default();
    for _ in 0..2 {
        env.populate_actions(actions);
        env.step(actions.drain(..).choose(rng).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicUsize, RwLock};

    use rand::{Rng, SeedableRng};
    use takzero::{
        fast_tak::Game,
        network::{net3::Net3, Network},
    };
    use tch::Device;

    use crate::{self_play::run, target::Replay, BetaNet};

    // NOTE TO SELF:
    // Decrease constants above to actually see results before you die.
    #[test]
    fn self_play_works() {
        const SEED: u64 = 1234;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net3::new(Device::Cpu, Some(rng.gen()));
        let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

        let (replay_tx, replay_rx) = crossbeam::channel::unbounded::<Replay<Game<3, 0>>>();

        run::<_, Net3>(Device::cuda_if_available(), rng.gen(), &beta_net, replay_tx);
        while let Ok(replay) = replay_rx.recv() {
            println!("{replay}");
        }
    }
}
