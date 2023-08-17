use std::{array, path::PathBuf, sync::atomic::Ordering};

use crossbeam::channel::Sender;
use rand::{Rng, SeedableRng};
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

const BATCH_SIZE: usize = 64;
const SAMPLED: usize = 32;
const SIMULATIONS: u32 = 1024;
const STEPS_BEFORE_CHECKING_NETWORK: usize = 100_000; // TODO: Think more about this number

// TODO: Save replays
// - figure out what format to use
// - figure out where and when
// - all in one file? multiple files?

/// Populate the replay buffer with new state-action pairs from self-play.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    mut tx: Sender<Replay<E>>,
    replay_path: PathBuf,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = NET::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    let mut replays: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());

    loop {
        // TODO: self-play
        self_play(
            &mut rng,
            &net,
            &mut actions,
            &mut replays,
            &mut trajectories,
            &mut tx,
        );

        //  Get the latest network
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index >= net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
        }

        if cfg!(test) {
            break;
        }
    }
}

fn self_play<E: Environment, A: Agent<E>>(
    rng: &mut impl Rng,
    agent: &A,

    replays_batch: &mut [Vec<Replay<E>>],
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],

    tx: &mut Sender<Replay<E>>,
) {
    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| E::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());

    for _ in 0..STEPS_BEFORE_CHECKING_NETWORK {
        let top_actions = gumbel_sequential_halving(
            &mut nodes,
            &envs,
            agent,
            rng,
            SAMPLED,
            SIMULATIONS,
            actions,
            trajectories,
        );

        // Update replays.
        replays_batch
            .par_iter_mut()
            .zip(&envs)
            .zip(&top_actions)
            .for_each(|((replays, env), action)| {
                // Push start of fresh replay.
                replays.push(Replay {
                    env: env.clone(),
                    actions: Default::default(),
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
            .zip(&mut envs)
            .zip(top_actions)
            .for_each(|((node, env), action)| {
                node.descend(&action);
                env.step(action);
            });

        // Refresh finished environments and nodes.
        replays_batch
            .iter_mut()
            .zip(&mut nodes)
            .zip(&mut envs)
            .filter_map(|((replays, node), env)| {
                env.terminal().map(|_| {
                    *env = E::default();
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

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicUsize, RwLock};

    use fast_tak::{takparse::Tps, Game};
    use rand::{Rng, SeedableRng};
    use takzero::network::{net3::Net3, Network};
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

        run::<_, Net3>(
            Device::cuda_if_available(),
            rng.gen(),
            &beta_net,
            replay_tx,
            Default::default(),
        );
        while let Ok(replay) = replay_rx.recv() {
            let tps: Tps = replay.env.into();
            println!(
                "{tps} {:?}",
                replay
                    .actions
                    .into_iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
            );
        }
    }
}
