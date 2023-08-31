use std::{
    array,
    collections::VecDeque,
    fmt,
    fs::OpenOptions,
    io::Write,
    path::Path,
    sync::{atomic::Ordering, RwLock},
};

use arrayvec::ArrayVec;
use rand::{
    distributions::WeightedIndex,
    prelude::Distribution,
    seq::IteratorRandom,
    Rng,
    SeedableRng,
};
use rand_chacha::ChaCha8Rng;
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

use crate::{target::Replay, BetaNet, MAXIMUM_REPLAY_BUFFER_SIZE, STEP};

const BATCH_SIZE: usize = 64;

const SAMPLED: usize = 16;
const SIMULATIONS: u32 = 512;

const STEPS_BEFORE_CHECKING_NETWORK: usize = 400; // TODO: Think more about this number

const RANDOM_GAMES: u32 = 8;
const WEIGHTED_RANDOM_PLIES: u16 = 30;

/// Populate the replay buffer with new state-action pairs from self-play.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    replay_queue: &RwLock<VecDeque<Replay<E>>>,
    replay_path: &Path,
    primary: bool,
) where
    Replay<E>: fmt::Display,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let chacha_seed = rng.gen();

    let mut net = NET::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| E::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut replays_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut rngs: [_; BATCH_SIZE] = array::from_fn(|i| {
        let mut rng = ChaCha8Rng::from_seed(chacha_seed);
        rng.set_stream(i as u64);
        rng
    });

    // Initialize replay buffer with random moves.
    if primary {
        replay_queue.write().unwrap().extend(
            envs.par_iter_mut()
                .zip(&mut replays_batch)
                .zip(&mut actions)
                .zip(&mut rngs)
                .map(|(((env, replays), actions), rng)| {
                    let mut replays_buffer = Vec::new();
                    for _ in 0..RANDOM_GAMES {
                        while env.terminal().is_none() {
                            // Choose random action.
                            env.populate_actions(actions);
                            let action = actions.drain(..).choose(rng).unwrap();
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
                            // Take a step in the environment.
                            env.step(action);
                        }
                        replays_buffer.append(replays);
                        *env = E::default();
                    }
                    replays_buffer.into_par_iter()
                })
                .flatten()
                .collect::<Vec<_>>(),
        );
    }

    loop {
        self_play(
            &mut rng,
            &mut rngs,
            &net,
            &mut envs,
            &mut nodes,
            &mut replays_batch,
            &mut actions,
            &mut trajectories,
            replay_queue,
        );

        // Truncate replay queue if it gets too long.
        let mut lock = replay_queue.write().unwrap();
        if lock.len() > MAXIMUM_REPLAY_BUFFER_SIZE {
            lock.truncate(MAXIMUM_REPLAY_BUFFER_SIZE);
        }
        drop(lock);

        //  Get the latest network
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
            log::info!("Updating self-play model to beta{net_index}");

            // While doing this, also save the replay buffer
            if primary {
                let s: String = replay_queue
                    .read()
                    .unwrap()
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                let path = replay_path.join("replays.txt");
                rayon::spawn(move || {
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(path)
                        .expect("replay file path should be valid and writable");
                    file.write_all(s.as_bytes()).unwrap();
                });
            }
        }

        if cfg!(test) {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn self_play<E: Environment, A: Agent<E>>(
    rng: &mut impl Rng,
    rngs: &mut [ChaCha8Rng],
    agent: &A,

    envs: &mut [E],
    nodes: &mut [Node<E>],
    replays_batch: &mut [Vec<Replay<E>>],
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],

    replay_queue: &RwLock<VecDeque<Replay<E>>>,
) {
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, rng));
    nodes
        .par_iter_mut()
        .for_each(|node| *node = Node::default());

    for _ in 0..STEPS_BEFORE_CHECKING_NETWORK {
        let mut top_actions = gumbel_sequential_halving(
            nodes,
            envs,
            agent,
            SAMPLED,
            SIMULATIONS,
            actions,
            trajectories,
            Some(rng),
        );
        // For openings, sample actions according to visits instead.
        envs.par_iter()
            .zip(rngs.par_iter_mut())
            .zip(nodes.par_iter_mut())
            .zip(&mut top_actions)
            .filter(|(((env, _), _), _)| env.steps() < WEIGHTED_RANDOM_PLIES)
            .for_each(|(((_, rng), node), top_action)| {
                let weighted_index =
                    WeightedIndex::new(node.children.iter().map(|(_, child)| child.visit_count))
                        .unwrap();
                *top_action = node.children[weighted_index.sample(rng)].0.clone();
            });

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
            .for_each(|replay| replay_queue.write().unwrap().push_front(replay));
    }

    // Salvage replays from unfinished games.
    for replays in replays_batch {
        let len = replays.len().saturating_sub(STEP);
        replays
            .drain(..)
            .take(len)
            .for_each(|replay| replay_queue.write().unwrap().push_front(replay));
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
    use std::{
        collections::VecDeque,
        path::PathBuf,
        sync::{atomic::AtomicUsize, RwLock},
    };

    use rand::{Rng, SeedableRng};
    use takzero::network::{net3::Net3, Network};
    use tch::Device;

    use crate::{self_play::run, BetaNet};

    // NOTE TO SELF:
    // Decrease constants above to actually see results before you die.
    #[test]
    fn self_play_works() {
        const SEED: u64 = 1234;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net3::new(Device::Cpu, Some(rng.gen()));
        let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

        let replay_queue = RwLock::new(VecDeque::new());

        run::<_, Net3>(
            Device::cuda_if_available(),
            rng.gen(),
            &beta_net,
            &replay_queue,
            &PathBuf::default(),
            true,
        );

        for replay in &*replay_queue.read().unwrap() {
            println!("{replay}");
        }
    }
}
