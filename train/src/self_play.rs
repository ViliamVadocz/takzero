use std::{
    array,
    collections::VecDeque,
    fs::OpenOptions,
    io::Write,
    path::Path,
    sync::atomic::{AtomicU32, Ordering},
};

use arrayvec::ArrayVec;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use takzero::{
    network::Network,
    search::{
        agent::Agent,
        env::Environment,
        node::{gumbel::gumbel_sequential_halving, Node},
    },
    target::Replay,
};
use tch::Device;

use crate::{new_opening, Env, Net, ReplayBuffer, SharedNet, MAXIMUM_REPLAY_BUFFER_SIZE, STEP};

pub const BATCH_SIZE: usize = 32;
pub const SAMPLED: usize = 64;
pub const SIMULATIONS: u32 = 1024;

pub const NEW_REPLAYS_PER_TRAINING_STEP: u32 = 10; // TODO: Currently not enforced

// This number should be large enough that there are also late-game positions.
pub const STEPS_BEFORE_CHECKING_NETWORK: usize = 500;

pub const WEIGHTED_RANDOM_PLIES: u16 = 30;

pub const GREEDY_AGENTS: usize = 1;
pub const BASELINE_AGENTS: usize = BATCH_SIZE / 2 - GREEDY_AGENTS;
pub const LOW_BETA_AGENTS: usize = BATCH_SIZE / 4;
pub const HIGH_BETA_AGENTS: usize = BATCH_SIZE / 4;
pub const LOW_BETA: f32 = 0.02;
pub const HIGH_BETA: f32 = 0.2;

#[allow(clippy::assertions_on_constants)]
const _: () =
    assert!(GREEDY_AGENTS + BASELINE_AGENTS + LOW_BETA_AGENTS + HIGH_BETA_AGENTS == BATCH_SIZE);

/// Populate the replay buffer with new state-action pairs from self-play.
pub fn run(
    device: Device,
    seed: u64,
    shared_net: &SharedNet,
    replay_buffer: &ReplayBuffer,
    training_steps: &AtomicU32,
    replay_path: &Path,
) {
    log::debug!("started self-play thread");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let chacha_seed = rng.gen();

    // Copy shared network weights to the network on this thread.
    let mut net = Net::new(device, None);
    let mut net_index = shared_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| Env::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut replays_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut rngs: [_; BATCH_SIZE] = array::from_fn(|i| {
        let mut rng = ChaCha8Rng::from_seed(chacha_seed);
        rng.set_stream(i as u64);
        rng
    });

    let betas: Vec<f32> = [
        vec![0.0; GREEDY_AGENTS],
        vec![0.0; BASELINE_AGENTS],
        vec![LOW_BETA; LOW_BETA_AGENTS],
        vec![HIGH_BETA; HIGH_BETA_AGENTS],
    ]
    .concat();

    loop {
        self_play(
            &mut rng,
            &mut rngs,
            &net,
            &betas,
            &mut envs,
            &mut nodes,
            &mut replays_batch,
            &mut actions,
            &mut trajectories,
            replay_buffer,
            training_steps,
        );

        // Truncate replay buffer if it gets too long.
        let mut lock = replay_buffer.write().unwrap();
        if lock.len() > MAXIMUM_REPLAY_BUFFER_SIZE {
            log::info!("truncating replay buffer, len = {}", lock.len());
            lock.truncate(MAXIMUM_REPLAY_BUFFER_SIZE);
        }
        drop(lock);

        //  Get the latest network
        log::info!("checking if there is a new model for self-play");
        let maybe_new_net_index = shared_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
            log::info!("updating self-play model to shared_net_{net_index}");

            // While doing this, also save the replay buffer
            let s: String = replay_buffer
                .read()
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .collect();
            let path = replay_path.join("replays.txt");
            std::thread::spawn(move || {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)
                    .expect("replay file path should be valid and writable");
                file.write_all(s.as_bytes()).unwrap();
            });
            log::info!("saved replays to file");
        }

        if cfg!(test) {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn self_play(
    rng: &mut impl Rng,
    rngs: &mut [ChaCha8Rng],
    agent: &Net,
    betas: &[f32],

    envs: &mut [Env],
    nodes: &mut [Node<Env>],
    replays_batch: &mut [Vec<Replay<Env>>],
    actions: &mut [Vec<<Env as Environment>::Action>],
    trajectories: &mut [Vec<usize>],

    replay_buffer: &ReplayBuffer,
    training_steps: &AtomicU32,
) {
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, rng));
    nodes.iter_mut().for_each(|node| *node = Node::default());

    let mut temp_replay_buffer = VecDeque::new();
    for _ in 0..STEPS_BEFORE_CHECKING_NETWORK {
        log::info!("step");
        let mut context: Vec<_> = (0..nodes.len())
            .map(|_| <Net as Agent<Env>>::Context::default())
            .collect();
        let mut top_actions = gumbel_sequential_halving(
            nodes,
            envs,
            agent,
            SAMPLED,
            SIMULATIONS,
            betas,
            &mut context,
            actions,
            trajectories,
            Some(rng),
        );
        // For openings, sample actions according to visits instead.
        envs.iter()
            .zip(rngs.iter_mut())
            .zip(nodes.iter_mut())
            .zip(&mut top_actions)
            .skip(GREEDY_AGENTS)
            // .take(BASELINE_AGENTS)
            .filter(|(((env, _), _), _)| env.steps() < WEIGHTED_RANDOM_PLIES)
            .for_each(|(((_, rng), node), top_action)| {
                let weighted_index =
                    WeightedIndex::new(node.children.iter().map(|(_, child)| child.visit_count))
                        .unwrap();
                *top_action = node.children[weighted_index.sample(rng)].0;
            });

        // Update replays.
        replays_batch
            .iter_mut()
            .zip(envs.iter())
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
                    replay.actions.push(*action);
                }
            });

        // Take a step in environments and nodes.
        nodes
            .iter_mut()
            .zip(envs.iter_mut())
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
            .for_each(|replay| temp_replay_buffer.push_front(replay));

        let steps = training_steps.load(Ordering::Relaxed);
        log::info!(
            "Adding {} replays at {steps} training steps",
            temp_replay_buffer.len()
        );
        replay_buffer
            .write()
            .unwrap()
            .append(&mut temp_replay_buffer);
    }

    // Salvage replays from unfinished games.
    for replays in replays_batch {
        let len = replays.len().saturating_sub(STEP);
        replays
            .drain(..)
            .take(len)
            .for_each(|replay| temp_replay_buffer.push_front(replay));
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        path::PathBuf,
        sync::{
            atomic::{AtomicU32, AtomicUsize},
            RwLock,
        },
    };

    use rand::{Rng, SeedableRng};
    use takzero::network::Network;
    use tch::Device;

    use crate::{self_play::run, Net, SharedNet};

    // NOTE TO SELF:
    // Decrease constants above to actually see results before you die.
    #[test]
    fn self_play_works() {
        const SEED: u64 = 1234;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net::new(Device::Cpu, Some(rng.gen()));
        let shared_net: SharedNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

        let replay_buffer = RwLock::new(VecDeque::new());

        run(
            Device::cuda_if_available(),
            rng.gen(),
            &shared_net,
            &replay_buffer,
            &AtomicU32::new(0),
            &PathBuf::default(),
        );

        for replay in &*replay_buffer.read().unwrap() {
            println!("{replay}");
        }
    }
}
