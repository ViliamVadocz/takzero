use std::{
    array,
    collections::VecDeque,
    fs::OpenOptions,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicU32, Ordering},
        RwLock,
    },
};

use arrayvec::ArrayVec;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use takzero::{
    network::Network,
    search::{
        agent::Agent,
        env::Environment,
        eval::Eval,
        node::{gumbel::gumbel_sequential_halving, Node},
        DISCOUNT_FACTOR,
    },
    target::{Replay, Target},
};
use tch::Device;

use crate::{
    new_opening,
    Env,
    Net,
    ReplayBuffer,
    SharedNet,
    MAXIMUM_EXPLOITATION_BUFFER_SIZE,
    MAXIMUM_REPLAY_BUFFER_SIZE,
    STEP,
};

pub const EXPLORATION_BATCH_SIZE: usize = 256;
pub const EXPLOITATION_BATCH_SIZE: usize = 512;
pub const SAMPLED: usize = 64;
pub const SIMULATIONS: u32 = 1024;

pub const WEIGHTED_RANDOM_PLIES: u16 = 30;

pub const GREEDY_AGENTS: usize = 1; // no noise
pub const BASELINE_AGENTS: usize = EXPLOITATION_BATCH_SIZE - GREEDY_AGENTS;
pub const LOW_BETA_AGENTS: usize = EXPLORATION_BATCH_SIZE / 2;
pub const HIGH_BETA_AGENTS: usize = EXPLORATION_BATCH_SIZE / 2;

#[allow(clippy::assertions_on_constants)]
const _: () = assert!(
    GREEDY_AGENTS + BASELINE_AGENTS == EXPLOITATION_BATCH_SIZE
        && LOW_BETA_AGENTS + HIGH_BETA_AGENTS == EXPLORATION_BATCH_SIZE
);

pub const TRAINING_STEPS_BEFORE_BETA: u32 = 2_000;
pub const LOW_BETA: f32 = 0.02;
pub const HIGH_BETA: f32 = 0.2;
pub const EXPLOITATION_STEP: usize = 20;

#[allow(clippy::too_many_lines)]
pub fn exploitation(
    device: Device,
    seed: u64,
    shared_net: &SharedNet,
    exploitation_buffer: &RwLock<VecDeque<Target<Env>>>,
    exploration_buffer: &RwLock<VecDeque<Replay<Env>>>,
    training_steps: &AtomicU32,
    replay_path: &Path,
) {
    log::debug!("started exploitation self-play thread");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let chacha_seed = rng.gen();

    // Copy shared network weights to the network on this thread.
    let mut net = Net::new(device, None);
    let mut net_index = shared_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
    let mut context = <Net as Agent<Env>>::Context::new(*shared_net.2.read().unwrap());

    let mut envs: [_; EXPLOITATION_BATCH_SIZE] = array::from_fn(|_| Env::default());
    let mut nodes: [_; EXPLOITATION_BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut replays_batch: [_; EXPLOITATION_BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut target_batch: [_; EXPLOITATION_BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; EXPLOITATION_BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; EXPLOITATION_BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut rngs: [_; EXPLOITATION_BATCH_SIZE] = array::from_fn(|i| {
        let mut rng = ChaCha8Rng::from_seed(chacha_seed);
        rng.set_stream(i as u64);
        rng
    });

    let zero_betas = [0.0; EXPLOITATION_BATCH_SIZE];

    let mut temp_target_buffer = VecDeque::new();
    let mut temp_replay_buffer = VecDeque::new();
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, &mut rng));

    loop {
        log::info!("exploit search");
        // Do Gumbel sequential halving.
        let mut top_actions = gumbel_sequential_halving(
            &mut nodes,
            &envs,
            &net,
            SAMPLED,
            SIMULATIONS,
            &zero_betas,
            &mut context,
            &mut actions,
            &mut trajectories,
            Some(&mut rng),
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

        // Update replays and targets.
        replays_batch
            .iter_mut()
            .zip(target_batch.iter_mut())
            .zip(envs.iter())
            .zip(nodes.iter())
            .zip(&top_actions)
            .for_each(|((((replays, targets), env), node), action)| {
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

                // Push start of fresh target.
                targets.push(Target {
                    env: env.clone(),
                    policy: node
                        .improved_policy(
                            #[cfg(not(feature = "baseline"))]
                            0.0,
                        )
                        .zip(node.children.iter())
                        .map(|(p, (a, _))| (*a, p))
                        .collect(),
                    value: f32::NAN, // Value still needs to be filled.
                    #[cfg(not(feature = "baseline"))]
                    ube: f32::NAN, // UBE still needs to be filled.
                });
                // Update N-step targets.
                if targets.len() > EXPLOITATION_STEP {
                    let index = targets.len().saturating_sub(EXPLOITATION_STEP + 1);
                    let target = &mut targets[index];

                    // Assign the target value.
                    let sign = if EXPLOITATION_STEP % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    target.value = sign
                        * DISCOUNT_FACTOR.powi(i32::try_from(EXPLOITATION_STEP).unwrap())
                        * node.evaluation.ply().map_or_else(
                            || f32::from(node.evaluation),
                            |ply| {
                                DISCOUNT_FACTOR.powi(i32::try_from(ply).unwrap())
                                    * f32::from(node.evaluation)
                            },
                        );
                    #[cfg(not(feature = "baseline"))]
                    {
                        // Assign the uncertainty target.
                        target.ube = DISCOUNT_FACTOR
                            .powi(2 * i32::try_from(EXPLOITATION_STEP).unwrap())
                            * f32::from(node.variance);
                    }
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

        // Deal with finished games.
        envs.iter_mut()
            .zip(nodes.iter_mut())
            .zip(actions.iter_mut())
            .zip(replays_batch.iter_mut())
            .zip(target_batch.iter_mut())
            .for_each(|((((env, node), actions), replays), targets)| {
                if let Some(result) = env.terminal() {
                    let mut value = f32::from(Eval::from(result));
                    temp_target_buffer.extend(targets.drain(..).rev().map(|mut target| {
                        if target.value.is_nan() {
                            // Complete value target
                            value *= -DISCOUNT_FACTOR;
                            target.value = value;
                            #[cfg(not(feature = "baseline"))]
                            {
                                // TODO: Is this supposed to be 0?
                                target.ube = 0.0;
                            }
                        }
                        target
                    }));
                    temp_replay_buffer.extend(replays.drain(..));

                    new_opening(env, actions, &mut rng);
                    *node = Node::default();
                }
            });

        // Add replays to buffer.
        if !temp_replay_buffer.is_empty() {
            // // for exploration we discard these replays.
            // temp_replay_buffer.clear();

            let steps = training_steps.load(Ordering::Relaxed);
            log::info!(
                "Adding {} replays at {steps} training steps",
                temp_replay_buffer.len()
            );
            let mut lock = exploration_buffer.write().unwrap();
            lock.append(&mut temp_replay_buffer);

            // Truncate replay buffer if it gets too long.
            if lock.len() > MAXIMUM_REPLAY_BUFFER_SIZE {
                log::info!("truncating replay buffer, len = {}", lock.len());
                lock.truncate(MAXIMUM_REPLAY_BUFFER_SIZE);
            }

            // While doing this, also save the replay buffer
            let s: String = lock.iter().map(ToString::to_string).collect();
            drop(lock);
            let path = replay_path.join(format!("replays_{steps:0>6}.txt"));
            std::thread::spawn(move || {
                match OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)
                {
                    Ok(mut file) => file.write_all(s.as_bytes()).unwrap(),
                    Err(err) => log::error!("Could not save replays: {err}"),
                }
            });
            log::info!("saved replays to file");
        }

        if !temp_target_buffer.is_empty() {
            let steps = training_steps.load(Ordering::Relaxed);
            log::info!(
                "Adding {} targets at {steps} training steps",
                temp_target_buffer.len()
            );
            let mut lock = exploitation_buffer.write().unwrap();
            lock.append(&mut temp_target_buffer);

            // Truncate target buffer if it gets too long.
            if lock.len() > MAXIMUM_EXPLOITATION_BUFFER_SIZE {
                log::info!("truncating target buffer, len = {}", lock.len());
                lock.truncate(MAXIMUM_EXPLOITATION_BUFFER_SIZE);
            }

            // While doing this, also save the target buffer
            let s: String = lock.iter().map(ToString::to_string).collect();
            drop(lock);
            let path = replay_path.join(format!("exploitation_targets_{steps:0>6}.txt"));
            std::thread::spawn(move || {
                match OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)
                {
                    Ok(mut file) => file.write_all(s.as_bytes()).unwrap(),
                    Err(err) => log::error!("Could not save replays: {err}"),
                }
            });
            log::info!("saved targets to file");
        }

        //  Get the latest network
        log::debug!("checking if there is a new model for self-play");
        let maybe_new_net_index = shared_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
            log::info!("updating self-play model to shared_net_{net_index}");

            context = <Net as Agent<Env>>::Context::new(*shared_net.2.read().unwrap());
        }

        if cfg!(test) {
            break;
        }
    }
}

// TODO: Consider beta scheduler?

#[allow(clippy::too_many_lines)]
pub fn exploration(
    device: Device,
    seed: u64,
    shared_net: &SharedNet,
    exploration_buffer: &ReplayBuffer,
    training_steps: &AtomicU32,
    replay_path: &Path,
) {
    log::debug!("started exploration self-play thread");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let chacha_seed = rng.gen();

    // Copy shared network weights to the network on this thread.
    let mut net = Net::new(device, None);
    let mut net_index = shared_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
    let mut context = <Net as Agent<Env>>::Context::new(*shared_net.2.read().unwrap());

    let mut envs: [_; EXPLORATION_BATCH_SIZE] = array::from_fn(|_| Env::default());
    let mut nodes: [_; EXPLORATION_BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut replays_batch: [_; EXPLORATION_BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; EXPLORATION_BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; EXPLORATION_BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut rngs: [_; EXPLORATION_BATCH_SIZE] = array::from_fn(|i| {
        let mut rng = ChaCha8Rng::from_seed(chacha_seed);
        rng.set_stream(i as u64);
        rng
    });

    let betas: Vec<f32> = [[LOW_BETA; LOW_BETA_AGENTS], [HIGH_BETA; HIGH_BETA_AGENTS]].concat();
    let zero_betas = [0.0f32; EXPLORATION_BATCH_SIZE];

    let mut temp_replay_buffer = VecDeque::new();
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, &mut rng));

    loop {
        // Do Gumbel sequential halving.
        let mut top_actions = gumbel_sequential_halving(
            &mut nodes,
            &envs,
            &net,
            SAMPLED,
            SIMULATIONS,
            if training_steps.load(Ordering::Relaxed) > TRAINING_STEPS_BEFORE_BETA {
                log::info!("search with beta");
                &betas
            } else {
                log::info!("search without beta");
                &zero_betas
            },
            &mut context,
            &mut actions,
            &mut trajectories,
            Some(&mut rng),
        );
        // For openings, sample actions according to visits instead.
        envs.iter()
            .zip(rngs.iter_mut())
            .zip(nodes.iter_mut())
            .zip(&mut top_actions)
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
                    new_opening(env, actions, &mut rng);
                    *node = Node::default();
                    replays.drain(..)
                })
            })
            .flatten()
            .for_each(|replay| temp_replay_buffer.push_front(replay));

        // Add replays to buffer.
        if !temp_replay_buffer.is_empty() {
            let steps = training_steps.load(Ordering::Relaxed);
            log::info!(
                "Adding {} replays at {steps} training steps",
                temp_replay_buffer.len()
            );
            let mut lock = exploration_buffer.write().unwrap();
            lock.append(&mut temp_replay_buffer);

            // Truncate replay buffer if it gets too long.
            if lock.len() > MAXIMUM_REPLAY_BUFFER_SIZE {
                log::info!("truncating replay buffer, len = {}", lock.len());
                lock.truncate(MAXIMUM_REPLAY_BUFFER_SIZE);
            }

            // While doing this, also save the replay buffer
            let s: String = lock.iter().map(ToString::to_string).collect();
            drop(lock);
            let path = replay_path.join(format!("replays_{steps:0>6}.txt"));
            std::thread::spawn(move || {
                match OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)
                {
                    Ok(mut file) => file.write_all(s.as_bytes()).unwrap(),
                    Err(err) => log::error!("Could not save replays: {err}"),
                }
            });
            log::info!("saved replays to file");
        }

        //  Get the latest network
        log::debug!("checking if there is a new model for self-play");
        let maybe_new_net_index = shared_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
            log::info!("updating self-play model to shared_net_{net_index}");

            context = <Net as Agent<Env>>::Context::new(*shared_net.2.read().unwrap());
        }

        if cfg!(test) {
            break;
        }
    }
}
