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
    network::{repr::game_to_tensor, Network},
    search::{
        agent::Agent,
        env::Environment,
        eval::Eval,
        node::{gumbel::gumbel_sequential_halving, Node},
        DISCOUNT_FACTOR,
    },
    target::{Replay, Target},
};
use tch::{Device, Tensor};

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

pub const BATCH_SIZE: usize = 128;
pub const SAMPLED: usize = 64;
pub const SIMULATIONS: u32 = 1024;

pub const WEIGHTED_RANDOM_PLIES: u16 = 30;

pub const GREEDY_AGENTS: usize = 1; // no noise
pub const BASELINE_AGENTS: usize = BATCH_SIZE - GREEDY_AGENTS;
pub const LOW_BETA_AGENTS: usize = BATCH_SIZE / 2;
pub const HIGH_BETA_AGENTS: usize = BATCH_SIZE / 2;

#[allow(clippy::assertions_on_constants)]
const _: () = assert!(
    GREEDY_AGENTS + BASELINE_AGENTS == BATCH_SIZE
        && LOW_BETA_AGENTS + HIGH_BETA_AGENTS == BATCH_SIZE
);

pub const LOW_BETA: f32 = 0.02;
pub const HIGH_BETA: f32 = 0.2;
pub const EXPLOITATION_STEP: usize = 20;

#[allow(clippy::too_many_lines)]
pub fn exploitation(
    device: Device,
    seed: u64,
    shared_net: &SharedNet,
    exploitation_buffer: &RwLock<VecDeque<Target<Env>>>,
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

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| Env::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut replays_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut target_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut rngs: [_; BATCH_SIZE] = array::from_fn(|i| {
        let mut rng = ChaCha8Rng::from_seed(chacha_seed);
        rng.set_stream(i as u64);
        rng
    });

    let betas = [0.0; BATCH_SIZE];

    let mut temp_target_buffer = VecDeque::new();
    let mut temp_replay_buffer = VecDeque::new();
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, &mut rng));

    loop {
        log::info!("search");
        // Do Gumbel sequential halving.
        let mut top_actions = gumbel_sequential_halving(
            &mut nodes,
            &envs,
            &net,
            SAMPLED,
            SIMULATIONS,
            &betas,
            &mut context,
            &mut actions,
            &mut trajectories,
            Some(&mut rng),
        );
        let raw_rnd: Vec<f32> = context
            .normalize(
                &net.forward_rnd(
                    &Tensor::cat(
                        &envs
                            .iter()
                            .map(|env| game_to_tensor(env, device))
                            .collect::<Vec<_>>(),
                        0,
                    ),
                    false,
                ),
            )
            .try_into()
            .unwrap();

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
            .zip(raw_rnd)
            .for_each(|(((((replays, targets), env), node), action), rnd)| {
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
                        .improved_policy(0.0)
                        .zip(node.children.iter())
                        .map(|(p, (a, _))| (*a, p))
                        .collect(),
                    value: f32::NAN, // Value still needs to be filled.
                    ube: 0.0,        // Will be updated.
                });

                // Accumulate discounted RND.
                let from = targets.len().saturating_sub(EXPLOITATION_STEP);
                for (i, target) in targets[from..].iter_mut().enumerate() {
                    let step = EXPLOITATION_STEP - i; // TODO check if step correct
                    target.ube += DISCOUNT_FACTOR.powi(2 * i32::try_from(step).unwrap()) * rnd;
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                if targets.len() > EXPLOITATION_STEP {
                    let target = &mut targets[from]; // TODO: Check this index

                    // Assign the target value.
                    let sign = if EXPLOITATION_STEP % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    target.value = sign
                        * DISCOUNT_FACTOR.powi(EXPLOITATION_STEP as i32)
                        * node.evaluation.ply().map_or_else(
                            || f32::from(node.evaluation),
                            |ply| DISCOUNT_FACTOR.powi(ply as i32) * f32::from(node.evaluation),
                        );
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
                            target.value = value; // TODO: Check if correct
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
            // for exploration we discard these replays.
            temp_replay_buffer.clear();

            // let steps = training_steps.load(Ordering::Relaxed);
            // log::info!(
            //     "Adding {} replays at {steps} training steps",
            //     temp_replay_buffer.len()
            // );
            // let mut lock = exploration_buffer.write().unwrap();
            // lock.append(&mut temp_replay_buffer);

            // // Truncate replay buffer if it gets too long.
            // if lock.len() > MAXIMUM_REPLAY_BUFFER_SIZE {
            //     log::info!("truncating replay buffer, len = {}", lock.len());
            //     lock.truncate(MAXIMUM_REPLAY_BUFFER_SIZE);
            // }

            // // While doing this, also save the replay buffer
            // let s: String = lock.iter().map(ToString::to_string).collect();
            // drop(lock);
            // let path = replay_path.join(format!("replays_{steps:0>6}.txt"));
            // std::thread::spawn(move || {
            //     let mut file = OpenOptions::new()
            //         .write(true)
            //         .create(true)
            //         .truncate(true)
            //         .open(path)
            //         .expect("replay file path should be valid and writable");
            //     file.write_all(s.as_bytes()).unwrap();
            // });
            // log::info!("saved replays to file");
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
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)
                    .expect("target file path should be valid and writable");
                file.write_all(s.as_bytes()).unwrap();
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

    let betas: Vec<f32> = [[LOW_BETA; LOW_BETA_AGENTS], [HIGH_BETA; HIGH_BETA_AGENTS]].concat();

    let mut temp_replay_buffer = VecDeque::new();
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, &mut rng));

    loop {
        log::info!("search");
        // Do Gumbel sequential halving.
        let mut top_actions = gumbel_sequential_halving(
            &mut nodes,
            &envs,
            &net,
            SAMPLED,
            SIMULATIONS,
            &betas,
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
