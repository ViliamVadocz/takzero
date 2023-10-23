#![warn(clippy::pedantic, clippy::style, clippy::nursery)]
// https://github.com/rust-lang/rust-clippy/issues/8538
#![allow(clippy::iter_with_drain)]

use std::{
    collections::VecDeque,
    fmt,
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, AtomicUsize},
        RwLock,
    },
};

use arrayvec::ArrayVec;
use clap::Parser;
use fast_tak::Game;
use rand::{distributions::Uniform, prelude::*};
use takzero::{
    network::{net5::Net5, Network},
    search::{agent::Agent, env::Environment, DISCOUNT_FACTOR, STEP},
    target::{Augment, Replay, Target},
};
use tch::{nn::VarStore, Device};

use crate::{
    self_play::{
        BASELINE_AGENTS,
        EXPLOITATION_STEP,
        GREEDY_AGENTS,
        HIGH_BETA,
        HIGH_BETA_AGENTS,
        LOW_BETA,
        LOW_BETA_AGENTS,
        WEIGHTED_RANDOM_PLIES,
    },
    training::{
        EFFECTIVE_BATCH_SIZE,
        LEARNING_RATE,
        PUBLISHES_BETWEEN_SAVE,
        STEPS_BETWEEN_PUBLISH,
    },
};

mod reanalyze;
mod self_play;
mod training;

#[derive(Parser, Debug)]
struct Args {
    /// Path to store models
    #[arg(long)]
    model_path: PathBuf,
    /// Path to store replays
    #[arg(long)]
    replay_path: PathBuf,
    /// Load an existing replay file
    #[arg(long)]
    load_replay: Option<PathBuf>,
    /// Path to model to resume training
    #[arg(long)]
    resume: Option<PathBuf>,
}

// The environment to learn.
const N: usize = 5;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;
#[rustfmt::skip] #[allow(dead_code)] const fn assert_env<E: Environment>() where Replay<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
type Net = Net5;
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

// RW-lock to the replay buffer.
type ReplayBuffer = RwLock<VecDeque<Replay<Env>>>;
// For sharing network weights.
type SharedNet<'a> = (AtomicUsize, RwLock<&'a mut VarStore>, RwLock<f64>);

const MINIMUM_REPLAY_BUFFER_SIZE: usize = 10_000;
const MAXIMUM_REPLAY_BUFFER_SIZE: usize = 100_000_000;
const SELF_PLAY_DEVICE: Device = Device::Cuda(0);
const TRAINING_DEVICE: Device = Device::Cuda(0);
const REANALYZE_PER_DEVICE: &[(Device, usize)] = &[
    (Device::Cuda(0), 6),
    (Device::Cuda(1), 8),
    (Device::Cuda(2), 8),
    (Device::Cuda(3), 8),
];

fn assert_paths_are_correct(args: &Args) {
    assert!(
        args.model_path.is_dir(),
        "`model_path` should point to a directory"
    );
    assert!(
        args.replay_path.is_dir(),
        "`replay_path` should point to a directory"
    );
    if let Some(path) = args.resume.as_ref() {
        assert!(path.is_file(), "`resume` should be a file");
        assert_eq!(
            path.extension().unwrap(),
            "ot",
            "model extension should be `.ot`"
        );
    }
    if let Some(path) = args.load_replay.as_ref() {
        assert!(path.is_file(), "`load_replay` should be a file");
    }
}

fn make_replay_buffer(args: &Args, rng: &mut impl Rng) -> ReplayBuffer {
    let replay_buffer: ReplayBuffer =
        RwLock::new(VecDeque::with_capacity(MAXIMUM_REPLAY_BUFFER_SIZE + 1000));
    #[allow(clippy::option_if_let_else)]
    if let Some(path) = args.load_replay.as_ref() {
        // Initialize replay buffer from file.
        let file = OpenOptions::new().read(true).open(path).unwrap();
        let mut lock = replay_buffer.write().unwrap();
        for line in BufReader::new(file).lines() {
            let replay: Replay<Env> = line.unwrap().parse().unwrap();
            lock.push_front(replay);
        }
    } else {
        initialize_buffer_with_random_moves(&replay_buffer, rng);
    }
    replay_buffer
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    assert_paths_are_correct(&args);

    // Generate seeds.
    let seed: u64 = rand::thread_rng().gen();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let exploitation_seed: u64 = rng.gen();
    let exploration_seed: u64 = rng.gen();
    let reanalyze_seeds: Vec<u64> = (&mut rng)
        .sample_iter(Uniform::new_inclusive(u64::MIN, u64::MAX))
        .take(
            REANALYZE_PER_DEVICE
                .iter()
                .map(|(_, n)| n)
                .product::<usize>(),
        )
        .collect();

    // Create network.
    let mut net = args.resume.as_ref().map_or_else(
        || Net::new(Device::Cpu, Some(rng.gen())),
        |path| Net::load(path, Device::Cpu).unwrap(),
    );
    net.vs_mut().freeze();
    net.save(args.model_path.join(file_name(0))).unwrap();

    print_hyper_parameters(&net, seed, &args);

    // Create synchronization primitives.
    let shared_net: SharedNet = (
        AtomicUsize::new(0),
        RwLock::new(net.vs_mut()),
        RwLock::new(0.0),
    );
    let (batch_tx, batch_rx) = crossbeam::channel::bounded::<Vec<Target<Env>>>(64);
    let exploitation_buffer = RwLock::new(Vec::new());
    let exploration_buffer = make_replay_buffer(&args, &mut rng);
    let training_steps = AtomicU32::new(0);

    log::info!("Begin.");
    std::thread::scope(|s| {
        // Exploitation thread.
        s.spawn(|| {
            tch::no_grad(|| {
                self_play::exploitation(
                    SELF_PLAY_DEVICE,
                    exploitation_seed,
                    &shared_net,
                    &exploitation_buffer,
                    &training_steps,
                    &args.replay_path,
                );
            });
        });
        // Exploration thread.
        s.spawn(|| {
            tch::no_grad(|| {
                self_play::exploration(
                    SELF_PLAY_DEVICE,
                    exploration_seed,
                    &shared_net,
                    &exploration_buffer,
                    &training_steps,
                    &args.replay_path,
                );
            });
        });

        // Training thread.
        s.spawn(|| {
            training::run(
                TRAINING_DEVICE,
                &shared_net,
                batch_rx,
                &exploitation_buffer,
                &training_steps,
                &args.model_path,
            );
        });
        // Reanalyze threads.
        let mut seed_iter = reanalyze_seeds.into_iter();
        for (device, amount) in REANALYZE_PER_DEVICE {
            for seed in seed_iter.by_ref().take(*amount) {
                let shared_net = &shared_net;
                let batch_tx = &batch_tx;
                let replay_buffer = &exploration_buffer;
                s.spawn(move || {
                    tch::no_grad(|| {
                        reanalyze::run(device, seed, shared_net, batch_tx, replay_buffer);
                    });
                });
            }
        }
    });
}

fn file_name(n: u32) -> String {
    format!("{n:0>6}_steps.ot")
}

fn initialize_buffer_with_random_moves(replay_buffer: &ReplayBuffer, rng: &mut impl Rng) {
    let mut actions = Vec::new();
    let mut replays = Vec::new();
    let mut env = Env::default();
    while replay_buffer.read().unwrap().len() < MINIMUM_REPLAY_BUFFER_SIZE {
        new_opening(&mut env, &mut actions, rng);
        while env.terminal().is_none() {
            // Choose random action.
            env.populate_actions(&mut actions);
            let action = actions.drain(..).choose(rng).unwrap();
            // Push start of fresh replay.
            replays.push(Replay {
                env: env.clone(),
                actions: ArrayVec::default(),
            });
            // Update existing replays.
            let from = replays.len().saturating_sub(STEP);
            for replay in &mut replays[from..] {
                replay.actions.push(action);
            }
            // Take a step in the environment.
            env.step(action);
        }
        replay_buffer.write().unwrap().extend(replays.drain(..));
    }
    log::debug!("generated random data to fill the replay buffer");
}

fn new_opening<E: Environment>(env: &mut E, actions: &mut Vec<E::Action>, rng: &mut impl Rng) {
    *env = E::default();
    for _ in 0..2 {
        env.populate_actions(actions);
        env.step(actions.drain(..).choose(rng).unwrap());
    }
}

fn print_hyper_parameters(net: &Net, seed: u64, args: &Args) {
    println!("=== Args ===");
    println!("{args:#?}");

    println!("=== Tak ===");
    println!("N = {N}");
    println!("HALF_KOMI = {HALF_KOMI}");

    println!("=== Search ===");
    println!("DISCOUNT_FACTOR = {DISCOUNT_FACTOR}");
    println!("TEMPORAL_DIFFERENCE_STEP = {STEP}");

    println!("=== Devices ===");
    println!("SELF_PLAY_DEVICE = {SELF_PLAY_DEVICE:?}");
    println!("TRAINING_DEVICE = {TRAINING_DEVICE:?}");
    println!("REANALYZE_PER_DEVICE = {REANALYZE_PER_DEVICE:?}");

    println!("=== Training ===");
    println!("MINIMUM_REPLAY_BUFFER_SIZE = {MINIMUM_REPLAY_BUFFER_SIZE}");
    println!("MAXIMUM_REPLAY_BUFFER_SIZE = {MAXIMUM_REPLAY_BUFFER_SIZE}");

    println!("self_play::BATCH_SIZE = {}", self_play::BATCH_SIZE);
    println!("self_play::SAMPLED = {}", self_play::SAMPLED);
    println!("self_play::SIMULATIONS = {}", self_play::SIMULATIONS);
    println!("WEIGHTED_RANDOM_PLIES = {WEIGHTED_RANDOM_PLIES}");
    println!("GREEDY_AGENTS = {GREEDY_AGENTS}");
    println!("BASELINE_AGENTS = {BASELINE_AGENTS}");
    println!("LOW_BETA_AGENTS = {LOW_BETA_AGENTS}");
    println!("HIGH_BETA_AGENTS = {HIGH_BETA_AGENTS}");
    println!("LOW_BETA = {LOW_BETA}");
    println!("HIGH_BETA = {HIGH_BETA}");
    println!("EXPLOITATION_STEP = {EXPLOITATION_STEP}");

    println!("reanalyze::BATCH_SIZE = {}", reanalyze::BATCH_SIZE);
    println!("reanalyze::SAMPLED = {}", reanalyze::SAMPLED);
    println!("reanalyze::SIMULATIONS = {}", reanalyze::SIMULATIONS);

    println!("LEARNING_RATE = {LEARNING_RATE}");
    println!("EFFECTIVE_BATCH_SIZE = {EFFECTIVE_BATCH_SIZE}");
    println!("STEPS_BETWEEN_PUBLISH = {STEPS_BETWEEN_PUBLISH}");
    println!("PUBLISHED_BETWEEN_SAVE = {PUBLISHES_BETWEEN_SAVE}");

    println!("seed = {seed}");

    println!("=== Network ===");
    let mut variables: Vec<_> = net.vs().variables().into_iter().collect();
    variables.sort_by_key(|(name, _)| name.clone());
    let mut total_params = 0;
    for (name, tensor) in variables {
        println!("{name} = {tensor:?}");
        if !name.starts_with("rnd") {
            total_params += tensor.size().into_iter().product::<i64>();
        }
    }
    println!("total_params = {total_params}");
}
