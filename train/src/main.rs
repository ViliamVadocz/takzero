#![warn(clippy::pedantic, clippy::style, clippy::nursery)]
// https://github.com/rust-lang/rust-clippy/issues/8538
#![allow(clippy::iter_with_drain)]

use std::{
    collections::VecDeque,
    fmt,
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::{atomic::AtomicUsize, RwLock},
};

use arrayvec::ArrayVec;
use clap::Parser;
use rand::prelude::*;
use takzero::{
    fast_tak::Game,
    network::{net5::Net5, Network},
    search::{agent::Agent, env::Environment, STEP},
};
use target::{Augment, Replay, Target};
use tch::{nn::VarStore, Device};

mod reanalyze;
mod self_play;
mod target;
mod training;

#[derive(Parser, Debug)]
struct Args {
    /// Path to store models
    #[arg(long)]
    model_path: PathBuf,
    /// Path to store replays
    #[arg(long)]
    replay_path: PathBuf,
    /// Seed for the RNG
    #[arg(long, default_value_t = 42)]
    seed: u64,
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

// RW-lock to the variable store for the beta network.
type BetaNet<'a> = (AtomicUsize, RwLock<&'a mut VarStore>);
// Reference counted RW-lock to the replay buffer.
type ReplayBuffer = RwLock<VecDeque<Replay<Env>>>;

const TRAINING_DEVICE: Device = Device::Cuda(0);
const SELF_PLAY_DEVICE: Device = Device::Cuda(1);
const REANALYZE_DEVICE_1: Device = Device::Cuda(2);
const REANALYZE_DEVICE_2: Device = Device::Cuda(3);

const SELF_PLAY_THREADS: usize = 8;
const REANALYZE_THREADS_1: usize = 16;
const REANALYZE_THREADS_2: usize = 16;
const REANALYZE_THREADS: usize = REANALYZE_THREADS_1 + REANALYZE_THREADS_2;

const MINIMUM_REPLAY_BUFFER_SIZE: usize = 10_000;
const MAXIMUM_REPLAY_BUFFER_SIZE: usize = 2_000_000;

fn main() {
    let args = Args::parse();
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

    env_logger::init();

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let self_play_seeds: [u64; SELF_PLAY_THREADS] = rng.gen();
    let reanalyze_seeds: [u64; REANALYZE_THREADS] = rng.gen();

    let mut net = args.resume.as_ref().map_or_else(
        || Net::new(Device::Cpu, Some(rng.gen())),
        |path| Net::load(path, Device::Cpu).unwrap(),
    );
    net.save(args.model_path.join(file_name(0))).unwrap();
    let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

    let (batch_tx, batch_rx) = crossbeam::channel::bounded::<Vec<Target<Env>>>(64);

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
        initialize_buffer_with_random_moves(&replay_buffer, &mut rng);
    }

    log::info!("Begin.");
    std::thread::scope(|s| {
        // Self-play threads.
        for (i, seed) in self_play_seeds
            .into_iter()
            .take(SELF_PLAY_THREADS)
            .enumerate()
        {
            let beta_net = &beta_net;
            let replay_path = &args.replay_path;
            let replay_buffer = &replay_buffer;
            s.spawn(move || {
                tch::no_grad(|| {
                    self_play::run(
                        SELF_PLAY_DEVICE,
                        seed,
                        beta_net,
                        replay_buffer,
                        replay_path,
                        i == 0,
                    );
                });
            });
        }

        // Reanalyze threads.
        for (i, seed) in reanalyze_seeds
            .into_iter()
            .take(REANALYZE_THREADS)
            .enumerate()
        {
            let beta_net = &beta_net;
            let replay_buffer = &replay_buffer;
            let batch_tx = &batch_tx;
            let device = if i < REANALYZE_THREADS_1 {
                REANALYZE_DEVICE_1
            } else {
                REANALYZE_DEVICE_2
            };
            s.spawn(move || {
                tch::no_grad(|| {
                    reanalyze::run(device, seed, beta_net, batch_tx, replay_buffer);
                });
            });
        }

        // Training thread.
        s.spawn(|| {
            tch::with_grad(|| {
                training::run(TRAINING_DEVICE, &beta_net, batch_rx, &args.model_path);
            });
        });
    });
}

fn file_name(n: u64) -> String {
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
