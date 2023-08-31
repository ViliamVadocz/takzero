#![warn(clippy::pedantic, clippy::style, clippy::nursery)]
// #![warn(clippy::unwrap_used)]

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{atomic::AtomicUsize, RwLock},
};

use clap::Parser;
use rand::prelude::*;
use takzero::{
    fast_tak::Game,
    network::{net5::Net5, Network},
    search::agent::Agent,
};
use target::Target;
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
}

// The environment to learn.
const N: usize = 5;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;

// The network architecture.
type Net = Net5;

// Steps for TD-learning.
const STEP: usize = 5;

// Reference counted RW-lock to the variable store for the beta network.
type BetaNet<'a> = (AtomicUsize, RwLock<&'a mut VarStore>);

const SELF_PLAY_DEVICE: Device = Device::Cuda(0);
const REANALYZE_DEVICE: Device = Device::Cuda(1);
const TRAINING_DEVICE: Device = Device::Cuda(2);

const MAXIMUM_REPLAY_BUFFER_SIZE: usize = 1_000_000;

fn main() {
    run::<Net>();
}

/// Essentially a generic main function.
fn run<NET: Network + Agent<Env>>() {
    let args = Args::parse();
    assert!(
        args.model_path.is_dir(),
        "`model_path` should point to a directory"
    );
    assert!(
        args.replay_path.is_dir(),
        "`replay_path` should point to a directory"
    );

    env_logger::init();

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let seeds: [u64; 6] = rng.gen();

    let mut net = NET::new(Device::Cpu, Some(rng.gen()));
    net.save(args.model_path.join(file_name(0))).unwrap();
    let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

    let (batch_tx, batch_rx) = crossbeam::channel::unbounded::<Vec<Target<Env>>>();

    let replay_queue = RwLock::new(VecDeque::with_capacity(MAXIMUM_REPLAY_BUFFER_SIZE));

    log::info!("Begin.");
    std::thread::scope(|s| {
        // Self-play threads.
        s.spawn(|| {
            tch::no_grad(|| {
                self_play::run::<_, Net>(
                    SELF_PLAY_DEVICE,
                    seeds[0],
                    &beta_net,
                    &replay_queue,
                    &args.replay_path,
                    true,
                );
            });
        });
        s.spawn(|| {
            tch::no_grad(|| {
                self_play::run::<_, Net>(
                    SELF_PLAY_DEVICE,
                    seeds[1],
                    &beta_net,
                    &replay_queue,
                    &args.replay_path,
                    false,
                );
            });
        });
        s.spawn(|| {
            tch::no_grad(|| {
                self_play::run::<_, Net>(
                    SELF_PLAY_DEVICE,
                    seeds[2],
                    &beta_net,
                    &replay_queue,
                    &args.replay_path,
                    false,
                );
            });
        });

        // Reanalyze threads.
        s.spawn(|| {
            tch::no_grad(|| {
                reanalyze::run::<_, Net>(
                    REANALYZE_DEVICE,
                    seeds[3],
                    &beta_net,
                    &batch_tx,
                    &replay_queue,
                );
            });
        });
        s.spawn(|| {
            tch::no_grad(|| {
                reanalyze::run::<_, Net>(
                    REANALYZE_DEVICE,
                    seeds[4],
                    &beta_net,
                    &batch_tx,
                    &replay_queue,
                );
            });
        });
        s.spawn(|| {
            tch::no_grad(|| {
                reanalyze::run::<_, Net>(
                    REANALYZE_DEVICE,
                    seeds[5],
                    &beta_net,
                    &batch_tx,
                    &replay_queue,
                );
            });
        });

        // Training thread.
        s.spawn(|| {
            tch::with_grad(|| {
                training::run::<N, HALF_KOMI, Net>(
                    TRAINING_DEVICE,
                    &beta_net,
                    batch_rx,
                    &args.model_path,
                );
            });
        });
    });
}

fn file_name(n: u64) -> String {
    format!("{n:0>6}_steps.ot")
}
