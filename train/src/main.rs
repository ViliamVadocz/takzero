#![warn(clippy::pedantic, clippy::style, clippy::nursery)]
// #![warn(clippy::unwrap_used)]

use std::{
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
use target::{Replay, Target};
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
const TRAINING_DEVICE: Device = Device::Cuda(3);

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
    let seeds: [u64; 2] = rng.gen();

    let mut net = NET::new(Device::Cpu, Some(rng.gen()));
    net.save(args.model_path.join(file_name(0))).unwrap();
    let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

    let (replay_tx, replay_rx) = crossbeam::channel::unbounded::<Replay<Env>>();
    let (batch_tx, batch_rx) = crossbeam::channel::unbounded::<Vec<Target<Env>>>();

    log::info!("Begin.");
    std::thread::scope(|s| {
        s.spawn(|| {
            tch::no_grad(|| {
                self_play::run::<_, Net>(SELF_PLAY_DEVICE, seeds[0], &beta_net, &replay_tx);
            });
        });
        s.spawn(|| {
            tch::no_grad(|| {
                reanalyze::run::<_, Net>(
                    REANALYZE_DEVICE,
                    seeds[1],
                    &beta_net,
                    replay_rx,
                    batch_tx,
                    &args.replay_path,
                );
            });
        });
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
