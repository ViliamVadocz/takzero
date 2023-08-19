use std::{
    path::PathBuf,
    sync::{atomic::AtomicUsize, RwLock},
};

use clap::Parser;
use fast_tak::Game;
use rand::prelude::*;
use takzero::{
    network::{net3::Net3, Network},
    search::agent::Agent,
};
use target::{Replay, Target};
use tch::{nn::VarStore, Device};

// #[warn(clippy::pedantic, clippy::style, clippy::nursery)]

// Windows allocator sucks, so use MiMalloc instead.
#[cfg(windows)]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod evaluation;
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
    /// Path to store evaluation statistics
    #[arg(long)]
    statistics_path: PathBuf,
    /// Seed for the RNG
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// The environment to learn.
const N: usize = 3;
const HALF_KOMI: i8 = 0;
type Env = Game<N, HALF_KOMI>;

// The network architecture.
type Net = Net3;

// Steps for TD-learning.
const STEP: usize = 5;

// Reference counted RW-lock to the variable store for the beta network.
type BetaNet<'a> = (AtomicUsize, RwLock<&'a mut VarStore>);

fn main() {
    run::<Net>()
}

/// Essentially a generic main function.
fn run<NET: Network + Agent<Env>>() {
    let args = Args::parse();
    assert!(args.model_path.is_dir(), "`model_path` should point to a directory");
    assert!(args.replay_path.is_dir(), "`replay_path` should point to a directory");

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let seeds: [u64; 3] = rng.gen();

    let mut net = NET::new(Device::Cpu, Some(rng.gen()));
    net.save(args.model_path.join(file_name(0))).unwrap();
    let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

    println!("Ready, set, go!");
    std::thread::scope(|s| {
        let (replay_tx, replay_rx) = crossbeam::channel::unbounded::<Replay<Env>>();
        let (batch_tx, batch_rx) = crossbeam::channel::unbounded::<Vec<Target<Env>>>();

        #[rustfmt::skip]
        s.spawn(|| tch::no_grad(|| reanalyze::run::<_, Net>(Device::Cuda(0), seeds[0], &beta_net, replay_rx, batch_tx, args.replay_path)));
        s.spawn(|| tch::no_grad(|| self_play::run::<_, Net>(Device::Cuda(1), seeds[1], &beta_net, replay_tx)));
        s.spawn(|| tch::no_grad(|| evaluation::run::<_, Net>(Device::Cuda(2), seeds[2], &beta_net, args.statistics_path)));
        s.spawn(|| training::run::<N, HALF_KOMI, Net>(Device::Cuda(3), &beta_net, batch_rx, args.model_path));
    });
}

fn file_name(n: u64) -> String {
    format!("{n:0>3}.pt")
}
