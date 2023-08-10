use std::path::PathBuf;

use clap::Parser;
use rand::{Rng, SeedableRng};
use takzero::network::{net3::Net3, Network};
use target::{Replay, Target};
use tch::Device;

mod evaluation;
mod reanalyze;
mod self_play;
mod target;
mod training;

const N: usize = 3;
const HALF_KOMI: i8 = 0;
type Net = Net3;

#[derive(Parser, Debug)]
struct Args {
    /// Path to store models
    #[arg(short, long)]
    model_path: PathBuf,
    /// Path to store replays
    #[arg(short, long)]
    replay_path: PathBuf,
    /// Seed for the RNG
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
    /// Starting model number
    #[arg(short, long)]
    start_model: Option<u64>,
}

fn main() {
    run::<Net>()
}

// TODO: Could constrain with Agent<Game<N, HALF_KOMI>>?
fn run<NET: Network>() {
    let args = Args::parse();

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let net = match args.start_model {
        None => {
            let net = NET::new(Device::Cpu, Some(rng.gen()));
            net.save(args.model_path.join(file_name(0))).unwrap();
            net
        }
        Some(n) => NET::load(args.model_path.join(file_name(n)), Device::Cpu)
            .expect("Model path should be valid"),
    };
    let model_number = args.start_model.unwrap_or_default();

    std::thread::scope(|s| {
        let (replay_tx, replay_rx) = crossbeam::channel::unbounded::<Replay<N, HALF_KOMI>>();
        let (batch_tx, batch_rx) = crossbeam::channel::unbounded::<Box<[Target<N, HALF_KOMI>]>>();

        s.spawn(|| tch::no_grad(|| self_play::run(replay_tx)));
        s.spawn(|| tch::no_grad(|| reanalyze::run(replay_rx, batch_tx)));
        s.spawn(|| training::run(batch_rx)); // TODO: distribute newest model
        s.spawn(|| tch::no_grad(|| evaluation::run()));
    });
}

fn file_name(n: u64) -> String {
    format!("{n:0>3}.pt")
}
