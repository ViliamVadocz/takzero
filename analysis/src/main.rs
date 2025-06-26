use std::{
    io::{BufRead, Write as _},
    path::PathBuf,
};

use clap::Parser;
use fast_tak::takparse::{Move, Tps};
use takzero::{
    network::{
        net6_simhash::{Env, Net},
        Network,
    },
    search::{env::Environment, node::Node},
};

const BETA: f32 = 0.0;
const BATCH_SIZE: usize = 128;

#[derive(Parser, Debug)]
struct Args {
    /// Path to model to load
    #[arg(long)]
    model_path: PathBuf,
    /// Run an example game
    #[arg(long)]
    example: bool,
    /// Starting position written as TPS
    #[arg(long)]
    tps: Option<Tps>,
}

fn run_example(mut env: Env, mut node: Node<Env>, agent: &Net) {
    while env.terminal().is_none() {
        println!("tps: {}", Tps::from(env.clone()));
        node.simulate_batch(agent, &env, BETA, BATCH_SIZE);
        let action = node.select_best_action();
        println!(">>> {action}");
        node.descend(&action);
        env.step(action);
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let device = tch::Device::cuda_if_available();
    if !device.is_cuda() {
        log::warn!("CUDA not available, using CPU");
    }
    let agent = Net::load_partial(args.model_path, device).unwrap();

    let mut env = args.tps.map(Env::from).unwrap_or_default();
    let mut node = Node::default();
    if args.example {
        return run_example(env, node, &agent);
    }

    // TODO: History
    // let mut history = vec![];
    let mut input = String::new();
    loop {
        input.clear();
        println!("tps: {}", Tps::from(env.clone()));
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().lock().read_line(&mut input).unwrap();
        let trim = input.trim();
        if let Ok(mov) = trim.parse::<Move>() {
            match env.play(mov) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("{e}");
                    continue;
                }
            }
            node.descend(&mov);
        } else {
            node.simulate_batch(&agent, &env, BETA, BATCH_SIZE);
        }
        println!("{node}");
    }
}
