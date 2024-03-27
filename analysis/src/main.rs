use std::{
    // fs::OpenOptions,
    // io::BufReader,
    io::{BufRead, Write},
    path::PathBuf,
};

use clap::Parser;
use fast_tak::takparse::{Move, Tps};
use takzero::{
    network::{
        net4_big::{Env, Net},
        Network,
    },
    search::{env::Environment, node::Node},
    // target::Replay,
};
use tch::Device;

const DEVICE: Device = Device::Cuda(0);
const BETA: f32 = 0.0;

#[derive(Parser, Debug)]
struct Args {
    /// Path to model to load.
    #[arg(long)]
    model_path: PathBuf,
    /// Run an example game with this many visits per step.
    #[arg(long)]
    example_visits: Option<usize>,
    /// Starting position written as TPS
    #[arg(long)]
    tps: Option<Tps>,
}

// use rand::prelude::*;

fn main() {
    let args = Args::parse();

    let agent = Net::load(args.model_path, DEVICE).unwrap();

    // let mut rng = StdRng::seed_from_u64(123);
    // let file = OpenOptions::new()
    //     .read(true)
    //     .open("directed-random-01-replays.txt")
    //     .unwrap();
    // let replays = BufReader::new(file)
    //     .lines()
    //     .filter_map(|line| line.ok()?.parse::<Replay<Env>>().ok())
    //     .choose_multiple(&mut rng, 1000);

    // for mut replay in replays {
    //     replay.advance(rng.gen_range(0..replay.len()));
    //     let env = replay.env;
    //     let mut node = Node::default();
    //     for _ in 0..100 {
    //         node.simulate_simple(&agent, env.clone(), BETA);
    //     }
    //     println!(
    //         "{}",
    //         node.children
    //             .iter()
    //             .map(|(a, child)| format!("{a}:{},", child.visit_count))
    //             .collect::<String>()
    //     );
    // }
    // return;

    let mut env = args.tps.map(Env::from).unwrap_or_default();
    let mut node = Node::default();
    if let Some(visits) = args.example_visits {
        while env.terminal().is_none() {
            println!("tps: {}", Tps::from(env.clone()));
            for _ in 0..visits {
                node.simulate_simple(&agent, env.clone(), BETA);
            }
            println!("{node}");
            let action = node.select_best_action();
            println!(">>> {action}");
            node.descend(&action);
            env.step(action);
        }
        return;
    }

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
            let visits: u32 = trim.parse().unwrap_or(1);
            println!("simulating {visits} visits");
            for _ in 0..visits {
                node.simulate_simple(&agent, env.clone(), BETA);
            }
        }
        println!("{node}");
    }
}
