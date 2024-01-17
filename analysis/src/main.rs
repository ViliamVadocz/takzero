use std::{
    io::{BufRead, Write},
    path::PathBuf,
};

use clap::Parser;
use fast_tak::takparse::Move;
use takzero::{
    network::{
        net5::{Env, Net},
        Network,
    },
    search::node::Node,
};
use tch::Device;

const DEVICE: Device = Device::Cuda(0);
const BETA: f32 = 0.0;

#[derive(Parser, Debug)]
struct Args {
    /// Path to model to load.
    #[arg(long)]
    model_path: PathBuf,
}

fn main() {
    let args = Args::parse();

    let agent = Net::load(args.model_path, DEVICE).unwrap();
    let mut env = Env::default();
    let mut node = Node::default();

    let mut input = String::new();
    loop {
        input.clear();
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
