use std::{
    io::{BufRead, Write},
    path::PathBuf,
};

use clap::Parser;
use fast_tak::{takparse::Move, Game};
use takzero::{
    network::{
        net4::{Net4, RndNormalizationContext},
        Network,
    },
    search::node::Node,
};
use tch::Device;

const N: usize = 4;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;
type Net = Net4;

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
    let mut context = RndNormalizationContext::new(0.0);

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
            println!("{node}");
        } else {
            let visits: u32 = trim.parse().unwrap_or(1);
            println!("simulating {visits} visits");
            for _ in 0..visits {
                node.simulate_simple(&agent, env.clone(), BETA, &mut context);
            }
            println!("{node}");
        }
    }
}
