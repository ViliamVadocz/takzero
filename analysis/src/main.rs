use std::io::{BufRead, Write};

use takzero::{
    fast_tak::{takparse::Move, Game},
    network::{net5::Net5, Network},
    search::node::Node,
};
use tch::Device;

const N: usize = 5;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;
type Net = Net5;

const DEVICE: Device = Device::Cuda(0);
const BETA: f32 = 0.0;

fn main() {
    let net = Net::load(".\\_data\\5x5\\0\\models\\002000_steps.ot", DEVICE).unwrap();
    let mut env = Env::default();
    let mut node = Node::default();

    let mut input = String::new();
    loop {
        input.clear();
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().lock().read_line(&mut input).unwrap();
        if let Ok(mov) = input.trim().parse::<Move>() {
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
            println!("simulating");
            for _ in 0..128 {
                node.simulate_simple(&net, env.clone(), BETA);
            }
            println!("{node}");
        }
    }
}
