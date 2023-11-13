use std::io::{BufRead, Write};

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

fn main() {
    let net = Net::load(".\\033000_steps.ot", DEVICE).unwrap();
    let mut env = Env::default();
    let mut node = Node::default();
    let mut context = RndNormalizationContext::new(4.038908004760742);

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
                node.simulate_simple(&net, env.clone(), BETA, &mut context);
            }
            println!("{node}");
        }
    }
}
