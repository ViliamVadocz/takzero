#![feature(iter_array_chunks)]

use std::{
    fmt::Write,
    fs::OpenOptions,
    io::{BufRead, BufReader, Write as _},
    path::PathBuf,
};

use clap::Parser;
use fast_tak::takparse::{Move, Tps};
use rand::prelude::*;
use takzero::{
    network::{
        net6_simhash::{Env, Net},
        repr::game_to_tensor,
        HashNetwork,
        Network,
    },
    search::{
        env::Environment,
        node::{batched::BatchedMCTS, Node},
    },
    target::Replay,
};
use tch::Device;

const DEVICE: Device = Device::Cuda(0);
const BETA: f32 = 0.0;
const BATCH_SIZE: usize = 128;

#[derive(Parser, Debug)]
struct Args {
    /// Path to model to load.
    #[arg(long)]
    model_path: PathBuf,
    /// Run an example game
    #[arg(long)]
    example: bool,
    /// Starting position written as TPS
    #[arg(long)]
    tps: Option<Tps>,
}

#[allow(unused)]
fn gather_policy_data(agent: &Net, rng: &mut impl Rng) {
    let file = OpenOptions::new().read(true).open("replays.txt").unwrap();
    let env_batches = BufReader::new(file)
        .lines()
        .filter_map(|line| line.ok()?.parse::<Replay<Env>>().ok())
        .choose_multiple(rng, 1024)
        .into_iter()
        .map(|mut replay| {
            replay.advance(rng.gen_range(0..replay.len()));
            replay.env
        })
        .array_chunks::<BATCH_SIZE>()
        .collect::<Vec<_>>();

    let mut line = String::new();
    for envs in env_batches {
        let mut batched_mcts = BatchedMCTS::from_envs(envs);

        // for _ in 0..800 {
        //     batched_mcts.simulate(&agent, &[BETA; 128]);
        // }
        batched_mcts.gumbel_sequential_halving(agent, &[BETA; 128], 64, 768, rng);

        for (node, _) in batched_mcts.nodes_and_envs() {
            line.clear();
            node.children.iter().for_each(|(a, child)| {
                write!(
                    &mut line,
                    "{a}:{}:{}:{}:{},",
                    child.visit_count, child.evaluation, child.std_dev, child.logit
                )
                .unwrap();
            });
            println!("{line}");
        }
    }
}

fn main() {
    let args = Args::parse();
    let agent = Net::load_partial(args.model_path, DEVICE).unwrap();
    let mut rng = StdRng::seed_from_u64(123);

    let mut env = args.tps.map(Env::from).unwrap_or_default();
    let mut node = Node::default();
    if args.example {
        while env.terminal().is_none() {
            println!("tps: {}", Tps::from(env.clone()));
            // for _ in 0..visits {
            //     node.simulate_simple(&agent, env.clone(), BETA);
            // }
            let mut batched_mcts = BatchedMCTS::from_envs([env.clone()]);
            let (bm_node, _) = batched_mcts.nodes_and_envs_mut().next().unwrap();
            std::mem::swap(bm_node, &mut node);
            batched_mcts.gumbel_sequential_halving(&agent, &[BETA], 64, 768, &mut rng);
            let (bm_node, _) = batched_mcts.nodes_and_envs_mut().next().unwrap();
            std::mem::swap(bm_node, &mut node);
            println!("{node}");

            // Print raw network output.
            let xs = tch::Tensor::concat(
                &node
                    .children
                    .iter()
                    .map(|(a, _)| {
                        let mut clone = env.clone();
                        clone.step(*a);
                        game_to_tensor(&clone, tch::Device::Cpu)
                    })
                    .collect::<Vec<_>>(),
                0,
            )
            .to(DEVICE);
            let local_unc: Vec<f32> = agent.forward_hash(&xs).try_into().unwrap();
            let (_policy, value, ube) = agent.forward_t(&xs, false);
            let value_out: Vec<Vec<f32>> = value.try_into().unwrap();
            let ube_out: Vec<Vec<f32>> = ube.exp().try_into().unwrap();

            println!("network output:");
            println!("[action]  [value]  [local]  [ ube ]");
            let mut network_output = local_unc
                .into_iter()
                .zip(value_out)
                .zip(ube_out)
                .zip(node.children.iter())
                .collect::<Vec<_>>();
            network_output.sort_by_key(|(_, (_, n))| n.visit_count);
            network_output.reverse();
            for (((local, value), ube), (action, _)) in network_output {
                println!(
                    "{: ^8}  {:+.4}  {local:+.4}  {:+.4}",
                    action.to_string(),
                    value[0],
                    ube[0]
                );
            }
            println!();

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
            // let visits: u32 = trim.parse().unwrap_or(1);
            // println!("simulating {visits} visits");
            // for _ in 0..visits {
            //     node.simulate_simple(&agent, env.clone(), BETA);
            // }
            let mut batched_mcts = BatchedMCTS::from_envs([env.clone()]);
            let (bm_node, _) = batched_mcts.nodes_and_envs_mut().next().unwrap();
            std::mem::swap(bm_node, &mut node);
            batched_mcts.gumbel_sequential_halving(&agent, &[BETA], 64, 768, &mut rng);
            let (bm_node, _) = batched_mcts.nodes_and_envs_mut().next().unwrap();
            std::mem::swap(bm_node, &mut node);
        }
        println!("{node}");
    }
}
