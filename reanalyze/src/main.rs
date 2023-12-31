use std::{
    fmt,
    fs::{read_dir, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use clap::Parser;
use rand::prelude::*;
use takzero::{
    network::{
        net5::{Env, Net, RndNormalizationContext},
        Network,
    },
    search::{
        agent::Agent,
        env::Environment,
        node::{batched::BatchedMCTS, Node},
    },
    target::{policy_target_from_proportional_visits, Augment, Replay, Target},
};
use tch::Device;

#[rustfmt::skip] #[allow(dead_code)] const fn assert_env<E: Environment>() where Target<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const DEVICE: Device = Device::Cuda(0);
const BATCH_SIZE: usize = 128;
const VISITS: u32 = 800;
const BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find models
    /// and also where to save targets.
    #[arg(long)]
    directory: PathBuf,
}

// TODO:
// - load replays in reanalyze
// - reanalyze to create new targets
// - learn samples from both target locations

fn main() {
    env_logger::init();
    let args = Args::parse();

    let seed: u64 = rand::thread_rng().gen();
    log::info!("seed = {seed}");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(DEVICE, Some(rng.gen()));
    let mut batched_mcts = BatchedMCTS::new(&mut rng, BETA, RndNormalizationContext::new(0.0));

    let mut model_steps = 0;
    loop {
        if let Some((new_steps, model_path)) = get_model_path_with_most_steps(&args.directory) {
            if new_steps > model_steps {
                model_steps = new_steps;
                log::info!("Loading new model: {}", model_path.display());

                net = loop {
                    match Net::load(&model_path, DEVICE) {
                        Ok(net) => break net,
                        Err(err) => {
                            log::error!("Cannot load model: {err}");
                            std::thread::sleep(std::time::Duration::from_secs(1));
                        }
                    }
                }
            }
        }

        let mut replays = get_replays(&args.directory, model_steps, &mut rng);
        if replays.len() < BATCH_SIZE {
            let time = std::time::Duration::from_secs(30);
            log::info!("Not enough replays. Sleeping for {time:?}.");
            std::thread::sleep(time);
            continue;
        }

        // Randomly pick a position from each replay.
        replays
            .iter_mut()
            .for_each(|replay| replay.advance(rng.gen_range(0..replay.len() - 1)));

        batched_mcts
            .nodes_and_envs_mut()
            .zip(replays)
            .for_each(|((node, env), replay)| {
                *node = Node::default();
                *env = replay.env;
            });

        // Perform search.
        for _ in 0..VISITS {
            batched_mcts.simulate(&net);
        }

        // Create targets.
        let contents: String = batched_mcts
            .nodes_and_envs()
            .map(|(node, env)| {
                let value = if node.evaluation.is_known() {
                    node.evaluation
                } else {
                    node.children
                        .iter()
                        .max_by_key(|(_, child)| child.visit_count)
                        .expect("all non-terminal nodes should have at least one child")
                        .1
                        .evaluation
                        .negate()
                }
                .into();
                let policy = policy_target_from_proportional_visits(node);
                let ube = 1.0; // TODO
                Target {
                    env: env.clone(),
                    policy,
                    value,
                    ube,
                }
                .to_string()
            })
            .collect();

        // Save targets to file.
        if let Err(err) = OpenOptions::new()
            .append(true)
            .create(true)
            .open(
                args.directory
                    .join(format!("targets-reanalyze_{model_steps:0>6}.txt")),
            )
            .map(|mut file| file.write_all(contents.as_bytes()))
        {
            log::error!(
                "Could not save targets to file [{err}], so here they are instead:\n{contents}"
            );
        } else {
            log::info!("Saved targets to file.");
        }
    }
}

/// Get the path to the model file (ending with ".ot")
/// which has the highest number of steps (number after '_')
/// in the given directory.
fn get_model_path_with_most_steps(directory: &PathBuf) -> Option<(u32, PathBuf)> {
    read_dir(directory)
        .unwrap()
        .filter_map(|res| res.ok().map(|entry| entry.path()))
        .filter(|p| p.extension().map(|ext| ext == "ot").unwrap_or_default())
        .filter_map(|p| {
            Some((
                p.file_stem()?
                    .to_str()?
                    .split_once('_')?
                    .1
                    .parse::<u32>()
                    .ok()?,
                p,
            ))
        })
        .max_by_key(|(s, _)| *s)
}

/// Sample a Vec of replays in the `directory`.
fn get_replays(directory: &Path, _model_steps: u32, rng: &mut impl Rng) -> Vec<Replay<Env>> {
    read_dir(directory)
        .unwrap()
        .filter_map(|res| res.ok().map(|entry| entry.path()))
        .filter(|p| p.extension().map(|ext| ext == "txt").unwrap_or_default())
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str()?.split_once('_'))
                .map(|(before, _after)| before == "replays")
                .unwrap_or_default()
        })
        .filter_map(|p| {
            Some(
                BufReader::new(OpenOptions::new().read(true).open(p).ok()?)
                    .lines()
                    .filter_map(|line| line.unwrap().parse().ok()),
            )
        })
        .flatten()
        .choose_multiple(rng, BATCH_SIZE)
}
