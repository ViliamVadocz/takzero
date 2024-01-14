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
use tch::{Device, TchError};

#[rustfmt::skip]
#[allow(dead_code)] const fn assert_env<E: Environment>() where Target<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const DEVICE: Device = Device::Cuda(0);
const BATCH_SIZE: usize = 128;
const VISITS: u32 = 800;
const BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const MIN_POSITIONS: usize = 640000;
const _: () = assert!(MIN_POSITIONS > BATCH_SIZE);

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find models
    /// and also where to save targets.
    #[arg(long)]
    directory: PathBuf,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    let seed: u64 = rand::thread_rng().gen();
    log::info!("seed = {seed}");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(DEVICE, Some(rng.gen()));
    let mut batched_mcts = BatchedMCTS::new(&mut rng, BETA, RndNormalizationContext::new(0.0));
    let mut position_buffer = Vec::new();
    let mut replays_read = 0;

    loop {
        loop {
            match Net::load(&args.directory.join("model_latest.ot"), DEVICE) {
                Ok(new_net) => {
                    net = new_net;
                    break;
                }
                Err(TchError::Torch(err)) => {
                    log::warn!("Cannot load model (internal torch error): {err}, not retrying.");
                    break;
                }
                Err(err) => {
                    log::error!("Cannot load model (some other reason): {err}, retrying.");
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }

        // Fill the position buffer.
        if let Err(err) = fill_buffer_with_positions_from_replays(
            &mut position_buffer,
            &mut replays_read,
            &args.directory.join("replays.txt"),
        ) {
            log::error!("Cannot fill position buffer: {err}");
        };

        if position_buffer.len() < MIN_POSITIONS {
            let duration = std::time::Duration::from_secs(60);
            log::info!(
                "Not enough positions yet ({}), sleeping for {duration:?}",
                position_buffer.len()
            );
            std::thread::sleep(duration);
            continue;
        }

        // Sample a batch.
        let batch = position_buffer.choose_multiple(&mut rng, BATCH_SIZE);
        batched_mcts
            .nodes_and_envs_mut()
            .zip(batch)
            .for_each(|((node, env), replay_env)| {
                *node = Node::default();
                *env = replay_env.clone();
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
            .open(args.directory.join(format!("targets-reanalyze.txt")))
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
#[allow(unused)]
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

/// Fill the buffer with new positions from the replay file.
fn fill_buffer_with_positions_from_replays(
    buffer: &mut Vec<Env>,
    replays_read: &mut usize,
    file_path: &Path,
) -> std::io::Result<()> {
    buffer.extend(
        BufReader::new(OpenOptions::new().read(true).open(file_path)?)
            .lines()
            .skip(*replays_read)
            .map(|x| {
                *replays_read += 1;
                x.unwrap()
            })
            .filter_map(|line| line.parse().ok())
            .flat_map(|replay: Replay<Env>| {
                let mut env = replay.env;
                replay
                    .actions
                    .into_iter()
                    .map(|a| {
                        env.step(a);
                        env.clone()
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            }),
    );
    Ok(())
}

/// Sample a Vec of replays in the `directory`.
#[allow(unused)]
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
