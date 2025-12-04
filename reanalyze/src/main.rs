use std::{
    fmt,
    fs::{read_dir, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, Write},
    path::{Path, PathBuf},
};

use clap::Parser;
use rand::prelude::*;
use takzero::{
    network::{
        net6_simhash::{Env, Net},
        Network,
    },
    search::{
        agent::Agent,
        env::Environment,
        node::{batched::BatchedMCTS, Node},
    },
    target::{Augment, Replay, Target},
};
use tch::{Device, TchError};
use thiserror::Error;

#[rustfmt::skip]
#[allow(dead_code)] const fn assert_env<E: Environment>() where Target<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const DEVICE: Device = Device::Cuda(0);
const BATCH_SIZE: usize = 128;
const SAMPLED_ACTIONS: usize = 64;
const SEARCH_BUDGET: u32 = 768;
const ZERO_BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const MIN_POSITIONS: usize = 4000 * 128 / 4; // steps before reanalyze * batch size / forced uses
const _: () = assert!(MIN_POSITIONS > BATCH_SIZE);
const MAX_REANALYZE_BUFFER_LEN: usize = 32_000;

#[cfg(feature = "exploration")]
const MAX_EXPLORATION_BUFFER_SIZE: usize = 0;
#[cfg(feature = "exploration")]
const EXPLORATION_POSITIONS_IN_BATCH: usize = 0;
#[cfg(feature = "exploration")]
const _: () = assert!(EXPLORATION_POSITIONS_IN_BATCH <= BATCH_SIZE);

const UBE_TARGET_BETA: f32 = 0.25;

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find models
    /// and also where to save targets.
    #[arg(long)]
    directory: PathBuf,
}

#[allow(clippy::too_many_lines)]
fn main() {
    env_logger::init();
    let args = Args::parse();

    let seed: u64 = rand::rng().random();
    log::info!("seed = {seed}");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net;
    let mut batched_mcts = BatchedMCTS::<BATCH_SIZE, _>::new(&mut rng);
    let mut position_buffer = Vec::new();
    let mut replays_seek = 0;
    #[cfg(feature = "exploration")]
    let mut exploration_buffer = Vec::new();
    #[cfg(feature = "exploration")]
    let mut exploration_replays_seek = 0;

    loop {
        loop {
            let reanalyze = match read_buffer_lengths(&args.directory) {
                Ok((_, reanalyze)) => reanalyze,
                Err(err) => {
                    log::error!("Could not read buffer lengths: {err}");
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
            };
            if reanalyze > MAX_REANALYZE_BUFFER_LEN {
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
            log::debug!("Checked that there more reanalyze targets are needed.");

            match Net::load(args.directory.join("model_latest.ot"), DEVICE) {
                Ok(new_net) => {
                    net = new_net;
                    break;
                }
                Err(TchError::Torch(err)) => {
                    log::warn!("Cannot load model (internal torch error): {err}, retrying.");
                    std::thread::sleep(std::time::Duration::from_secs(1));
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
            &mut replays_seek,
            &args.directory.join("replays.txt"),
        ) {
            log::error!("Cannot fill position buffer: {err}");
        }

        // Update exploration buffer.
        #[cfg(feature = "exploration")]
        {
            if let Err(err) = fill_buffer_with_positions_from_replays(
                &mut exploration_buffer,
                &mut exploration_replays_seek,
                &args.directory.join("replays-exploration.txt"),
            ) {
                log::error!("Cannot fill recent buffer: {err}");
            };
            if exploration_buffer.len() > MAX_EXPLORATION_BUFFER_SIZE {
                exploration_buffer
                    .drain(..exploration_buffer.len() - MAX_EXPLORATION_BUFFER_SIZE)
                    .for_each(drop);
            }
        }

        if position_buffer.len() < MIN_POSITIONS {
            let duration = std::time::Duration::from_secs(60);
            log::info!(
                "Not enough positions yet ({}), sleeping for {duration:?}",
                position_buffer.len()
            );
            std::thread::sleep(duration);
            continue;
        }
        log::debug!("Number of positions: {}", position_buffer.len());

        // Sample a batch.
        let mut batch = Vec::new();
        #[cfg(feature = "exploration")]
        batch.extend(
            exploration_buffer
                .sample(&mut rng, EXPLORATION_POSITIONS_IN_BATCH)
                .cloned(),
        );
        batch.extend(
            position_buffer
                .sample(&mut rng, BATCH_SIZE - batch.len())
                .cloned(),
        );
        batched_mcts
            .nodes_and_envs_mut()
            .zip(batch)
            .for_each(|((node, env), replay_env)| {
                *node = Node::default();
                *env = replay_env;
            });

        // Perform search.
        // for _ in 0..VISITS {
        //     batched_mcts.simulate(&net, &ZERO_BETA);
        // }
        let selected = batched_mcts.gumbel_sequential_halving(
            &net,
            &ZERO_BETA,
            SAMPLED_ACTIONS,
            SEARCH_BUDGET,
            &mut rng,
        );

        // Create targets.
        let contents: String = batched_mcts
            .nodes_and_envs()
            .zip(selected)
            .map(|((node, env), selected_action)| {
                let value = if node.evaluation.is_known() {
                    node.evaluation
                } else {
                    node.children
                        .iter()
                        .find(|(a, _)| *a == selected_action)
                        .expect("all non-terminal nodes should have at least one child")
                        .1
                        .evaluation
                        .negate()
                }
                .into();
                let policy = node
                    .children
                    .iter()
                    .map(|(a, _)| a)
                    .copied()
                    .zip(node.improved_policy(node.most_visited_count()))
                    .collect(); // policy_target_from_proportional_visits(node);
                let ube = node.ube_target(UBE_TARGET_BETA).into_inner();

                // Log UBE statistics.
                // let root = node.std_dev * node.std_dev;
                // let max_std_dev = node
                //     .children
                //     .iter()
                //     .map(|(_, child)| child.std_dev)
                //     .max()
                //     .unwrap_or_default();
                // let max = max_std_dev * max_std_dev;
                // log::debug!(
                //     "[UBE STATS] ply: {}, bf: {}, root: {root:.5}, max: {max:.5}, target:
                // {ube:.5}",     env.ply,
                //     policy.len()
                // );

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
            .open(args.directory.join("targets-reanalyze.txt"))
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
        .filter(|p| p.extension().is_some_and(|ext| ext == "ot"))
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
    replays_seek: &mut u64,
    file_path: &Path,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(OpenOptions::new().read(true).open(file_path)?);
    reader
        .seek(std::io::SeekFrom::Start(*replays_seek))
        .expect("Replay file should not get shorter.");
    buffer.extend(
        reader
            .by_ref()
            .lines()
            .filter_map(|line| line.unwrap().parse().ok())
            .flat_map(|replay: Replay<Env>| replay.states().collect::<Vec<_>>()),
    );
    *replays_seek = reader
        .stream_position()
        .expect("Replay file should not get shorter.");
    Ok(())
}

/// Sample a Vec of replays in the `directory`.
#[allow(unused)]
fn get_replays(directory: &Path, _model_steps: u32, rng: &mut impl Rng) -> Vec<Replay<Env>> {
    read_dir(directory)
        .unwrap()
        .filter_map(|res| res.ok().map(|entry| entry.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "txt"))
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str()?.split_once('_'))
                .is_some_and(|(before, _after)| before == "replays")
        })
        .filter_map(|p| {
            Some(
                BufReader::new(OpenOptions::new().read(true).open(p).ok()?)
                    .lines()
                    .filter_map(|line| line.unwrap().parse().ok()),
            )
        })
        .flatten()
        .sample(rng, BATCH_SIZE)
}

#[derive(Debug, Error)]
enum ReadBufferLengthsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing component")]
    MissingComponent,
    #[error("wrong checksum")]
    WrongCheckSum,
}

fn read_buffer_lengths(directory: &Path) -> Result<(usize, usize), ReadBufferLengthsError> {
    let buffer_lengths = std::fs::read_to_string(directory.join("buffer_lengths.txt"))?;
    let mut nums = buffer_lengths.split(',').filter_map(|s| s.parse().ok());
    let selfplay: usize = nums
        .next()
        .ok_or(ReadBufferLengthsError::MissingComponent)?;
    let reanalyze: usize = nums
        .next()
        .ok_or(ReadBufferLengthsError::MissingComponent)?;
    let checksum: usize = nums
        .next()
        .ok_or(ReadBufferLengthsError::MissingComponent)?;
    if selfplay + reanalyze != checksum {
        return Err(ReadBufferLengthsError::WrongCheckSum);
    }
    Ok((selfplay, reanalyze))
}
