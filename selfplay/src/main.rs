use std::{
    fmt,
    fs::{read_dir, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use clap::Parser;
use fast_tak::takparse::Move;
use ordered_float::NotNan;
use rand::prelude::*;
use takzero::{
    network::{
        net5::{Env, Net, RndNormalizationContext},
        Network,
    },
    search::{agent::Agent, env::Environment, eval::Eval, node::batched::BatchedMCTS},
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
const WEIGHTED_RANDOM_PLIES: u16 = 30;
const NOISE_ALPHA: f32 = 0.05;
const NOISE_RATIO: f32 = 0.1;
const NOISE_PLIES: u16 = 60;

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

    // Initialize buffers.
    let mut policy_targets: [_; BATCH_SIZE] = std::array::from_fn(|_| Vec::new());
    let mut targets = Vec::new();
    let mut complete_replays = Vec::new();

    let mut batched_mcts = BatchedMCTS::new(&mut rng, BETA, RndNormalizationContext::new(0.0));

    for steps in 0.. {
        log::info!("Step: {steps}");
        let start = std::time::Instant::now();
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
        log::debug!("Loading model took {:?}.", start.elapsed());

        // One simulation batch to initialize root policy if it has not been done yet.
        batched_mcts.simulate(&net);

        // Apply noise.
        batched_mcts.apply_noise(&mut rng, NOISE_PLIES, NOISE_ALPHA, NOISE_RATIO);

        // Search.
        for _ in 0..VISITS {
            batched_mcts.simulate(&net);
        }

        let selected_actions =
            batched_mcts.select_actions_in_selfplay(&mut rng, WEIGHTED_RANDOM_PLIES);
        take_a_step(&mut batched_mcts, &mut policy_targets, &selected_actions);
        restart_envs_and_complete_targets(
            &mut batched_mcts,
            &mut policy_targets,
            &mut targets,
            &mut complete_replays,
            &mut rng,
        );

        if !targets.is_empty() {
            save_targets_to_file(&mut targets, &args.directory);
        }
        if !complete_replays.is_empty() {
            save_replays_to_file(&mut complete_replays, &args.directory);
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

type PolicyTarget = (Env, Box<[(Move, NotNan<f32>)]>);

/// Take a step in each environment.
/// Generate target policy.
fn take_a_step(
    batched_mcts: &mut BatchedMCTS<BATCH_SIZE, Env, Net>,
    policy_targets: &mut [Vec<PolicyTarget>],
    selected_actions: &[Move],
) {
    batched_mcts
        .nodes_and_envs()
        .zip(policy_targets)
        .for_each(|((node, env), policy_targets)| {
            policy_targets.push((env.clone(), policy_target_from_proportional_visits(node)));
        });
    batched_mcts.step(selected_actions);
}

/// Restart any finished environments.
/// Complete targets of finished games using the game result.
#[allow(clippy::too_many_arguments)]
fn restart_envs_and_complete_targets(
    batched_mcts: &mut BatchedMCTS<BATCH_SIZE, Env, Net>,
    policy_targets: &mut [Vec<PolicyTarget>],
    targets: &mut Vec<Target<Env>>,
    finished_replays: &mut Vec<Replay<Env>>,
    rng: &mut impl Rng,
) {
    batched_mcts
        .restart_terminal_envs(rng)
        .zip(policy_targets)
        .for_each(|(terminal_and_replay, policy_targets)| {
            if let Some((terminal, replay)) = terminal_and_replay {
                let mut value = Eval::from(terminal);
                for (env, policy) in policy_targets.drain(..).rev() {
                    value = value.negate();
                    targets.push(Target {
                        env,
                        value: f32::from(value),
                        ube: 0.0,
                        policy,
                    });
                }
                finished_replays.push(replay);
            }
        });
}

/// Save targets to a file. Drains the target Vec.
fn save_targets_to_file(targets: &mut Vec<Target<Env>>, directory: &Path) {
    let contents: String = targets.drain(..).map(|target| target.to_string()).collect();
    if let Err(err) = OpenOptions::new()
        .append(true)
        .create(true)
        .open(directory.join(format!("targets-selfplay.txt")))
        .map(|mut file| file.write_all(contents.as_bytes()))
    {
        log::error!(
            "Could not save targets to file [{err}], so here they are instead:\n{contents}"
        );
    }
}

/// Save replays to a file. Drains the replays Vec.
fn save_replays_to_file(replays: &mut Vec<Replay<Env>>, directory: &Path) {
    let contents: String = replays.drain(..).map(|target| target.to_string()).collect();
    if let Err(err) = OpenOptions::new()
        .append(true)
        .create(true)
        .open(directory.join(format!("replays.txt")))
        .map(|mut file| file.write_all(contents.as_bytes()))
    {
        log::error!(
            "Could not save replays to file [{err}], so here they are instead:\n{contents}"
        );
    }
}
