use std::{
    fmt,
    fs::{read_dir, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use clap::Parser;
use fast_tak::{takparse::Move, Game};
use rand::{distributions::WeightedIndex, prelude::*};
use takzero::{
    network::{
        net4::{Net4 as Net, RndNormalizationContext},
        Network,
    },
    search::{
        agent::Agent,
        env::Environment,
        eval::Eval,
        node::{gumbel::batched_simulate, Node},
        DISCOUNT_FACTOR,
    },
    target::{Augment, Replay, Target},
};
use tch::Device;

// The environment to learn.
const N: usize = 4;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;
#[rustfmt::skip] #[allow(dead_code)] const fn assert_env<E: Environment>() where Replay<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const BATCH_SIZE: usize = 256;
const VISITS: u32 = 800;
const BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const WEIGHTED_RANDOM_PLIES: u16 = 30;
const NOISE_ALPHA: f32 = 0.2;
const NOISE_RATIO: f32 = 0.1;
const NOISE_PLIES: u16 = 20;

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find models
    /// and also where to save targets.
    #[arg(long)]
    directory: PathBuf,
}

fn main() {
    let args = Args::parse();

    let seed: u64 = rand::thread_rng().gen();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(Device::Cuda(0), Some(rng.gen()));

    // Initialize buffers.
    let mut actions: [_; BATCH_SIZE] = std::array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = std::array::from_fn(|_| Vec::new());
    let mut nodes: [_; BATCH_SIZE] = std::array::from_fn(|_| Node::default());
    let mut envs: [_; BATCH_SIZE] = actions
        .iter_mut()
        .map(|actions| Env::new_opening(&mut rng, actions))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut replays: [_; BATCH_SIZE] = std::array::from_fn(|_| Vec::new());
    let mut targets = Vec::new();
    let mut context = RndNormalizationContext::new(0.0);

    let mut prev_steps = 0;
    for steps in 0.. {
        if let Some((new_steps, model_path)) = get_model_path_with_most_steps(&args.directory) {
            if new_steps == prev_steps {
                continue;
            }
            prev_steps = new_steps;
            log::info!("Loading new model: {}", model_path.display());
            net = Net::load(model_path, Device::Cuda(0))
                .expect("Path should lead to a valid model file");
        }

        // One simulation batch to initialize root policy if it has not been done yet.
        batched_simulate(
            &mut nodes,
            &envs,
            &net,
            &BETA,
            &mut context,
            &mut actions,
            &mut trajectories,
        );

        // Apply noise.
        nodes
            .iter_mut()
            .zip(&envs)
            .filter(|(_, env)| env.ply < NOISE_PLIES)
            .for_each(|(node, _)| node.apply_dirichlet(&mut rng, NOISE_ALPHA, NOISE_RATIO));

        // Search.
        for _ in 0..VISITS {
            batched_simulate(
                &mut nodes,
                &envs,
                &net,
                &BETA,
                &mut context,
                &mut actions,
                &mut trajectories,
            );
        }

        let selected_actions = select_actions(&mut nodes, &envs, &mut rng);
        take_a_step(&mut nodes, &mut envs, &mut replays, selected_actions);
        restart_envs_and_complete_targets(
            &mut nodes,
            &mut envs,
            &mut replays,
            &mut actions,
            &mut targets,
            &mut rng,
        );

        if !targets.is_empty() {
            save_targets_to_file(&mut targets, &args.directory, steps);
        }
    }
}

/// Create a new target policy based on proportional visit counts.
fn policy_target(node: &Node<Env>) -> Box<[(Move, f32)]> {
    node.children
        .iter()
        .map(|(a, child)| (*a, child.visit_count as f32 / node.visit_count as f32))
        .collect::<Vec<_>>()
        .into_boxed_slice()
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

/// Select which actions to take in environments.
fn select_actions(nodes: &mut [Node<Env>], envs: &[Env], rng: &mut impl Rng) -> Vec<Move> {
    nodes
        .iter_mut()
        .zip(envs)
        .map(|(node, env)| {
            if node.evaluation.is_win() {
                node.children
                    .iter()
                    .min_by_key(|(_, child)| child.evaluation)
                    .expect("there should be at least one child")
                    .0
            } else if env.steps() < WEIGHTED_RANDOM_PLIES {
                let weighted_index =
                    WeightedIndex::new(node.children.iter().map(|(_, child)| child.visit_count))
                        .unwrap();
                node.children[weighted_index.sample(rng)].0
            } else {
                node.children
                    .iter()
                    .max_by_key(|(_, child)| child.visit_count)
                    .expect("there should be at least one child")
                    .0
            }
        })
        .collect::<Vec<_>>()
}

type PolicyTarget = (Env, Box<[(Move, f32)]>);

/// Take a step in each environment.
/// Generate target policy.
fn take_a_step(
    nodes: &mut [Node<Env>],
    envs: &mut [Env],
    replays: &mut [Vec<PolicyTarget>],
    selected_actions: impl IntoIterator<Item = Move>,
) {
    nodes
        .iter_mut()
        .zip(envs)
        .zip(selected_actions)
        .zip(replays)
        .for_each(|(((node, env), action), replay)| {
            replay.push((env.clone(), policy_target(node)));
            node.descend(&action);
            env.step(action);
        });
}

/// Restart any finished environments.
/// Complete targets of finished games using the game result.
fn restart_envs_and_complete_targets(
    nodes: &mut [Node<Env>],
    envs: &mut [Env],
    replays: &mut [Vec<PolicyTarget>],
    actions: &mut [Vec<Move>],
    targets: &mut Vec<Target<Env>>,
    rng: &mut impl Rng,
) {
    nodes
        .iter_mut()
        .zip(envs)
        .zip(replays)
        .zip(actions)
        .for_each(|(((node, env), replay), actions)| {
            if let Some(terminal) = env.terminal() {
                // Reset game.
                *env = Env::new_opening(rng, actions);
                *node = Node::default();

                let mut eval = f32::from(Eval::from(terminal).negate());
                for (env, policy) in replay.drain(..).rev() {
                    targets.push(Target {
                        env,
                        value: eval,
                        ube: 0.0,
                        policy,
                    });
                    eval *= -DISCOUNT_FACTOR;
                }
            }
        });
}

/// Create a new file in the given directory called targets_{steps}.txt
/// with the new targets. Drains the target Vec.
fn save_targets_to_file(targets: &mut Vec<Target<Env>>, directory: &Path, steps: u32) {
    let contents: String = targets.drain(..).map(|target| target.to_string()).collect();
    if let Err(err) = OpenOptions::new()
        .append(true)
        .create(true)
        .open(directory.join(format!("targets_{steps}.txt")))
        .map(|mut file| file.write_all(contents.as_bytes()))
    {
        log::error!(
            "Could not save replays to file [{err}], so here they are instead:\n{contents}"
        );
    }
}
