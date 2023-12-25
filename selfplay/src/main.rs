use std::{fmt, fs::OpenOptions, io::Write};

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

fn main_() {
    let seed: u64 = rand::thread_rng().gen();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let net = Net::new(Device::Cuda(0), Some(rng.gen()));

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

    loop {
        println!("step!");

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
            )
        }

        // Select actions.
        let selected_actions = nodes
            .iter_mut()
            .zip(&envs)
            .map(|(node, env)| {
                // println!("{node}");
                if node.evaluation.is_win() {
                    node.children
                        .iter()
                        .min_by_key(|(_, child)| child.evaluation)
                        .expect("there should be at least one child")
                        .0
                } else if env.steps() < WEIGHTED_RANDOM_PLIES {
                    let weighted_index = WeightedIndex::new(
                        node.children.iter().map(|(_, child)| child.visit_count),
                    )
                    .unwrap();
                    node.children[weighted_index.sample(&mut rng)].0
                } else {
                    node.children
                        .iter()
                        .max_by_key(|(_, child)| child.visit_count)
                        .expect("there should be at least one child")
                        .0
                }
            })
            .collect::<Vec<_>>();

        // Take a step and generate target policy.
        nodes
            .iter_mut()
            .zip(&mut envs)
            .zip(selected_actions)
            .zip(&mut replays)
            .for_each(|(((node, env), action), replay)| {
                replay.push((env.clone(), policy_target(node)));
                node.descend(&action);
                env.step(action);
            });

        // Restart finished games. Complete targets with value.
        nodes
            .iter_mut()
            .zip(&mut envs)
            .zip(&mut replays)
            .zip(&mut actions)
            .for_each(|(((node, env), replay), actions)| {
                if let Some(terminal) = env.terminal() {
                    // Reset game.
                    *env = Env::new_opening(&mut rng, actions);
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

        // Save targets to file.
        if !targets.is_empty() {
            let contents: String = targets.drain(..).map(|target| target.to_string()).collect();
            if let Err(err) = OpenOptions::new()
                .append(true)
                .create(true)
                .open("targets.txt")
                .map(|mut file| file.write_all(contents.as_bytes()))
            {
                println!(
                    "could not save replays to file [{err}], so here they are instead:\n{contents}"
                )
            }
        }
    }
}

fn policy_target(node: &Node<Env>) -> Box<[(Move, f32)]> {
    node.children
        .iter()
        .map(|(a, child)| (*a, child.visit_count as f32 / node.visit_count as f32))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}
