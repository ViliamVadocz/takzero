use std::{cmp::Reverse, ops::Div};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gumbel};
use rayon::prelude::*;

use super::{
    super::{agent::Agent, env::Environment},
    mcts::Forward,
    policy::sigma,
    Node,
};
use crate::search::eval::Eval;

// TODO: avoid allocating new tensor or moving between cpu/gpu?
fn batched_simulate<E: Environment, A: Agent<E>>(
    nodes: &mut [Node<E>],
    envs: &[E],
    agent: &A,

    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],
) {
    debug_assert!(actions.iter().all(Vec::is_empty));
    debug_assert!(trajectories.iter().all(Vec::is_empty));

    let (indices, batch): (Vec<usize>, Vec<_>) = nodes
        .par_iter_mut()
        .zip(envs)
        .zip(actions.par_iter_mut())
        .zip(trajectories.par_iter_mut())
        .enumerate()
        .filter_map(|(index, (((node, env), actions), trajectory))| {
            match node.forward(trajectory, env.clone()) {
                Forward::Known(eval) => {
                    // If the result is known just propagate it now.
                    node.backward_known_eval(trajectory.drain(..), eval);
                    None
                }
                Forward::NeedsNetwork(env) => {
                    env.populate_actions(actions);
                    Some((index, (env, std::mem::take(actions))))
                    // We are taking the actions because we need owned.
                }
            }
        })
        .unzip();
    let (batch_envs, batch_actions): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
    let output = agent.policy_value(&batch_envs, &batch_actions);

    let borrows: Vec<_> = filter_by_unique_ascending_indices(
        nodes.iter_mut().zip(actions).zip(trajectories.iter_mut()),
        indices,
    )
    .collect();
    borrows
        .into_par_iter()
        .zip(output)
        .zip(batch_actions)
        .for_each(
            |((((node, old_actions), trajectory), (policy, value)), mut actions)| {
                node.backward_network_eval(
                    trajectory.drain(..),
                    actions.drain(..).map(|a| (a.clone(), policy[a])),
                    value,
                );
                // Restore old actions.
                let _ = std::mem::replace(old_actions, actions);
            },
        );
}

/// Filter an iterator by a collection of unique ascending indices.
///
/// # Panics
///
/// Panics if any index is greater than the length of the iterator,
/// or if the indices are not unique and ascending.
pub fn filter_by_unique_ascending_indices<T>(
    mut iter: impl Iterator<Item = T>,
    indices: impl IntoIterator<Item = usize>,
) -> impl Iterator<Item = T> {
    let mut prev = 0;
    indices.into_iter().map(move |i| {
        let res = iter.nth(i - prev).unwrap();
        prev = i + 1;
        res
    })
}

#[allow(
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
/// Perform Sequential halving with Gumbel sampling.
/// Assumes none of the environments are terminal.
pub fn gumbel_sequential_halving<E: Environment, A: Agent<E>, R: Rng>(
    nodes: &mut [Node<E>],
    envs: &[E],
    agent: &A,
    sampled: usize,
    simulations: u32,

    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],
    mut gumbel_noise: Option<(&mut R, &mut [Vec<f32>])>,
) -> Vec<E::Action> {
    debug_assert_eq!(nodes.len(), envs.len());
    debug_assert_eq!(nodes.len(), actions.len());
    debug_assert_eq!(nodes.len(), trajectories.len());
    debug_assert!(actions.iter().all(Vec::is_empty));
    debug_assert!(trajectories.iter().all(Vec::is_empty));
    debug_assert!(envs.iter().all(|env| env.terminal().is_none()));
    #[allow(clippy::debug_assert_with_mut_call)]
    if let Some((_, gumbel_noise)) = &gumbel_noise {
        debug_assert_eq!(nodes.len(), gumbel_noise.len());
        debug_assert!(gumbel_noise.iter().all(Vec::is_empty));
    }

    // Run one simulation on all nodes.
    batched_simulate(nodes, envs, agent, actions, trajectories);

    // Add gumbel noise to policy.
    if let Some((rng, gumbel_noise)) = &mut gumbel_noise {
        let gumbel_distr = Gumbel::new(0.0, 1.0).unwrap();
        let seed = rng.gen();
        gumbel_noise
            .par_iter_mut()
            .zip(nodes.par_iter_mut())
            .enumerate()
            .for_each(|(i, (noise, node))| {
                let mut rng = ChaCha8Rng::from_seed(seed);
                rng.set_stream(i as u64);
                noise.extend(
                    node.children
                        .iter_mut()
                        .zip(gumbel_distr.sample_iter(&mut rng))
                        .map(|((_, child), g)| {
                            child.policy += g;
                            g
                        }),
                );
            });
    }

    // Sequential halving.
    let mut search_sets: Vec<_> = nodes
        .par_iter_mut()
        // Sort and sample.
        .map(|node| {
            node.children.sort_unstable_by_key(|(_, child)| {
                Reverse(child.evaluation.negate().map(|_| child.policy))
            });
            let len = sampled.min(node.children.len());
            &mut node.children[..len]
        })
        .collect();

    let simulations_per_halving = (simulations.div(
        search_sets
            .iter()
            .map(|search_set| search_set.len())
            .min()
            .unwrap()
            .ilog2(),
    )) as usize;
    let mut iteration = 1;
    // Keep halving until only one action remains.
    while search_sets.iter().any(|search_set| search_set.len() > 1) {
        // Simulate.
        for i in 0..simulations_per_halving {
            // Pick out children to simulate.
            let (mut nodes, envs): (Vec<_>, Vec<_>) = search_sets
                .par_iter_mut()
                .zip(envs)
                .map(|(search_set, env)| {
                    let index = i % search_set.len();
                    let action = &search_set[index].0;
                    let child = std::mem::take(&mut search_set[index].1);
                    let mut clone = env.clone();
                    clone.step(action.clone());
                    (child, clone)
                })
                .unzip();
            batched_simulate(&mut nodes, &envs, agent, actions, trajectories);
            // Restore children.
            search_sets
                .par_iter_mut()
                .zip(nodes)
                .for_each(|(search_set, child)| {
                    let index = i % search_set.len();
                    let _ = std::mem::replace(&mut search_set[index].1, child);
                });
        }

        // Sort and sample.
        search_sets.par_iter_mut().for_each(|search_set| {
            let fake_visit_count = (iteration * simulations_per_halving / sampled) as f32;
            search_set.sort_unstable_by_key(|(_, child)| {
                Reverse(
                    child
                        .evaluation
                        .negate()
                        .map(|q| sigma(q, fake_visit_count) + child.policy),
                )
            });
            let len = search_set.len().div(2).max(1);
            *search_set = unsafe { std::mem::transmute(&mut search_set[..len]) };
        });
        iteration += 1;
    }

    let top_actions = search_sets
        .into_par_iter()
        .map(|search_set| search_set.iter().next().unwrap().0.clone())
        .collect();

    // Remove Gumbel noise from policies
    if let Some((_, gumbel_noise)) = &mut gumbel_noise {
        nodes
            .par_iter_mut()
            .zip(gumbel_noise.par_iter_mut())
            .for_each(|(node, noise)| {
                node.children
                    .iter_mut()
                    .zip(noise.drain(..))
                    .for_each(|((_, child), g)| child.policy -= g);
            });
    }

    // Recompute root node statistics.
    nodes.par_iter_mut().for_each(|node| {
        node.visit_count = node
            .children
            .iter()
            .map(|(_, child)| child.visit_count)
            .sum::<u32>()
            + 1;
        let child_evaluations = node.children.iter().map(|(_, child)| child.evaluation);
        node.evaluation = child_evaluations.clone().find(Eval::is_loss).map_or_else(
            || child_evaluations.min().unwrap().negate(), // wrong but whatever
            |loss| loss.negate(),
        );
    });

    top_actions
}

#[cfg(test)]
mod tests {
    use std::array;

    use fast_tak::Game;
    use rand::SeedableRng;
    use rayon::prelude::*;

    use crate::search::{
        agent::dummy::Dummy,
        eval::Eval,
        node::{gumbel::gumbel_sequential_halving, Node},
    };

    #[test]
    fn find_win_with_gumbel() {
        const SAMPLED: usize = 100;
        const SIMULATIONS: u32 = 100;
        const SEED: u64 = 42;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "b3"]);

        let mut nodes = [Node::default()];
        let top = gumbel_sequential_halving(
            &mut nodes,
            &[game],
            &Dummy,
            SAMPLED,
            SIMULATIONS,
            &mut [vec![]],
            &mut [vec![]],
            Some((&mut rng, &mut [vec![]])),
        );
        let top_action = top.into_iter().next().unwrap();
        let root = nodes.into_iter().next().unwrap();

        println!("{root}");
        assert_eq!(top_action, "c1".parse().unwrap());
    }

    #[test]
    fn realize_loss() {
        const SAMPLED: usize = 100;
        const SIMULATIONS: u32 = 1024;
        const SEED: u64 = 123;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "a3>", "c2"]);

        let mut nodes = [Node::default()];
        let top = gumbel_sequential_halving(
            &mut nodes,
            &[game],
            &Dummy,
            SAMPLED,
            SIMULATIONS,
            &mut [vec![]],
            &mut [vec![]],
            Some((&mut rng, &mut [vec![]])),
        );
        let _top_action = top.into_iter().next().unwrap();
        let root = nodes.into_iter().next().unwrap();

        println!("{root}");
        assert_eq!(root.evaluation, Eval::Loss(2));
    }

    #[test]
    fn gumbel_batched() {
        const BATCH_SIZE: usize = 128;
        const SAMPLED: usize = 8;
        const SIMULATIONS: u32 = 128;
        const SEED: u64 = 9099;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut games: [Game<3, 0>; BATCH_SIZE] = array::from_fn(|_| Game::default());
        let mut nodes: [Node<_>; BATCH_SIZE] = array::from_fn(|_| Node::default());
        let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| vec![]);
        let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| vec![]);
        let mut gumbel_noise: [_; BATCH_SIZE] = array::from_fn(|_| vec![]);

        for _ in 0..4 {
            let top_actions = gumbel_sequential_halving(
                &mut nodes,
                &games,
                &Dummy,
                SAMPLED,
                SIMULATIONS,
                &mut actions,
                &mut trajectories,
                Some((&mut rng, &mut gumbel_noise)),
            );

            println!("=== === ===");
            for node in &nodes {
                println!("{node}");
            }

            nodes
                .par_iter_mut()
                .zip(&mut games)
                .zip(top_actions)
                .for_each(|((node, env), action)| {
                    node.descend(&action);
                    env.play(action).unwrap();
                });
        }
    }
}
