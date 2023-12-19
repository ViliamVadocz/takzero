use std::{cmp::Reverse, ops::Div};

use ordered_float::NotNan;
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
use crate::search::{eval::Eval, node::policy::softmax};

// TODO: avoid allocating new tensor or moving between cpu/gpu?
pub fn batched_simulate<E: Environment, A: Agent<E>>(
    nodes: &mut [Node<E>],
    envs: &[E],
    agent: &A,
    betas: &[f32],

    context: &mut A::Context,
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],
) {
    debug_assert!(actions.iter().all(Vec::is_empty));
    debug_assert!(trajectories.iter().all(Vec::is_empty));

    let mut mask = vec![false; nodes.len()];
    let (env_batch, actions_batch): (Vec<_>, Vec<_>) = nodes
        .par_iter_mut()
        .zip(envs)
        .zip(&mut mask)
        .zip(actions.par_iter_mut())
        .zip(trajectories.par_iter_mut())
        .zip(betas.par_iter())
        .map(|(((((node, env), mask), actions), trajectory), beta)| {
            match node.forward(
                trajectory,
                env.clone(),
                #[cfg(not(feature = "baseline"))]
                *beta,
            ) {
                Forward::Known(eval) => {
                    // If the result is known just propagate it now.
                    node.backward_known_eval(trajectory.drain(..), eval);
                    (E::default(), Vec::new())
                }
                Forward::NeedsNetwork(env) => {
                    *mask = true;
                    env.populate_actions(actions);
                    // We are taking the actions we need owned.
                    (env, std::mem::take(actions))
                }
            }
        })
        .unzip();
    #[cfg(feature = "baseline")]
    let output = agent.policy_value(&env_batch, &actions_batch, &mask, context);
    #[cfg(not(feature = "baseline"))]
    let output = agent.policy_value_uncertainty(&env_batch, &actions_batch, &mask, context);
    debug_assert_eq!(output.len(), mask.iter().filter(|x| **x).count());

    nodes
        .iter_mut()
        .zip(actions)
        .zip(trajectories)
        .zip(actions_batch)
        .zip(&mask)
        .filter(|(_, mask)| **mask)
        .zip(output)
        .map(
            |(((((node, old_actions), trajectory), moved_actions), _mask), output)| {
                (node, trajectory, old_actions, moved_actions, output)
            },
        )
        .par_bridge()
        .for_each(
            |(node, trajectory, old_actions, mut moved_actions, output)| {
                #[cfg(feature = "baseline")]
                let (policy, value) = output;
                #[cfg(not(feature = "baseline"))]
                let (policy, value, uncertainty) = output;
                node.backward_network_eval(
                    trajectory.drain(..),
                    moved_actions.drain(..).map(|a| (a.clone(), policy[a])),
                    value,
                    #[cfg(not(feature = "baseline"))]
                    uncertainty,
                );
                // Restore old actions.
                let _ = std::mem::replace(old_actions, moved_actions);
            },
        );
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
    betas: &[f32],

    context: &mut A::Context,
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],
    rng: Option<&mut R>,
) -> Vec<E::Action> {
    debug_assert!(!nodes.is_empty());
    debug_assert_eq!(nodes.len(), envs.len());
    debug_assert_eq!(nodes.len(), actions.len());
    debug_assert_eq!(nodes.len(), trajectories.len());
    debug_assert_eq!(nodes.len(), betas.len());
    debug_assert!(actions.iter().all(Vec::is_empty));
    debug_assert!(trajectories.iter().all(Vec::is_empty));
    debug_assert!(envs.iter().all(|env| env.terminal().is_none()));

    let zeros = vec![0.0; betas.len()];

    // Run one simulation on all nodes.
    batched_simulate(
        nodes,
        envs,
        agent,
        &zeros, // betas, (Only use betas for root)
        context,
        actions,
        trajectories,
    );

    let original_logits_batch: Vec<Vec<_>> = nodes
        .iter()
        .map(|node| {
            node.children
                .iter()
                .map(|(a, child)| (a.clone(), child.logit))
                .collect()
        })
        .collect();
    // Add gumbel noise to policy.
    if let Some(rng) = rng {
        let gumbel_distr = Gumbel::new(0.0, 1.0).unwrap();
        let seed = rng.gen();
        nodes.par_iter_mut().enumerate().for_each(|(i, node)| {
            let mut rng = ChaCha8Rng::from_seed(seed);
            rng.set_stream(i as u64);
            node.children
                .iter_mut()
                .zip(gumbel_distr.sample_iter(&mut rng))
                .for_each(|((_, child), g)| child.logit += g);
        });
    }

    // Sequential halving.
    let mut search_sets: Vec<_> = nodes
        .par_iter_mut()
        // Sort and sample.
        .map(|node| {
            node.children.sort_unstable_by_key(|(_, child)| {
                Reverse(child.evaluation.negate().map(|_| child.logit))
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
            batched_simulate(
                &mut nodes,
                &envs,
                agent,
                &zeros, // betas, (Only use betas for root)
                context,
                actions,
                trajectories,
            );
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
        search_sets
            .par_iter_mut()
            .zip(betas)
            .for_each(|(search_set, beta)| {
                let fake_visit_count = (iteration * simulations_per_halving / sampled) as f32;
                search_set.sort_unstable_by_key(|(_, child)| {
                    Reverse(child.evaluation.negate().map(|q| {
                        sigma(
                            q,
                            #[cfg(not(feature = "baseline"))]
                            child.variance,
                            #[cfg(not(feature = "baseline"))]
                            *beta,
                            fake_visit_count,
                        ) + child.logit
                    }))
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

    // Remove Gumbel noise from logits
    nodes
        .par_iter_mut()
        .zip(original_logits_batch)
        .for_each(|(node, original_logits)| {
            node.children.iter_mut().for_each(|(action, child)| {
                let p = original_logits
                    .iter()
                    .find_map(|(a, p)| (a == action).then_some(p))
                    .unwrap();
                child.logit = *p;
            });
        });

    // Recompute root node statistics.
    nodes.par_iter_mut().zip(betas).for_each(|(node, beta)| {
        node.visit_count = node
            .children
            .iter()
            .map(|(_, child)| child.visit_count)
            .sum::<u32>()
            + 1;

        let evaluations = node.children.iter().map(|(_, child)| &child.evaluation);

        if evaluations.clone().any(Eval::is_loss) || evaluations.clone().all(Eval::is_known) {
            // Node is solved.
            node.evaluation = evaluations.min().unwrap().negate();
            #[cfg(not(feature = "baseline"))]
            {
                node.variance = NotNan::default();
            }
        } else {
            // Slightly different formula than in the Gumbel MuZero paper.
            // Here we are ignoring the original network eval because we no longer have
            // access to it. node.evaluation could be used, but that already includes
            // child evaluations because of tree re-use.
            let policy = softmax(node.children.iter().map(|(_, c)| c.logit));
            let visited_children = node
                .children
                .iter()
                .map(|(_, child)| child)
                .zip(policy)
                .filter(|(child, _)| child.visit_count > 0);
            let sum_policies: NotNan<f32> =
                visited_children.clone().map(|(_, policy)| policy).sum();
            let weighted_q: NotNan<f32> = visited_children
                .clone()
                .map(|(child, policy)| policy * f32::from(child.evaluation.negate()))
                .sum();
            node.evaluation = Eval::new_not_nan_value(weighted_q / sum_policies);

            #[cfg(not(feature = "baseline"))]
            {
                // For variance we use a completely different formula.
                // We take the mean of variances of children with highest value + variance.
                // TODO: Figure out whether this makes any sense.
                let mut visited_children: Vec<_> = visited_children.collect();
                visited_children.sort_unstable_by_key(|(child, _policy)| {
                    child
                        .evaluation
                        .negate()
                        .map(|value| value + beta * child.variance.sqrt())
                });
                let len = visited_children.len();
                let take = len / 4; // TODO: Make into hyperparameter
                node.variance = visited_children
                    .into_iter()
                    .skip(len - take)
                    .map(|(child, _policy)| child.variance)
                    .sum::<NotNan<f32>>()
                    / take as f32;
            }
        }
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
        const BETA: f32 = 0.0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "b3"]);

        let mut nodes = [Node::default()];
        let top = gumbel_sequential_halving(
            &mut nodes,
            &[game],
            &Dummy,
            SAMPLED,
            SIMULATIONS,
            &[BETA],
            &mut (),
            &mut [vec![]],
            &mut [vec![]],
            Some(&mut rng),
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
        const BETA: f32 = 0.0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "a3>", "c2"]);

        let mut nodes = [Node::default()];
        let top = gumbel_sequential_halving(
            &mut nodes,
            &[game],
            &Dummy,
            SAMPLED,
            SIMULATIONS,
            &[BETA],
            &mut (),
            &mut [vec![]],
            &mut [vec![]],
            Some(&mut rng),
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
        const BETA: f32 = 0.0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut games: [Game<3, 0>; BATCH_SIZE] = array::from_fn(|_| Game::default());
        let mut nodes: [Node<_>; BATCH_SIZE] = array::from_fn(|_| Node::default());
        let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| vec![]);
        let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| vec![]);

        for _ in 0..4 {
            let top_actions = gumbel_sequential_halving(
                &mut nodes,
                &games,
                &Dummy,
                SAMPLED,
                SIMULATIONS,
                &[BETA; BATCH_SIZE],
                &mut (),
                &mut actions,
                &mut trajectories,
                Some(&mut rng),
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
