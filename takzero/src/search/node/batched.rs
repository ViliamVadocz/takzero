use std::cmp::Reverse;

use ordered_float::NotNan;
use rand::Rng;
use rand_distr::{Distribution, Gumbel};

use super::Node;
use crate::{
    search::{
        agent::Agent, env::{Environment, Terminal}, eval::Eval, node::{
            mcts::{ActionPolicy, Forward},
            policy::{sigma, softmax},
        }
    },
    target::Replay,
};

// TODO: Use itertools to make the zips nicer.
// TODO: Add rayon later.

pub struct BatchedMCTS<const BATCH_SIZE: usize, E: Environment> {
    nodes: [Node<E>; BATCH_SIZE],
    envs: [E; BATCH_SIZE],
    actions: [Vec<E::Action>; BATCH_SIZE],
    trajectories: [Vec<usize>; BATCH_SIZE],
    replays: [Replay<E>; BATCH_SIZE],
}

impl<const BATCH_SIZE: usize, E: Environment> BatchedMCTS<BATCH_SIZE, E> {
    pub fn new(rng: &mut impl Rng) -> Self {
        let mut actions = Vec::new();
        let envs = std::array::from_fn(|_| E::new_opening(rng, &mut actions));
        Self::from_envs(envs)
    }

    pub fn from_envs(envs: [E; BATCH_SIZE]) -> Self {
        Self {
            nodes: std::array::from_fn(|_| Node::default()),
            actions: std::array::from_fn(|_| Vec::new()),
            trajectories: std::array::from_fn(|_| Vec::new()),
            replays: std::array::from_fn(|i| Replay::new(envs[i].clone())),
            envs,
        }
    }

    pub fn nodes_and_envs(&self) -> impl Iterator<Item = (&Node<E>, &E)> {
        self.nodes.iter().zip(&self.envs)
    }

    pub fn nodes_and_envs_mut(&mut self) -> impl Iterator<Item = (&mut Node<E>, &mut E)> {
        self.nodes.iter_mut().zip(&mut self.envs)
    }

    /// Do a single batched simulation step.
    ///
    /// # Panics
    ///
    /// Panics if the actions or trajectories are not empty.
    /// Also panics if any logit is NaN.
    pub fn simulate<A: Agent<E>>(&mut self, agent: &A, betas: &[f32]) {
        assert!(self.actions.iter().all(Vec::is_empty));
        assert!(self.trajectories.iter().all(Vec::is_empty));

        // Forward pass.
        let (batch, forward): (Vec<_>, Vec<_>) = self
            .nodes
            .iter_mut()
            .zip(&self.envs)
            .zip(&mut self.actions)
            .zip(&mut self.trajectories)
            .zip(betas)
            .filter_map(|((((node, env), actions), trajectory), beta)| {
                match node.forward(trajectory, env.clone(), *beta) {
                    Forward::Known(eval) => {
                        // If the result is known just propagate it now.
                        node.backward_known_eval(trajectory.drain(..), eval);
                        None
                    }
                    Forward::NeedsNetwork(env) => {
                        env.populate_actions(actions);
                        // We are taking the actions because we need owned Vecs.
                        Some(((env, std::mem::take(actions)), (node, trajectory, actions)))
                    }
                }
            })
            .unzip();
        if batch.is_empty() {
            return;
        }

        // Backward pass.
        let (env_batch, actions_batch): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
        let output = agent.policy_value_uncertainty(&env_batch, &actions_batch);
        forward
            .into_iter()
            .zip(
                #[allow(clippy::needless_collect)]
                output.collect::<Vec<_>>(),
            )
            .zip(actions_batch)
            .for_each(|((forward, output), mut moved_actions)| {
                let (node, trajectory, old_actions) = forward;
                let (policy, value, uncertainty) = output;

                // Calculate probabilities from logits.
                let probabilities = softmax(policy.clone().into_iter().map(|(_, p)| p));
                // Do backwards pass.
                node.backward_network_eval(
                    trajectory.drain(..),
                    policy
                        .into_iter()
                        .zip(probabilities)
                        .map(|((action, logit), probability)| ActionPolicy {
                            action,
                            logit,
                            probability,
                        }),
                    value,
                    uncertainty,
                );
                // Restore old actions.
                moved_actions.clear();
                let _ = std::mem::replace(old_actions, moved_actions);
            });
    }

    /// Takes a step in all environments and nodes.
    ///
    /// # Panics
    ///
    /// Panics if there are fewer or more actions than `BATCH_SIZE`.
    pub fn step(&mut self, actions: &[E::Action]) {
        assert_eq!(actions.len(), BATCH_SIZE);
        self.nodes
            .iter_mut()
            .zip(&mut self.envs)
            .zip(&mut self.replays)
            .zip(actions)
            .for_each(|(((node, env), replay), action)| {
                if !node.is_terminal() {
                    node.descend(action);
                    replay.push(action.clone());
                    env.step(action.clone());
                }
            });
    }

    pub fn apply_noise(&mut self, rng: &mut impl Rng, noise_alpha: f32, noise_ratio: f32) {
        self.nodes
            .iter_mut()
            .zip(&self.envs)
            .for_each(|(node, _)| node.apply_dirichlet(rng, noise_alpha, noise_ratio));
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn select_best_actions(&self) -> [E::Action; BATCH_SIZE] {
        self.nodes
            .iter()
            .zip(&self.envs)
            .map(|(node, _)| node.select_best_action())
            .collect::<Vec<_>>()
            .try_into()
            .expect("the number of nodes and envs should be equal to BATCH_SIZE")
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn select_actions_in_selfplay(
        &self,
        rng: &mut impl Rng,
        weighted_random_steps: u16,
    ) -> [E::Action; BATCH_SIZE] {
        self.nodes
            .iter()
            .zip(&self.envs)
            .map(|(node, env)| {
                node.select_selfplay_action(env.steps() < weighted_random_steps, rng)
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("the number of nodes and envs should be equal to BATCH_SIZE")
    }

    pub fn restart_terminal_envs<'a>(
        &'a mut self,
        rng: &'a mut impl Rng,
    ) -> impl Iterator<Item = Option<(Terminal, Replay<E>)>> + 'a {
        self.nodes
            .iter_mut()
            .zip(&mut self.envs)
            .zip(&mut self.actions)
            .zip(&mut self.replays)
            .map(|(((node, env), actions), replay)| {
                let terminal = env.terminal();
                if terminal.is_some() {
                    // Reset game.
                    *env = E::new_opening(rng, actions);
                    *node = Node::default();
                }
                terminal.map(|t| (t, std::mem::replace(replay, Replay::new(env.clone()))))
            })
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::missing_panics_doc)]
    pub fn gumbel_sequential_halving<A: Agent<E>>(
        &mut self,
        agent: &A,
        betas: &[f32],
        sampled_actions: usize,
        search_budget: u32,
        rng: &mut impl Rng,
    ) -> [E::Action; BATCH_SIZE] {
        assert!(sampled_actions > 0, "At least one action must be sampled");
        assert_eq!(
            search_budget % (sampled_actions.ilog2() * sampled_actions as u32),
            0,
            "The search budget should be a multiple of k*log2(k) for clean visits"
        );

        // Do a single batched step to make sure all roots are initialized.
        self.simulate(agent, betas);

        // Generate Gumbel noise.
        let gumbel_distr = Gumbel::new(0.0, 1.0).unwrap();
        let mut gumbel_noise = gumbel_distr.sample_iter(rng);

        // Sample actions based on logits + Gumbel noise.
        let mut selected_sets: Vec<Vec<_>> = self
            .nodes
            .iter_mut()
            .map(|node| {
                let mut selected_set: Vec<_> = node
                    .children
                    .iter_mut()
                    .zip(gumbel_noise.by_ref())
                    .map(|((a, child), gumbel_noise)| (child.logit + gumbel_noise, a, child))
                    .collect();
                selected_set.sort_by_key(|(x, ..)| Reverse(*x));
                selected_set.truncate(sampled_actions);
                selected_set
            })
            .collect();

        let steps = sampled_actions.ilog2();
        let visits_per_step = search_budget / steps;
        let mut visits_to_most_visited_action = 0;
        let mut remaining_actions = sampled_actions;

        for _ in 0..steps {
            let visits_per_action = visits_per_step / remaining_actions as u32;

            for i in 0..remaining_actions {
                let mut nodes_and_envs: Vec<_> = selected_sets
                    .iter_mut()
                    .zip(&self.envs)
                    .map(|(set, env)| {
                        let mut env = env.clone();
                        let i: usize = i % set.len();
                        env.step(set[i].1.clone());
                        (&mut *set[i].2, env)
                    })
                    .collect();
                for _ in 0..visits_per_action {
                    // TODO: Refactor this and `simulate()` into one function.
                    // =========================================================================================

                    assert!(self.actions.iter().all(Vec::is_empty));
                    assert!(self.trajectories.iter().all(Vec::is_empty));

                    // Forward pass.
                    let (batch, forward): (Vec<_>, Vec<_>) = nodes_and_envs
                        .iter_mut()
                        .zip(&mut self.actions)
                        .zip(&mut self.trajectories)
                        .zip(betas)
                        .filter_map(|((((node, env), actions), trajectory), beta)| {
                            match node.forward(trajectory, env.clone(), *beta) {
                                Forward::Known(eval) => {
                                    // If the result is known just propagate it now.
                                    node.backward_known_eval(trajectory.drain(..), eval);
                                    None
                                }
                                Forward::NeedsNetwork(env) => {
                                    env.populate_actions(actions);
                                    // We are taking the actions because we need owned Vecs.
                                    Some((
                                        (env, std::mem::take(actions)),
                                        (node, trajectory, actions),
                                    ))
                                }
                            }
                        })
                        .unzip();
                    if batch.is_empty() {
                        continue;
                    }

                    // Backward pass.
                    let (env_batch, actions_batch): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
                    let output = agent.policy_value_uncertainty(&env_batch, &actions_batch);
                    forward
                        .into_iter()
                        .zip(
                            #[allow(clippy::needless_collect)]
                            output.collect::<Vec<_>>(),
                        )
                        .zip(actions_batch)
                        .for_each(|((forward, output), mut moved_actions)| {
                            let (node, trajectory, old_actions) = forward;
                            let (policy, value, uncertainty) = output;

                            // Calculate probabilities from logits.
                            let probabilities = softmax(policy.clone().into_iter().map(|(_, p)| p));
                            // Do backwards pass.
                            node.backward_network_eval(
                                trajectory.drain(..),
                                policy.into_iter().zip(probabilities).map(
                                    |((action, logit), probability)| ActionPolicy {
                                        action,
                                        logit,
                                        probability,
                                    },
                                ),
                                value,
                                uncertainty,
                            );
                            // Restore old actions.
                            moved_actions.clear();
                            let _ = std::mem::replace(old_actions, moved_actions);
                        });

                    // =========================================================================================
                }
            }

            visits_to_most_visited_action += visits_per_action;
            remaining_actions /= 2;

            // Halve the number of actions.
            for (selected_set, &beta) in selected_sets.iter_mut().zip(betas) {
                selected_set.sort_by_key(|(logits_plus_gumbel, _, child)| {
                    Reverse(
                        logits_plus_gumbel
                            + sigma(
                                child.evaluation.negate().into(),
                                child.std_dev,
                                beta,
                                visits_to_most_visited_action as f32,
                            ),
                    )
                });
                selected_set.truncate(remaining_actions);
            }
        }

        let selected = selected_sets
            .into_iter()
            .map(|mut selected_set| {
                assert_eq!(
                    selected_set.len(),
                    1,
                    "After sequential halving, every set should have exactly 1 action left"
                );
                selected_set.pop().unwrap().1.clone()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Recompute root statistics.
        self.nodes.iter_mut().for_each(|node| {
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
                node.std_dev = NotNan::default();
            } else {
                // Slightly different formula than in the Gumbel MuZero paper.
                // Here we are ignoring the original network eval because we no longer have
                // access to it.
                let visited_children = node
                    .children
                    .iter()
                    .map(|(_, child)| child)
                    .filter(|child| child.visit_count > 0);
                let sum_of_probabilities: NotNan<f32> =
                    visited_children.clone().map(|child| child.probability).sum();
                let weighted_q: NotNan<f32> = visited_children
                    .map(|child| child.probability * f32::from(child.evaluation.negate()))
                    .sum();
                node.evaluation = Eval::new_not_nan_value(weighted_q / sum_of_probabilities);
            }

            // FIXME: std_dev is not recomputed
        });

        selected
    }
}
