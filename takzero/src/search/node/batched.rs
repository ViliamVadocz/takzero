use rand::Rng;

use super::Node;
use crate::{
    search::{
        agent::Agent,
        env::{Environment, Terminal},
        node::{
            mcts::{ActionPolicy, Forward},
            policy::softmax,
        },
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
}
