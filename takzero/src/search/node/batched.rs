use super::Node;
use crate::search::{
    agent::Agent,
    env::Environment,
    node::{
        mcts::{ActionPolicy, Forward},
        policy::softmax,
    },
};

// TODO: Use itertools to make the zips nicer.
// TODO: Add rayon later.

pub struct BatchedMCTS<const BATCH_SIZE: usize, E: Environment, A: Agent<E>> {
    nodes: [Node<E>; BATCH_SIZE],
    envs: [E; BATCH_SIZE],
    actions: [Vec<E::Action>; BATCH_SIZE],
    trajectories: [Vec<usize>; BATCH_SIZE],
    betas: [f32; BATCH_SIZE],
    context: A::Context,
}

impl<const BATCH_SIZE: usize, E: Environment, A: Agent<E>> BatchedMCTS<BATCH_SIZE, E, A> {
    pub fn new(envs: [E; BATCH_SIZE], betas: [f32; BATCH_SIZE], context: A::Context) -> Self {
        Self {
            nodes: std::array::from_fn(|_| Node::default()),
            envs,
            actions: std::array::from_fn(|_| Vec::new()),
            trajectories: std::array::from_fn(|_| Vec::new()),
            betas,
            context,
        }
    }

    /// Do a single batched simulation step.
    ///
    /// # Panics
    ///
    /// Panics if the actions or trajectories are not empty.
    /// Also panics if any logit is NaN.
    pub fn simulate(&mut self, agent: &A) {
        assert!(self.actions.iter().all(Vec::is_empty));
        assert!(self.trajectories.iter().all(Vec::is_empty));

        let (batch, forward): (Vec<_>, Vec<_>) = self
            .nodes
            .iter_mut()
            .zip(&self.envs)
            .zip(&mut self.actions)
            .zip(&mut self.trajectories)
            .zip(&self.betas)
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
        let (env_batch, actions_batch): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
        let output = agent.policy_value_uncertainty(&env_batch, &actions_batch, &mut self.context);
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

    pub fn step(&mut self, actions: [E::Action; BATCH_SIZE]) {
        self.nodes
            .iter_mut()
            .zip(&mut self.envs)
            .zip(actions)
            .for_each(|((node, env), action)| {
                node.descend(&action);
                env.step(action);
            });
    }

    pub const fn nodes(&self) -> &[Node<E>] {
        &self.nodes
    }

    pub const fn envs(&self) -> &[E] {
        &self.envs
    }
}
