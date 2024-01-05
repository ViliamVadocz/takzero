use ordered_float::NotNan;

use super::Node;
use crate::search::{
    agent::Agent,
    env::Environment,
    node::{
        mcts::{ActionPolicy, Forward},
        policy::softmax,
    },
};

// TODO: avoid allocating new tensor or moving between cpu/gpu?
#[allow(clippy::missing_panics_doc)]
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
        .iter_mut()
        .zip(envs)
        .zip(&mut mask)
        .zip(actions.iter_mut())
        .zip(trajectories.iter_mut())
        .zip(betas.iter())
        .map(|(((((node, env), mask), actions), trajectory), beta)| {
            match node.forward(trajectory, env.clone(), *beta) {
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
        .for_each(
            |(node, trajectory, old_actions, mut moved_actions, output)| {
                let (policy, value, uncertainty) = output;
                let logits = moved_actions
                    .iter()
                    .map(|a| NotNan::new(policy[a.clone()]).expect("logit should not be NaN"))
                    .collect::<Vec<_>>();
                let probabilities = softmax(logits.clone().into_iter());
                node.backward_network_eval(
                    trajectory.drain(..),
                    moved_actions.drain(..).zip(logits).zip(probabilities).map(
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
                let _ = std::mem::replace(old_actions, moved_actions);
            },
        );
}
