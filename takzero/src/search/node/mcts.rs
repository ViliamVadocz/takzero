//! Monte Carlo Tree Search
//!
//! Each node stores the evaluation for a particular position.
//! This means that when choosing an action we minimize
//! the child evaluations (picking the action that results
//! in the worst position for our opponent).
//! During backpropagation we receive the child evalation
//! and negate it before using it to update the parent.
//!
//! The search is implemented in two steps, `forward` and
//! `backward_known_eval`+`backward_network_eval` to allow for batch
//! evaluations.
//!
//! A node solver has been implemented as well. It deduces
//! when a result (win, loss, or draw) is guaranteed.
//! The idea behind it is that if on of the children is a loss,
//! this node is a win, or if all children are wins then
//! this is a loss.

use ordered_float::NotNan;

use super::{
    super::{agent::Agent, env::Environment, eval::Eval, DISCOUNT_FACTOR},
    policy::softmax,
    Node,
};

/// Return value from [`Node::forward`] indicating if the evaluation is known
/// or if it needs to be propagated.
#[must_use]
pub enum Forward<E: Environment> {
    Known(Eval),
    NeedsNetwork(E),
}

pub struct Propagated {
    eval: Eval,
    variance: NotNan<f32>,
}

pub struct ActionPolicy<E: Environment> {
    pub action: E::Action,
    pub logit: NotNan<f32>,
    pub probability: NotNan<f32>,
}

impl<E: Environment> Node<E> {
    #[inline]
    fn update_mean_value(&mut self, value: f32) {
        if let Eval::Value(mean_value) = &mut self.evaluation {
            *mean_value += (-*mean_value + value) / (self.visit_count as f32);
        }
    }

    #[inline]
    fn update_standard_deviation(&mut self, variance: NotNan<f32>) {
        if self.evaluation.is_known() {
            return;
        }
        self.std_dev += (-self.std_dev + variance.sqrt()) / (self.visit_count as f32);
    }

    // TODO: Once pruning is added back, we can skip traversing all evaluations to
    // find min in the case of a loss because that is will be the first loss we
    // find.
    fn node_solver(&mut self, child_eval: Eval) {
        let evaluations = self.children.iter().map(|(_, node)| node.evaluation);

        // If we can choose a loss for the opponent, this position is a win.
        // If all moves are wins for the opponent, this node is a loss.
        // If all moves are wins or draws for the opponent, we choose to draw.
        if child_eval.is_loss() || evaluations.clone().all(|e| e.is_known()) {
            self.evaluation = evaluations.min().unwrap().negate();
            self.std_dev = NotNan::default();
        }
    }

    fn propagate_child_eval(
        &mut self,
        child_eval: Eval,
        child_variance: NotNan<f32>,
    ) -> Propagated {
        self.node_solver(child_eval);

        // If the position is solved, we just propagate the solved value instead.
        if self.evaluation.is_known() {
            return Propagated {
                eval: self.evaluation,
                variance: self.std_dev * self.std_dev,
            };
        }
        // Otherwise this position is not known and we just
        // back-propagate the child result.
        let negated = child_eval.negate().into();
        self.update_mean_value(negated);
        self.update_standard_deviation(child_variance);

        Propagated {
            eval: Eval::new_value(negated * DISCOUNT_FACTOR).unwrap(),
            variance: child_variance * DISCOUNT_FACTOR * DISCOUNT_FACTOR,
        }
    }

    /// Run the forward part of MCTS.
    /// One of `backward_known_eval` and `backward_network_eval`
    /// must be called afterwards.
    pub fn forward(&mut self, trajectory: &mut Vec<usize>, mut env: E, beta: f32) -> Forward<E> {
        debug_assert!(trajectory.is_empty());
        let mut node = self;

        loop {
            node.visit_count += 1;
            #[cfg(feature = "virtual")]
            {
                node.virtual_visits += 1;
            }
            // TODO: Prune all known results earlier
            // once visit count is not used for policy target.
            // Or don't - searching can still help find slower losses.
            if node.is_terminal() {
                break Forward::Known(node.evaluation);
            }
            if node.needs_initialization() {
                if let Some(terminal) = env.terminal() {
                    node.evaluation = terminal.into();
                    node.std_dev = NotNan::default();
                    break Forward::Known(node.evaluation);
                }
                break Forward::NeedsNetwork(env);
            }

            let index = node.select_with_puct(beta);
            trajectory.push(index);
            let (action, child) = &mut node.children[index];
            env.step(action.clone());
            node = child;
        }
    }

    /// Propagate a known eval through the tree.
    pub fn backward_known_eval(
        &mut self,
        mut trajectory: impl Iterator<Item = usize>,
        eval: Eval,
    ) -> Propagated {
        #[cfg(feature = "virtual")]
        {
            self.virtual_visits -= 1;
        }
        if let Some(index) = trajectory.next() {
            let Propagated {
                eval: child_eval,
                variance: child_variance,
            } = self.children[index].1.backward_known_eval(trajectory, eval);
            self.propagate_child_eval(child_eval, child_variance)
        } else {
            // Leaf reached, time to propagate upwards.
            Propagated {
                eval,
                variance: NotNan::default(),
            }
        }
    }

    /// Initialize a leaf node and propagate a network evaluation
    /// through the tree.
    ///
    /// # Panics
    ///
    /// Panics if any of the policies is NaN.
    pub fn backward_network_eval(
        &mut self,
        mut trajectory: impl Iterator<Item = usize>,
        policy: impl Iterator<Item = ActionPolicy<E>>,
        value: f32,
        variance: f32,
    ) -> Propagated {
        #[cfg(feature = "virtual")]
        {
            self.virtual_visits -= 1;
        }
        if let Some(index) = trajectory.next() {
            let Propagated {
                eval: child_eval,
                variance: child_variance,
            } = self.children[index]
                .1
                .backward_network_eval(trajectory, policy, value, variance);
            self.propagate_child_eval(child_eval, child_variance)
        } else {
            // Update mean value and standard deviation.
            // Note that this is not the same as self.propagate_child_eval()
            // because we do not negate!
            self.update_mean_value(value);
            let variance = NotNan::new(variance).expect("uncertainty/variance should not be NaN");
            self.update_standard_deviation(variance);

            // Finish leaf initialization.
            self.children = policy
                .map(
                    |ActionPolicy {
                         action,
                         logit,
                         probability,
                     }| {
                        (
                            action,
                            Self::from_logit_and_probability_and_parent_value_and_std_dev(
                                logit,
                                probability,
                                self.evaluation.into(),
                                self.std_dev,
                            ),
                        )
                    },
                )
                .collect();

            Propagated {
                eval: Eval::new_value(value * DISCOUNT_FACTOR)
                    .expect("value prediction should not be NaN"),
                variance: variance * DISCOUNT_FACTOR * DISCOUNT_FACTOR,
            }
        }
    }

    /// A non-batched version of simulate that does both
    /// forward and backward steps of MCTS. This is mainly
    /// used for testing.
    ///
    /// # Panics
    ///
    /// Panics if the agent does not return a prediction
    /// when needed.
    pub fn simulate_simple<A: Agent<E>>(&mut self, agent: &A, env: E, beta: f32) -> Propagated {
        let mut trajectory = Vec::new();
        match self.forward(&mut trajectory, env, beta) {
            Forward::Known(eval) => self.backward_known_eval(trajectory.into_iter(), eval),
            Forward::NeedsNetwork(env) => {
                let mut actions = [Vec::new()];
                env.populate_actions(&mut actions[0]);
                let (policy, value, uncertainty) = agent
                    .policy_value_uncertainty(&[env], &actions)
                    .next()
                    .expect("agent should return exactly one prediction");
                // Calculate probabilities from logits.
                let probabilities = softmax(policy.clone().into_iter().map(|(_, p)| p));
                // Do backwards pass.
                self.backward_network_eval(
                    trajectory.into_iter(),
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
                )
            }
        }
    }

    /// A batched version of simulate that does both forward
    /// and backward steps of MCTS on a single node.
    pub fn simulate_batch<A: Agent<E>>(
        &mut self,
        agent: &A,
        env: &E,
        beta: f32,
        batch_size: usize,
    ) {
        let mut trajectories = Vec::with_capacity(batch_size);
        let mut actionss = Vec::with_capacity(batch_size);
        let mut envs = Vec::with_capacity(batch_size);

        // Limit forward passes to some arbitrary number larger than batch_size to
        // ensure hitting known nodes doesn't usually make the batch smaller.
        for _ in 0..batch_size * 4 {
            let mut trajectory = Vec::new();
            match self.forward(&mut trajectory, env.clone(), beta) {
                Forward::Known(eval) => _ = self.backward_known_eval(trajectory.into_iter(), eval),
                Forward::NeedsNetwork(env) => {
                    trajectories.push(trajectory);

                    let mut actions = Vec::new();
                    env.populate_actions(&mut actions);
                    actionss.push(actions);

                    envs.push(env);
                }
            }

            // Cut the loop short if we filled our batch (typical termination condition).
            if trajectories.len() == batch_size {
                break;
            }
        }

        // Ensure we don't call `policy_value_uncertainty` with empty inputs
        if trajectories.is_empty() {
            return;
        }

        for ((policy, value, uncertainty), trajectory) in agent
            .policy_value_uncertainty(&envs, &actionss)
            .zip(trajectories)
        {
            // Calculate probabilities from logits.
            let probabilities = softmax(policy.clone().into_iter().map(|(_, p)| p));
            // Do backwards pass.
            self.backward_network_eval(
                trajectory.into_iter(),
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
        }
    }
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;

    use super::super::{
        super::{agent::dummy::Dummy, eval::Eval},
        Node,
    };
    use crate::search::{
        agent::simple::Simple,
        env::safecrack::{SafeCrack, SafeCracker},
        node::mcts::Propagated,
    };

    #[test]
    fn find_tinue_easy() {
        const MAX_VISITS: usize = 5_000;

        // <https://ptn.ninja/NoZQlgLgpgBARABwgOwHTLMgVgQzgXQFgAoUMAL1jgGYCTgB5BKDZAc3gGcB3
        // HBO0gBEc0eACYADGOqoJAdlQBGAGwDgAFTABbKpIBcYgBx7qAVjUAlKJwCuAGwjwLAWgkCSi1DBzUYAY
        // 0USMS8-MX9qEhkYACNfP2pnIA&name=MwD2Q&ply=5!>
        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "c1", "c2", "c3", "b3", "c3-"]);
        let mut root = Node::default();

        (0..MAX_VISITS)
            .find(|_| {
                matches!(
                    root.simulate_simple(&Dummy, game.clone(), 1.0),
                    Propagated {
                        eval: Eval::Win(_),
                        ..
                    }
                )
            })
            .expect("This position is solvable with MAX_VISITS.");

        println!("{root}");
        assert_eq!(
            root.children
                .iter()
                .find(|(_, node)| node.evaluation.is_loss())
                .unwrap()
                .0,
            "b1".parse().unwrap(),
        );
    }

    #[test]
    fn find_tinue_deeper() {
        const MAX_VISITS: usize = 50_000;

        // <https://ptn.ninja/NoZQlgLgpgBARABwgOwHTLMgVgQzgXQFgAoUMAL1jgGYCTgB5BKDZAc3gGcB3
        // HBO0gBEc0eACYADGOqoJAdlQBGAGwDgAFTABbKpIBcYgBx7qAVjUAlKJwCuAGwjwLAWgkCSi1DBzVvik
        // mJeAEaKMADGikA&name=MwD2Q&ply=3!>
        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "c1"]);
        let mut root = Node::default();

        (0..MAX_VISITS)
            .find(|i| {
                if i % 10_000 == 0 {
                    println!("{root}");
                }
                matches!(
                    root.simulate_simple(&Simple, game.clone(), 1.0),
                    Propagated {
                        eval: Eval::Win(_),
                        ..
                    }
                )
            })
            .expect("This position is solvable with MAX_VISITS.");

        println!("{root}");
        let winning_move = root
            .children
            .iter()
            .find(|(_, node)| node.evaluation.is_loss())
            .unwrap()
            .0;
        assert!(winning_move == "b2".parse().unwrap() || winning_move == "c2".parse().unwrap());
    }

    #[test]
    fn safe_cracker_value_propagation() {
        const VISITS: usize = 100_000;
        const KEY: [u8; 5] = [0, 1, 2, 3, 4];
        let env = SafeCrack::new(KEY.to_vec());
        let mut root = Node::default();

        assert!(f32::from(root.evaluation) == 0.0);
        for _ in 0..VISITS {
            root.simulate_simple(&SafeCracker, env.clone(), 0.0);
        }

        for k in KEY {
            println!("eval: {}", root.evaluation);
            for (action, child) in &root.children {
                println!("\t{}: eval: {}", action.unwrap(), child.evaluation);
            }
            assert!(f32::from(root.evaluation) > 0.0);

            for (action, child) in &root.children {
                if action.is_some_and(|x| x == k) {
                    assert!(f32::from(child.evaluation) < 0.0);
                } else {
                    assert!(f32::from(child.evaluation) == 0.0);
                }
            }

            root.descend(&Some(k));
            root.descend(&None);
        }

        assert!(f32::from(root.evaluation) > 0.0);
    }
}
