use ordered_float::NotNan;

use super::{
    super::{env::Environment, eval::Eval, agent::Agent},
    Node,
};

/// Return value from [`Node::forward`] indicating if the evaluation is known
/// or if it needs to be propagated.
#[must_use]
pub enum Forward<E: Environment> {
    Known(Eval),
    NeedsNetwork(E),
}

impl<E: Environment> Node<E> {
    fn update_mean_value(&mut self, value: f32) {
        if let Eval::Value(mean_value) = &mut self.evaluation {
            *mean_value = NotNan::new(
                mean_value
                    .into_inner()
                    .mul_add((self.visit_count - 1) as f32, value)
                    / self.visit_count as f32,
            )
            .expect("value should not be nan");
        } else {
            unreachable!("updating the mean value doesn't make sense if the result is known");
        };
    }

    fn propagate_child_eval(&mut self, child_eval: Eval) -> Eval {
        let evaluations = self.children.iter().map(|(_, node)| node.evaluation);
        match child_eval {
            // This move made the opponent lose, so this position is a win.
            Eval::Loss(_) => {
                self.evaluation = child_eval.negate();
                self.evaluation
            }

            // If all moves lead to wins for the opponent, this node is a loss.
            // If all moves lead to wins or draws for the opponent, we choose to draw.
            Eval::Draw(_) | Eval::Win(_) if evaluations.clone().all(|e| e.is_known()) => {
                self.evaluation = evaluations.min().unwrap().negate();
                self.evaluation
            }

            // Otherwise this position is not know and we just back-propagate the child result.
            _ => {
                let negated = child_eval.negate().into();
                self.update_mean_value(negated);
                Eval::new_value(negated).unwrap()
            }
        }
    }

    /// Run the forward part of MCTS.
    /// One of `backward_known_eval` and `backward_network_eval`
    /// must be called afterwards.
    pub fn forward(&mut self, trajectory: &mut Vec<usize>, mut env: E) -> Forward<E> {
        debug_assert!(trajectory.is_empty());
        let mut node = self;

        loop {
            node.visit_count += 1; // TODO: virtual visit?
            if node.evaluation.is_known() {
                break Forward::Known(node.evaluation);
            }
            if node.needs_initialization() {
                if let Some(terminal) = env.terminal() {
                    node.evaluation = terminal.into();
                    break Forward::Known(node.evaluation);
                }
                break Forward::NeedsNetwork(env);
            }

            let index = node.select_with_improved_policy();
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
    ) -> Eval {
        if let Some(index) = trajectory.next() {
            let child_eval = self.children[index].1.backward_known_eval(trajectory, eval);
            self.propagate_child_eval(child_eval)
        } else {
            // Leaf reached, time to propagate upwards.
            eval
        }
    }

    /// Initialize a leaf node and propagate a network evaluation through the
    /// tree.
    pub fn backward_network_eval(
        &mut self,
        mut trajectory: impl Iterator<Item = usize>,
        policy: impl Iterator<Item = (E::Action, f32)>,
        value: f32,
    ) -> Eval {
        if let Some(index) = trajectory.next() {
            let child_eval = self.children[index]
                .1
                .backward_network_eval(trajectory, policy, value);
            self.propagate_child_eval(child_eval)
        } else {
            // Finish leaf initialization.
            self.children = policy.map(|(a, p)| (a, Self::from_policy(p))).collect();
            self.evaluation = Eval::new_value(value).unwrap_or_else(|_| {
                log::warn!("value NaN");
                Eval::default()
            });
            self.evaluation
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn simulate_simple<A: Agent<E>>(&mut self, agent: &A, env: E) -> Eval {
        let mut trajectory = Vec::new();
        match self.forward(&mut trajectory, env) {
            Forward::Known(eval) => self.backward_known_eval(trajectory.into_iter(), eval),
            Forward::NeedsNetwork(env) => {
                let mut actions = [Vec::new()];
                env.populate_actions(&mut actions[0]);
                let (policy, value) = agent.policy_value(&[env], &actions).pop().unwrap();
                self.backward_network_eval(
                    trajectory.into_iter(),
                    actions
                        .into_iter()
                        .next()
                        .unwrap()
                        .into_iter()
                        .map(|a| (a.clone(), policy[a])),
                    value,
                )
            }
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

    #[test]
    fn find_tinue_easy() {
        const MAX_VISITS: usize = 5_000;

        // https://ptn.ninja/NoZQlgLgpgBARABwgOwHTLMgVgQzgXQFgAoUMAL1jgGYCTgB5BKDZAc3gGcB3HBO0gBEc0eACYADGOqoJAdlQBGAGwDgAFTABbKpIBcYgBx7qAVjUAlKJwCuAGwjwLAWgkCSi1DBzUYAY0USMS8-MX9qEhkYACNfP2pnIA&name=MwD2Q&ply=5!
        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "c1", "c2", "c3", "b3", "c3-"]);
        let mut root = Node::default();

        (0..MAX_VISITS)
            .find(|_| matches!(root.simulate_simple(&Dummy, game.clone()), Eval::Win(_)))
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

        // https://ptn.ninja/NoZQlgLgpgBARABwgOwHTLMgVgQzgXQFgAoUMAL1jgGYCTgB5BKDZAc3gGcB3HBO0gBEc0eACYADGOqoJAdlQBGAGwDgAFTABbKpIBcYgBx7qAVjUAlKJwCuAGwjwLAWgkCSi1DBzVvikmJeAEaKMADGikA&name=MwD2Q&ply=3!
        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "c1"]);
        let mut root = Node::default();

        (0..MAX_VISITS)
            .find(|_| matches!(root.simulate_simple(&Dummy, game.clone()), Eval::Win(_)))
            .expect("This position is solvable with MAX_VISITS.");

        println!("{root}");
        assert_eq!(
            root.children
                .iter()
                .find(|(_, node)| node.evaluation.is_loss())
                .unwrap()
                .0,
            "b2".parse().unwrap(), // Maybe also c2?
        );
    }
}
