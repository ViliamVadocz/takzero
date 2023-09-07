use ordered_float::NotNan;


use super::{
    super::{DISCOUNT_FACTOR, env::Environment, eval::Eval, agent::Agent},
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
    #[inline]
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
    
    #[inline]
    fn update_variance(&mut self, uncertainty: f32) {
        self.variance = (self.variance * (self.visit_count - 1) as f32 + uncertainty) / self.visit_count as f32;
    }

    fn propagate_child_eval(&mut self, child_eval: Eval, child_uncertainty: f32) -> (Eval, f32) {
        let evaluations = self.children.iter().map(|(_, node)| node.evaluation);
        match child_eval {
            // This move made the opponent lose, so this position is a win.
            Eval::Loss(_) => {
                self.evaluation = child_eval.negate();
                self.variance = 0.0;
                (self.evaluation, self.variance)
            }

            // If all moves lead to wins for the opponent, this node is a loss.
            // If all moves lead to wins or draws for the opponent, we choose to draw.
            Eval::Draw(_) | Eval::Win(_) if evaluations.clone().all(|e| e.is_known()) => {
                self.evaluation = evaluations.min().unwrap().negate();
                self.variance = 0.0;
                (self.evaluation, self.variance)
            }

            // Otherwise this position is not know and we just back-propagate the child result.
            _ => {
                let negated = child_eval.negate().into();
                self.update_mean_value(negated);
                self.update_variance(child_uncertainty);
                (Eval::new_value(negated * DISCOUNT_FACTOR).unwrap(), child_uncertainty * DISCOUNT_FACTOR * DISCOUNT_FACTOR)
            }
        }
    }

    /// Run the forward part of MCTS.
    /// One of `backward_known_eval` and `backward_network_eval`
    /// must be called afterwards.
    pub fn forward(&mut self, trajectory: &mut Vec<usize>, mut env: E, beta: f32) -> Forward<E> {
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

            let index = node.select_with_improved_policy(beta);
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
    ) -> (Eval, f32) {
        if let Some(index) = trajectory.next() {
            let (child_eval, child_uncertainty) = self.children[index].1.backward_known_eval(trajectory, eval);
            self.propagate_child_eval(child_eval, child_uncertainty)
        } else {
            // Leaf reached, time to propagate upwards.
            (eval, 0.0)
        }
    }

    /// Initialize a leaf node and propagate a network evaluation through the
    /// tree.
    pub fn backward_network_eval(
        &mut self,
        mut trajectory: impl Iterator<Item = usize>,
        policy: impl Iterator<Item = (E::Action, f32)>,
        value: f32,
        uncertainty: f32,
    ) -> (Eval, f32) {
        if let Some(index) = trajectory.next() {
            let (child_eval, child_uncertainty) = self.children[index]
                .1
                .backward_network_eval(trajectory, policy, value, uncertainty);
            self.propagate_child_eval(child_eval, child_uncertainty)
        } else {
            // Finish leaf initialization.
            self.children = policy.map(|(a, p)| (a, Self::from_policy(p))).collect();
            self.propagate_child_eval(Eval::new_value(value).unwrap_or_else(|_| {
                log::warn!("value NaN");
                Eval::default()
            }), uncertainty) 
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn simulate_simple<A: Agent<E>>(&mut self, agent: &A, env: E, beta: f32) -> (Eval, f32) {
        let mut trajectory = Vec::new();
        match self.forward(&mut trajectory, env, beta) {
            Forward::Known(eval) => self.backward_known_eval(trajectory.into_iter(), eval),
            Forward::NeedsNetwork(env) => {
                let mut actions = [Vec::new()];
                env.populate_actions(&mut actions[0]);
                let (policy, value, uncertainty) = agent.policy_value_uncertainty(&[env], &actions).pop().unwrap();
                self.backward_network_eval(
                    trajectory.into_iter(),
                    actions
                        .into_iter()
                        .next()
                        .unwrap()
                        .into_iter()
                        .map(|a| (a.clone(), policy[a])),
                    value,
                    uncertainty
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
            .find(|_| matches!(root.simulate_simple(&Dummy, game.clone(), 1.0), (Eval::Win(_), _)))
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
            .find(|_| matches!(root.simulate_simple(&Dummy, game.clone(), 1.0), (Eval::Win(_), _)))
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
