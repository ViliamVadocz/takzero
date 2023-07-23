use super::{agent::Agent, env::Environment, eval::Eval};

pub struct Node<E: Environment> {
    pub evaluation: Eval, // Q(s)
    pub visit_count: u32, // N(s_prev, a)
    pub policy: f32,      // P(s_prev, a)
    pub children: Box<[(E::Action, Self)]>,
}

impl<E: Environment> std::fmt::Debug for Node<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("visit_count", &self.visit_count)
            .field("evaluation", &self.evaluation)
            .field("policy", &self.policy)
            .field("children", &self.children.len())
            .finish()
    }
}

// TODO: Improve this
impl<E: Environment> std::fmt::Display for Node<E>
where
    E::Action: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{self:?}")?;
        for (mov, node) in &*self.children {
            writeln!(f, "{mov}\t{node:?}")?;
        }
        Ok(())
    }
}

impl<E: Environment> Default for Node<E> {
    fn default() -> Self {
        Self {
            visit_count: Default::default(),
            evaluation: Eval::default(),
            policy: Default::default(),
            children: Box::default(),
        }
    }
}

impl<E: Environment> Node<E> {
    #[must_use]
    pub fn from_policy(policy: f32) -> Self {
        Self {
            policy,
            ..Default::default()
        }
    }

    #[inline]
    #[must_use]
    pub const fn needs_initialization(&self) -> bool {
        self.visit_count <= 1
    }

    fn update_mean_value(&mut self, value: f32) {
        let Eval::Value(mean_value) = &mut self.evaluation else {
            unreachable!("Updating the mean value doesn't make sense if the result is known");
        };
        *mean_value =
            mean_value.mul_add((self.visit_count - 1) as f32, value) / self.visit_count as f32;
    }

    fn propagate_child_eval(&mut self, child_eval: Eval) -> Eval {
        self.update_mean_value(child_eval.negate().into());
        let evaluations = self.children.iter().map(|(_, node)| node.evaluation);

        match child_eval {
            // This move made the opponent lose, so this position is a win.
            Eval::Loss(_) => {
                self.evaluation = child_eval.negate();
                self.evaluation
            }

            // If all moves lead to wins for the opponent, this node is a loss.
            Eval::Win(_) if evaluations.clone().all(|e| e.is_win()) => {
                self.evaluation = Eval::Loss(
                    1 + evaluations
                        .filter_map(|e| e.ply())
                        .max()
                        .expect("There should be child evaluations."),
                );
                self.evaluation
            }

            // If all moves lead to wins or draws for the opponent, we choose to draw.
            Eval::Draw(_) | Eval::Win(_)
                if evaluations.clone().all(|e| e.is_win() || e.is_draw()) =>
            {
                self.evaluation = Eval::Draw(
                    1 + evaluations
                        .filter_map(|e| e.is_draw().then(|| e.ply().unwrap()))
                        .max()
                        .expect("There should be at least one draw."),
                );
                self.evaluation
            }

            // Otherwise this position is not know and we just back-propagate the child result.
            _ => Eval::Value(child_eval.negate().into()),
        }
    }

    pub fn initialize<A: Agent<E>>(&mut self, env: &E, actions: &mut Vec<E::Action>, agent: &A) {
        // Check if the position is terminal.
        if let Some(terminal) = env.terminal() {
            self.evaluation = terminal.into();
            return;
        }

        let policy = agent.policy(env);
        env.populate_actions(actions);

        self.children = actions
            .drain(..)
            .map(|action| (action.clone(), Self::from_policy(policy[action])))
            .collect();

        // Get static evaluation from agent.
        self.evaluation = Eval::Value(agent.value(env));
    }

    pub fn simulate<A: Agent<E>>(
        &mut self,
        mut env: E,
        actions: &mut Vec<E::Action>,
        agent: &A,
    ) -> Eval {
        self.visit_count += 1;
        if self.evaluation.is_known() {
            debug_assert!(
                !self.evaluation.is_win(),
                "Simulating a known win is useless because the action leading to this state \
                 should never be taken."
            );
            return self.evaluation;
        }
        if self.needs_initialization() {
            self.initialize(&env, actions, agent);
            return self.evaluation;
        }

        let (action, node) = self.select_with_improved_policy();
        env.step(action.clone());
        let child_eval = node.simulate(env, actions, agent);
        self.propagate_child_eval(child_eval)
    }
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;

    use super::super::{agent::dummy::Dummy, eval::Eval, mcts::Node};

    #[test]
    fn find_tinue_easy() {
        const MAX_VISITS: usize = 5_000;

        // https://ptn.ninja/NoZQlgLgpgBARABwgOwHTLMgVgQzgXQFgAoUMAL1jgGYCTgB5BKDZAc3gGcB3HBO0gBEc0eACYADGOqoJAdlQBGAGwDgAFTABbKpIBcYgBx7qAVjUAlKJwCuAGwjwLAWgkCSi1DBzUYAY0USMS8-MX9qEhkYACNfP2pnIA&name=MwD2Q&ply=5!
        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "c1", "c2", "c3", "b3", "c3-"]);
        let mut root = Node::default();
        let mut actions = Vec::new();

        (0..MAX_VISITS)
            .find(|_| {
                matches!(
                    root.simulate(game.clone(), &mut actions, &Dummy),
                    Eval::Win(_)
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

        // https://ptn.ninja/NoZQlgLgpgBARABwgOwHTLMgVgQzgXQFgAoUMAL1jgGYCTgB5BKDZAc3gGcB3HBO0gBEc0eACYADGOqoJAdlQBGAGwDgAFTABbKpIBcYgBx7qAVjUAlKJwCuAGwjwLAWgkCSi1DBzVvikmJeAEaKMADGikA&name=MwD2Q&ply=3!
        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "c1"]);
        let mut root = Node::default();
        let mut actions = Vec::new();

        (0..MAX_VISITS)
            .find(|_| {
                matches!(
                    root.simulate(game.clone(), &mut actions, &Dummy),
                    Eval::Win(_)
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
            "b2".parse().unwrap(), // Maybe also c2?
        );
    }
}
