use float_ord::FloatOrd;

use super::{agent::Agent, env::Environment, eval::Eval};

pub struct Node<E: Environment> {
    pub evaluation: Eval, // Q(s)
    pub visit_count: u32, // N(s_prev, a)
    pub policy: f32,      // P(s_prev, a)
    pub children: Box<[(E::Action, Self)]>,
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
    const fn needs_initialization(&self) -> bool {
        self.visit_count == 0
    }

    const fn is_known(&self) -> bool {
        match self.evaluation {
            Eval::Value(_) => false,
            Eval::Win(_) | Eval::Draw(_) | Eval::Loss(_) => true,
        }
    }

    fn update_mean_value(&mut self, value: f32) {
        #![allow(clippy::cast_precision_loss)]
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

    pub fn simulate<A: Agent<E>>(
        &mut self,
        mut env: E,
        actions: &mut Vec<E::Action>,
        agent: &A,
    ) -> Eval {
        debug_assert!(!self.is_known(), "Simulating a known result is useless.");

        if self.needs_initialization() {
            self.visit_count += 1;
            // Check if the position is terminal.
            if let Some(terminal) = env.terminal() {
                self.evaluation = terminal.into();
                return self.evaluation;
            }

            let policy = agent.policy(&env);
            env.populate_actions(actions);

            self.children = actions
                .drain(..)
                .map(|action| (action.clone(), Self::from_policy(policy[action])))
                .collect();

            // Get static evaluation from agent.
            self.evaluation = Eval::Value(agent.value(&env));
            return self.evaluation;
        }
        self.visit_count += 1;

        // Select action proportionally to policy.
        let Some((action, node)) = self
            .children
            .iter_mut()
            .filter(|(_, node)| !node.is_known())
            .max_by_key(|(_, node)| {
                #[allow(clippy::cast_precision_loss)]
                FloatOrd(node.policy - node.visit_count as f32 / ((self.visit_count + 1) as f32))
            })
        else {
            unreachable!("If this node is not known there should be some unknown nodes")
        };

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
        const MAX_VISITS: usize = 3_000;

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "c1", "c2", "c3", "b3", "c3-"]);
        let mut root = Node::default();
        let mut actions = Vec::new();

        #[allow(clippy::maybe_infinite_iter)]
        (0..MAX_VISITS)
            .find(|_| {
                matches!(
                    root.simulate(game.clone(), &mut actions, &Dummy),
                    Eval::Win(_)
                )
            })
            .expect("This position is solvable with MAX_VISITS.");

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
    fn find_tinue_harder() {
        const MAX_VISITS: usize = 300_000;

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "c1"]);
        let mut root = Node::default();
        let mut actions = Vec::new();

        #[allow(clippy::maybe_infinite_iter)]
        (0..MAX_VISITS)
            .find(|_| {
                matches!(
                    root.simulate(game.clone(), &mut actions, &Dummy),
                    Eval::Win(_)
                )
            })
            .expect("This position is solvable with MAX_VISITS.");

        assert_eq!(
            root.children
                .iter()
                .find(|(_, node)| node.evaluation.is_loss())
                .unwrap()
                .0,
            "c2".parse().unwrap(),
        );
    }
}

