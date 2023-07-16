use float_ord::FloatOrd;

use crate::{agent::Agent, env::Environment, eval::Eval};

pub struct Node<E: Environment> {
    pub visit_count: u32, // N(s, a)
    pub evaluation: Eval, // Q(s, a)
    pub policy: f32,      // P(s, a)
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
        #[allow(clippy::cast_precision_loss)]
        if let Eval::Value(mean_value) = &mut self.evaluation {
            *mean_value =
                mean_value.mul_add((self.visit_count - 1) as f32, value) / self.visit_count as f32;
        }
    }

    fn propagate_child_eval(&mut self, child_eval: Eval) -> Eval {
        let parent_view = child_eval.parent_view();
        self.update_mean_value(parent_view.into());
        let evaluations = self.children.iter().map(|(_, node)| node.evaluation);

        // Terminal node solver.
        match child_eval {
            // Opponent has a winning move, so this is position is a loss.
            Eval::Win(_) => {
                self.evaluation = parent_view;
                return parent_view;
            }

            // If all moves are losing for the opponent, this is a winning position.
            Eval::Loss(ply)
                if evaluations
                    .clone()
                    .all(|eval| matches!(eval, Eval::Loss(_))) =>
            {
                let win = Eval::Win(
                    1 + evaluations
                        .map(|eval| match eval {
                            Eval::Loss(ply) => ply,
                            _ => unreachable!(),
                        })
                        .max()
                        .unwrap_or(ply),
                );
                self.evaluation = win;
                return win;
            }

            // If all options are either a draw or a loss, this node is a draw.
            Eval::Draw(ply) | Eval::Loss(ply)
                if evaluations
                    .clone()
                    .all(|eval| matches!(eval, Eval::Draw(_) | Eval::Loss(_))) =>
            {
                let draw = Eval::Draw(
                    1 + evaluations
                        .filter_map(|eval| match eval {
                            Eval::Draw(ply) => Some(ply),
                            _ => None,
                        })
                        .min()
                        .unwrap_or(ply),
                );
                self.evaluation = draw;
                return draw;
            }

            _ => {}
        };

        parent_view
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
                return self.propagate_child_eval(terminal.into());
            }

            let policy = agent.policy(&env);
            env.populate_actions(actions);

            self.children = actions
                .drain(..)
                .map(|action| (action.clone(), Self::from_policy(policy[action])))
                .collect();
            return Eval::Value(agent.value(&env));
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
