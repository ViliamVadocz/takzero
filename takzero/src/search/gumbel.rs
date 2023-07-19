use float_ord::FloatOrd;
use rand::Rng;
use rand_distr::{Distribution, Gumbel};

use super::{agent::Agent, env::Environment, mcts::Node};

impl<E: Environment> Node<E> {
    fn visit_count_of_most_visited_child(&self) -> f32 {
        #![allow(clippy::cast_precision_loss)]
        self.children
            .iter()
            .map(|(_, node)| node.visit_count)
            .max()
            .unwrap_or_default() as f32
    }

    // TODO: Cleanup
    #[allow(clippy::missing_panics_doc)]
    pub fn sequential_halving_with_gumbel<A: Agent<E>>(
        &mut self,
        env: &E,
        actions: &mut Vec<E::Action>,
        rng: &mut impl Rng,
        agent: &A,
        sampled_actions: u32,
        simulations: u32,
    ) -> E::Action {
        #![allow(clippy::cast_possible_truncation)]
        let gumbel: Gumbel<f32> = Gumbel::new(0.0, 1.0).unwrap();

        if self.children.is_empty() {
            let policy = agent.policy(env);
            env.populate_actions(actions);
            // Sample actions with highest `logits + gumbel`.
            let mut noisy_policies: Vec<_> = gumbel
                .sample_iter(rng)
                .zip(actions.drain(..))
                .map(|(g, action)| (action.clone(), g + policy[action]))
                .collect();
            noisy_policies.sort_unstable_by_key(|&(_, noisy_policy)| FloatOrd(noisy_policy));
            self.children = noisy_policies
                .into_iter()
                .rev()
                .take(sampled_actions as usize)
                .map(|(action, noisy_policy)| (action, Self::from_policy(noisy_policy)))
                .collect();
        } else {
            // Add Gumbel noise to policies.
            self.children
                .iter_mut()
                .zip(gumbel.sample_iter(rng))
                .for_each(|((_, node), g)| node.policy += g);
        }

        // Sequential halving.
        let mut m = (self.children.len() as u32).min(sampled_actions);
        let number_of_halving_steps = m.ilog2();
        let mut completed_simulations = 0;
        for _ in 0..number_of_halving_steps {
            let most_visits = self.visit_count_of_most_visited_child();
            // Get the actions which maximize `logits + gumbel + sigma(value)`.
            let mut top: Vec<_> = self.children.iter_mut().collect();
            top.sort_unstable_by_key(|(_, node)| {
                sequential_halving_priority(node.policy, node.evaluation.into(), most_visits)
            });
            let mut top: Vec<_> = top.into_iter().rev().take(m as usize).collect();

            #[allow(clippy::cast_possible_truncation)]
            for _ in 0..simulations / (number_of_halving_steps * m) {
                for (action, node) in &mut top {
                    if node.is_known() {
                        // FIXME
                        continue;
                    }
                    let mut clone = env.clone();
                    clone.step(action.clone());
                    node.simulate(clone, actions, agent);
                }
            }
            completed_simulations += simulations / (number_of_halving_steps * m) * top.len() as u32;
            m /= 2;

            // Do any remaining simulations.
            if m == 1 && completed_simulations < simulations {
                for i in 0..(simulations - completed_simulations) {
                    let index = i as usize % top.len();
                    let (action, node) = &mut top[index];
                    if node.is_known() {
                        // FIXME
                        continue;
                    }
                    let mut clone = env.clone();
                    clone.step(action.clone());
                    node.simulate(clone, actions, agent);
                }
            }
        }
        self.visit_count += simulations;

        debug_assert_eq!(m, 1);
        let most_visits = self.visit_count_of_most_visited_child();
        self.children
            .iter()
            .max_by_key(|(_, node)| {
                // FIXME: What about when results are known?
                sequential_halving_priority(node.policy, node.evaluation.into(), most_visits)
            })
            .map(|(action, _)| action)
            .unwrap()
            .clone()
    }
}

fn sequential_halving_priority(
    policy_plus_gumbel: f32,
    value: f32,
    visit_count_of_most_visited: f32,
) -> FloatOrd<f32> {
    FloatOrd(policy_plus_gumbel + sigma(value, visit_count_of_most_visited))
}

fn sigma(q: f32, visit_count_of_most_visited: f32) -> f32 {
    const C_VISIT: f32 = 50.0;
    const C_SCALE: f32 = 1.0;
    (C_VISIT + visit_count_of_most_visited) * C_SCALE * q
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use rand::SeedableRng;

    use crate::search::{agent::dummy::Dummy, mcts::Node};

    #[test]
    fn idk() {
        const SAMPLED_ACTIONS: u32 = 100;
        const SIMULATIONS: u32 = 100;
        const SEED: u64 = 42;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "b3"]);
        let mut actions = Vec::new();

        let mut root = Node::default();
        let top_action = root.sequential_halving_with_gumbel(
            &game,
            &mut actions,
            &mut rng,
            &Dummy,
            SAMPLED_ACTIONS,
            SIMULATIONS,
        );

        println!("{root}");
        assert_eq!(root.visit_count, SIMULATIONS);
        assert_eq!(top_action, "c1".parse().unwrap());
    }
}
