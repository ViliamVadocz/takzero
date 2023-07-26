use std::cmp::Reverse;

use float_ord::FloatOrd;
use rand::Rng;
use rand_distr::{Distribution, Gumbel};

use super::{agent::Agent, env::Environment, mcts::Node, policy::sigma};

impl<E: Environment> Node<E> {
    // FIXME: Evaluation will no longer be accurate after this.
    #[allow(clippy::missing_panics_doc)]
    pub fn sequential_halving_with_gumbel<A: Agent<E>>(
        &mut self,
        env: &E,
        actions: &mut Vec<E::Action>,
        rng: &mut impl Rng,
        agent: &A,
        sampled_actions: usize,
        simulations: u32,
    ) -> E::Action {
        if self.children.is_empty() {
            self.initialize(env, actions, agent);
        }

        // Add Gumbel noise to policies.
        let gumbel: Gumbel<f32> = Gumbel::new(0.0, 1.0).unwrap();
        self.children
            .iter_mut()
            .zip(gumbel.sample_iter(rng))
            .for_each(|((_, node), g)| node.policy += g);

        // Sequential halving.
        let mut search_set: Vec<_> = self
            .children
            .iter_mut()
            .filter(|(_, node)| !node.evaluation.is_win())
            .map(|(a, b)| (a, b))
            .collect();
        let mut discard_set = Vec::with_capacity(search_set.len());

        let mut used_simulations = 0;
        let mut m = search_set.len().min(sampled_actions);
        let number_of_halving_steps = m.ilog2();
        for step in (1..=number_of_halving_steps).rev() {
            if search_set.len() < m {
                search_set.append(&mut discard_set);
            }
            // If there is a forced win we can just quit early.
            if search_set.iter().any(|(_, node)| node.evaluation.is_loss()) {
                break;
            }
            search_set.sort_unstable_by_key(|(_, node)| {
                Reverse(FloatOrd(
                    node.policy
                        + sigma(
                            node.evaluation.negate().into(),
                            // Pretend there are no previous visits (tree re-use)
                            2.0f32.powi((number_of_halving_steps - step) as i32),
                        ),
                ))
            });
            search_set.truncate(m);

            let simulations_per_action = (simulations - used_simulations) / step / m as u32;
            discard_set.extend(search_set.extract_if(|(action, node)| {
                for _ in 0..simulations_per_action {
                    used_simulations += 1;
                    let mut clone = env.clone();
                    clone.step(action.clone());
                    if node.simulate(clone, actions, agent).is_known() {
                        return false;
                    }
                }
                true
            }));

            m /= 2;
        }
        self.visit_count += used_simulations;

        let most_visited_count = self.most_visited_count();
        self.children
            .iter()
            .max_by_key(|(_, node)| {
                node.evaluation
                    .negate()
                    // FIXME: Using node visit count instead of max visit count at root
                    .map(|q| node.policy + sigma(q, most_visited_count))
            })
            .map(|(action, _)| action)
            .unwrap()
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use rand::SeedableRng;

    use crate::search::{agent::dummy::Dummy, mcts::Node};

    #[test]
    fn find_win_with_gumbel() {
        const SAMPLED_ACTIONS: usize = 100;
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
        // assert_eq!(root.visit_count, SIMULATIONS);
        assert_eq!(top_action, "c1".parse().unwrap());
    }
}
