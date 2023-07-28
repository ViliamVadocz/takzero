use std::cmp::Reverse;

use rand::Rng;
use rand_distr::{Distribution, Gumbel};

use super::{agent::Agent, env::Environment, eval::Eval, mcts::Node, policy::sigma};

impl<E: Environment> Node<E> {
    fn add_gumbel_to_policy(&mut self, gumbel_distr: Gumbel<f32>, rng: &mut impl Rng) -> Vec<f32> {
        let gumbel: Vec<f32> = gumbel_distr
            .sample_iter(rng)
            .take(self.children.len())
            .collect();
        self.children
            .iter_mut()
            .zip(&gumbel)
            .for_each(|((_, node), g)| node.policy += g);
        gumbel
    }

    fn remove_gumbel_from_policy(&mut self, gumbel: Vec<f32>) {
        self.children
            .iter_mut()
            .zip(gumbel)
            .for_each(|((_, node), g)| node.policy -= g);
    }

    fn sequential_halving<A: Agent<E>>(
        &mut self,
        env: &E,
        actions: &mut Vec<E::Action>,
        agent: &A,
        simulations: u32,
    ) -> (u32, Option<E::Action>) {
        // The search set keeps track of actions that are being considered.
        // The discard set keeps the others so that we can resample if we hit a
        // known result.
        let mut discard_set = Vec::with_capacity(self.children.len());
        let mut search_set: Vec<_> = self.children.iter_mut().collect();
        discard_set.extend(search_set.extract_if(|(_, node)| node.evaluation.is_known()));

        let mut target_number = search_set.len();
        let number_of_halving_steps = target_number.ilog2();
        let mut used_simulations = 0;
        let mut pretend_most_visits = 0;

        for step in 0..=number_of_halving_steps {
            // If too many actions were thrown away, resample from discard set.
            if search_set.len() < target_number {
                search_set.append(&mut discard_set);
            }
            // If the worst opponent evaluation is a forced win or loss we break the search.
            // In the case of a draw we still want to do a bit more search just in case.
            if let Some(eval) = search_set.iter().map(|(_, node)| node.evaluation).min() {
                if eval.is_win() || eval.is_loss() {
                    self.evaluation = eval.negate();
                    break;
                }
            }
            // Keep only the `target_number` most promising actions.
            search_set.sort_unstable_by_key(|(_, node)| {
                Reverse(
                    node.evaluation
                        .negate()
                        .map(|q| node.policy + sigma(q, pretend_most_visits as f32)),
                )
            });
            discard_set.extend(search_set.drain(target_number..));
            if target_number == 1 {
                break;
            }

            // Run simulations.
            let simulations_per_action = (simulations - used_simulations)
                / (number_of_halving_steps - step)
                / target_number as u32;
            // If a result is known we discard it early.
            discard_set.extend(search_set.extract_if(|(action, node)| {
                if let Some(simulations_before_known) = (0..simulations_per_action).find(|_| {
                    let mut clone = env.clone();
                    clone.step(action.clone());
                    node.simulate(clone, actions, agent).is_known()
                }) {
                    used_simulations += simulations_before_known;
                    true
                } else {
                    used_simulations += simulations_per_action;
                    false
                }
            }));
            pretend_most_visits += simulations_per_action;
            target_number /= 2;
        }

        (
            used_simulations,
            (search_set.len() == 1).then(|| search_set.into_iter().next().unwrap().0.clone()),
        )
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn sequential_halving_with_gumbel<A: Agent<E>>(
        &mut self,
        env: &E,
        actions: &mut Vec<E::Action>,
        rng: &mut impl Rng,
        agent: &A,
        simulations: u32,
    ) -> E::Action {
        if self.children.is_empty() {
            self.initialize(env, actions, agent);
        }
        if self.evaluation.is_known() {
            return self
                .children
                .iter()
                .min_by_key(|(_, node)| node.evaluation)
                .expect("The environment should have actions.")
                .0
                .clone();
        }

        let gumbel = self.add_gumbel_to_policy(Gumbel::new(0.0, 1.0).unwrap(), rng);
        let (used_simulations, action) = self.sequential_halving(env, actions, agent, simulations);
        self.remove_gumbel_from_policy(gumbel);
        self.visit_count += used_simulations;

        if self.evaluation.is_known() {
            // Root evaluation is known so we just pick the action
            // which minimizes the opponent's evaluation.
            self.children
                .iter()
                .min_by_key(|(_, node)| node.evaluation)
                .unwrap()
                .0
                .clone()
        } else {
            // Update evaluation.
            // FIXME: Using a janky formula made-up because it doesn't really matter.
            self.evaluation = Eval::Value(
                self.children
                    .iter()
                    .map(|(_, node)| {
                        let eval: f32 = node.evaluation.negate().into();
                        let ratio = node.visit_count as f32 / (self.visit_count - 1) as f32;
                        eval * ratio
                    })
                    .sum(),
            );

            action.expect("Sequential halving should arrive at a single action.")
        }
    }
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use rand::SeedableRng;

    use crate::search::{agent::dummy::Dummy, eval::Eval, mcts::Node};

    #[test]
    fn find_win_with_gumbel() {
        const SIMULATIONS: u32 = 100;
        const SEED: u64 = 42;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "b3"]);
        let mut actions = Vec::new();

        let mut root = Node::default();
        let top_action =
            root.sequential_halving_with_gumbel(&game, &mut actions, &mut rng, &Dummy, SIMULATIONS);

        println!("{root}");
        // assert_eq!(root.visit_count, SIMULATIONS);
        assert_eq!(top_action, "c1".parse().unwrap());
    }

    #[test]
    fn realize_loss() {
        const SIMULATIONS: u32 = 1024;
        const SEED: u64 = 123;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let game: Game<3, 0> = Game::from_ptn_moves(&["a3", "a1", "b1", "a3>", "c2"]);
        let mut actions = Vec::new();

        let mut root = Node::default();
        root.sequential_halving_with_gumbel(&game, &mut actions, &mut rng, &Dummy, SIMULATIONS);

        assert_eq!(root.evaluation, Eval::Loss(2));
    }
}
