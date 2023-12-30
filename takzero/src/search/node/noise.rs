use ordered_float::NotNan;
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

use super::Node;
use crate::search::env::Environment;

impl<E: Environment> Node<E> {
    #[allow(clippy::missing_panics_doc)]
    pub fn apply_dirichlet(&mut self, rng: &mut impl Rng, alpha: f32, ratio: f32) {
        assert!(
            !self.needs_initialization(),
            "cannot apply dirichlet noise without initialized policy"
        );
        let dirichlet = Dirichlet::new(&vec![alpha; self.children.len()]).unwrap();
        let samples = dirichlet.sample(rng);

        self.children
            .iter_mut()
            .zip(samples)
            .for_each(|((_, child), noise)| {
                child.probability = child.probability * (1.0 - ratio) + noise * ratio;
                child.logit = NotNan::new(child.probability.ln())
                    .expect("Logit from probability should not be NaN");
            });
    }
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use ordered_float::NotNan;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::search::{
        agent::dummy::Dummy,
        env::Environment,
        node::{policy::softmax, Node},
    };

    fn sum_of_probabilities<E: Environment>(node: &Node<E>) -> NotNan<f32> {
        node.children
            .iter()
            .map(|(_, child)| child.probability)
            .sum::<NotNan<f32>>()
    }

    #[test]
    fn distribution_stays_1_after_noise() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut node = Node::default();
        let env = Game::<3, 0>::default();
        node.simulate_simple(&Dummy, env, 0.0, &mut ());

        println!("{node}");
        // Sum of probabilities is 1 before noise.
        assert!((sum_of_probabilities(&node) - 1.0).abs() < 1.1 * f32::EPSILON);
        node.apply_dirichlet(&mut rng, 0.5, 0.2);

        println!("{node}");
        // Sum of probabilities is 1 after noise.
        assert!((sum_of_probabilities(&node) - 1.0).abs() < 1.1 * f32::EPSILON);
        // Softmax of new logits equals probabilities.
        softmax(node.children.iter().map(|(_, child)| child.logit))
            .zip(node.children.iter().map(|(_, child)| child.probability))
            .for_each(|(a, b)| assert!((a - b).abs() < f32::EPSILON));
    }
}
