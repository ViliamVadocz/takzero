pub mod batched;
pub mod debug;
// pub mod gumbel;
pub mod mcts;
pub mod noise;
pub mod policy;

use ordered_float::NotNan;
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};

use super::{env::Environment, eval::Eval};

pub struct Node<E: Environment> {
    pub evaluation: Eval,         // V(s_t) or Q(s_prev, a)
    pub visit_count: u32,         // N(s_prev, a)
    pub logit: NotNan<f32>,       // log(P(s_prev, a)) (network output)
    pub probability: NotNan<f32>, // P(s_prev, a) (normalized)
    pub std_dev: NotNan<f32>,     // average sqrt(clip(max(UBE(s_t), geo_sum_discount * RND(s_t))))
    pub children: Box<[(E::Action, Self)]>,
}

impl<E: Environment> Default for Node<E> {
    fn default() -> Self {
        Self {
            visit_count: Default::default(),
            evaluation: Eval::default(),
            logit: NotNan::default(),
            probability: NotNan::default(),
            std_dev: NotNan::default(),
            children: Box::default(),
        }
    }
}

impl<E: Environment> Node<E> {
    #[must_use]
    pub fn from_logit_and_probability(logit: NotNan<f32>, probability: NotNan<f32>) -> Self {
        Self {
            logit,
            probability,
            ..Default::default()
        }
    }

    #[inline]
    #[must_use]
    pub const fn needs_initialization(&self) -> bool {
        self.children.is_empty()
    }

    /// Descend in the tree, replacing the root the sub-tree for a given action.
    /// This allows for tree reuse.
    /// If the action was not visited, the node will `Node::default()`.
    pub fn descend(&mut self, action: &E::Action) {
        let mut me = std::mem::take(self);
        let Some((_, child)) = me.children.iter_mut().find(|(a, _)| action == a) else {
            return;
        };
        std::mem::swap(self, child);
        // TODO: Maybe deallocate children on another thread.
    }

    #[inline]
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.evaluation.ply().is_some_and(|ply| ply == 0)
    }

    /// Return the best action after search.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    #[must_use]
    pub fn select_best_action(&self) -> E::Action {
        if self.evaluation.is_known() {
            // The node is solved, pick the best action.
            self.children
                .iter()
                .min_by_key(|(_, child)| child.evaluation)
        } else {
            // Select the action with the most visits.
            self.children
                .iter()
                .max_by_key(|(_, child)| child.visit_count)
        }
        .expect("there should be at least one child")
        .0
        .clone()
    }

    /// Return an action to use in selfplay.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    pub fn select_selfplay_action(
        &self,
        proportional_sample: bool,
        rng: &mut impl Rng,
    ) -> E::Action {
        if self.evaluation.is_known() {
            // The node is solved, pick the best action.
            self.children
                .iter()
                .min_by_key(|(_, child)| child.evaluation)
                .expect("there should be at least one child")
                .0
                .clone()
        } else if proportional_sample {
            // Select an action randomly, proportional to visits.
            let weighted_index =
                WeightedIndex::new(self.children.iter().map(|(_, child)| child.visit_count))
                    .expect("there should be at least one child and visits cannot be negative");
            self.children[weighted_index.sample(rng)].0.clone()
        } else {
            // Select the action with the most visits.
            self.children
                .iter()
                .max_by_key(|(_, child)| child.visit_count)
                .expect("there should be at least one child")
                .0
                .clone()
        }
    }

    /// Get the UBE target from the root after search.
    #[must_use]
    pub fn ube_target(&self, beta: f32) -> NotNan<f32> {
        // Sort children according to `-V(s) + \beta * \sigma`
        let mut children: Vec<_> = self.children.iter().map(|(_, child)| child).collect();
        children.sort_unstable_by_key(|child| {
            NotNan::from(child.evaluation.negate()) + child.std_dev * beta
        });

        // Take the top `highest` children, skipping the top few to get
        // `lower_of_the_highest` children.
        let len = children.len();
        if len < 4 {
            return self.std_dev;
        }
        let highest = len / 2;
        let lower_of_the_highest = len / 4;

        // Take average of standard deviations.
        children
            .into_iter()
            .skip(len - highest)
            .take(lower_of_the_highest)
            .map(|child| child.std_dev)
            .sum::<NotNan<f32>>()
            / lower_of_the_highest as f32
    }
}
