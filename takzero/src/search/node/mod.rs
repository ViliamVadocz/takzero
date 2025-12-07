use ordered_float::NotNan;
use rand::{seq::IndexedRandom, Rng};

use super::{env::Environment, eval::Eval};

pub mod batched;
pub mod debug;
// pub mod gumbel;
pub mod mcts;
pub mod noise;
pub mod policy;

#[rustfmt::skip]
pub struct Node<E: Environment> {
    pub evaluation: Eval,         // V(s_t) or Q(s_prev, a)
    pub visit_count: u32,         // N(s_prev, a), incremented in forward pass
    #[cfg(feature = "virtual")]
    pub virtual_visits: u32,      // count number of unevaluated trajectories through this node
    pub logit: NotNan<f32>,       // log(P(s_prev, a)) (network output)
    pub probability: NotNan<f32>, // P(s_prev, a) (normalized)
    pub std_dev: NotNan<f32>,     // average sqrt(clamp(max(UBE(s_t), geo_sum_discount * RND(s_t))))
    pub children: Box<[(E::Action, Self)]>,
}

impl<E: Environment> Default for Node<E> {
    fn default() -> Self {
        Self {
            evaluation: Eval::default(),
            visit_count: Default::default(),
            #[cfg(feature = "virtual")]
            virtual_visits: Default::default(),
            logit: NotNan::default(),
            probability: NotNan::default(),
            std_dev: NotNan::default(),
            children: Box::default(),
        }
    }
}

struct PrincipalVariation<'a, E: Environment> {
    node: &'a Node<E>,
}

impl<E: Environment> Iterator for PrincipalVariation<'_, E> {
    type Item = E::Action;

    fn next(&mut self) -> Option<Self::Item> {
        if self.node.needs_initialization() || self.node.is_terminal() {
            return None;
        }
        let best_action = self.node.select_best_action();
        let (_, best_child) = self
            .node
            .children
            .iter()
            .find(|(action, _)| *action == best_action)
            .expect("Best action not found among node's children");

        self.node = best_child;
        Some(best_action.clone())
    }
}

impl<E: Environment> Node<E> {
    #[must_use]
    pub fn from_logit_and_probability_and_parent_value_and_std_dev(
        logit: NotNan<f32>,
        probability: NotNan<f32>,
        evaluation: NotNan<f32>,
        std_dev: NotNan<f32>,
    ) -> Self {
        Self {
            evaluation: Eval::new_not_nan_value(-evaluation),
            logit,
            probability,
            std_dev,
            ..Default::default()
        }
    }

    #[inline]
    #[must_use]
    pub const fn needs_initialization(&self) -> bool {
        self.children.is_empty() && !self.evaluation.is_known()
    }

    /// Returns an iterator over the Principal Variation of the search tree
    pub fn principal_variation(&self) -> impl Iterator<Item = E::Action> + '_ {
        PrincipalVariation { node: self }
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

    /// Returns the negated value of this node.
    /// When using virtual visits, they are counted as losses.
    #[inline]
    #[must_use]
    pub fn q_value(&self) -> NotNan<f32> {
        // #[cfg(feature = "virtual")]
        // {
        // https://discord.com/channels/176389490762448897/361023655465058307/1322698913328791724
        // const COEFFICIENT: f32 = 1.0;
        // let loss_frac = NotNan::new(self.virtual_visits as f32 / self.visit_count as
        // f32).unwrap_or_default() * COEFFICIENT; self.evaluation.map(|v| v *
        // (-loss_frac + 1.0) - loss_frac).negate().into() }
        // #[cfg(not(feature = "virtual"))]
        self.evaluation.negate().into()
    }

    /// Return the best action after search.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    #[must_use]
    pub fn select_best_action(&self) -> E::Action {
        // If the node is solved, pick the optimal action.
        if self.evaluation.is_known() {
            return self
                .children
                .iter()
                .min_by_key(|(_, child)| child.evaluation)
                .expect("There should be at least one child")
                .0
                .clone();
        }

        let most_visited = self
            .children
            .iter()
            .max_by_key(|(_, child)| child.visit_count)
            .expect("There should be at least one child");

        // Pick based on policy when the children have not been visited yet.
        if most_visited.1.visit_count == 0 {
            return self
                .children
                .iter()
                .max_by_key(|(_, child)| child.probability)
                .expect("there should be at least one child")
                .0
                .clone();
        }

        // Otherwise just return the most visited action.
        most_visited.0.clone()
    }

    /// Return an action to use in selfplay.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    pub fn select_selfplay_action(
        &self,
        proportional_sample_with_threshold: Option<u32>,
        rng: &mut impl Rng,
    ) -> E::Action {
        if self.evaluation.is_known() {
            return self.select_best_action();
        }
        let Some(threshold) = proportional_sample_with_threshold else {
            return self.select_best_action();
        };

        // Select an action randomly proportional to visits above the threshold.
        match self.children.choose_weighted(rng, |(_, child)| {
            if child.visit_count >= threshold {
                child.visit_count
            } else {
                0
            }
        }) {
            Ok((a, _)) => a.clone(),
            // No actions have been visited.
            Err(rand::seq::WeightError::InsufficientNonZero) => self.select_best_action(),
            Err(err) => {
                panic!("There should be at least one child, and visits small enough: {err}")
            }
        }
    }

    /// Get the UBE target from the root after search.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    #[must_use]
    pub fn ube_target(&self, beta: f32) -> NotNan<f32> {
        // UBE target = 0.0 when node is solved.
        if self.evaluation.is_known() || self.needs_initialization() {
            NotNan::default()
        } else {
            // Child with maximum value + beta * std_dev.
            let std_dev = self
                .children
                .iter()
                .map(|(_, child)| child)
                .max_by_key(|child| NotNan::from(child.evaluation.negate()) + child.std_dev * beta)
                .expect("There should be at least one child")
                .std_dev;
            std_dev * std_dev
        }
    }
}
