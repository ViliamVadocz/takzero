pub mod batched;
pub mod debug;
pub mod gumbel;
pub mod mcts;
pub mod noise;
pub mod policy;

use ordered_float::NotNan;

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
}
