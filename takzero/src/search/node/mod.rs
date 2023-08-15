pub mod debug;
pub mod gumbel;
pub mod mcts;
pub mod policy;

use super::{env::Environment, eval::Eval};

pub struct Node<E: Environment> {
    pub evaluation: Eval, // V(s_t) or Q(s_prev, a)
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
    #[must_use]
    pub const fn needs_initialization(&self) -> bool {
        self.visit_count <= 1
    }

    /// Get the sub-tree for a given action.
    /// This allows for tree reuse.
    ///
    /// # Panics
    ///
    /// Panics if the action has not been visited from the parent.
    #[must_use]
    pub fn play(mut self, action: &E::Action) -> Self {
        let Some((_, child)) = self.children.iter_mut().find(|(a, _)| action == a) else {
            return Self::default();
        };
        std::mem::take(child)
        // TODO: Maybe deallocate children on another thread.
    }
}
