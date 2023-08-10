use std::fmt;

use super::{super::env::Environment, Node};

impl<E: Environment> fmt::Debug for Node<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("visit_count", &self.visit_count)
            .field("evaluation", &self.evaluation)
            .field("policy", &self.policy)
            .field("children", &self.children.len())
            .finish()
    }
}

// TODO: Improve this
impl<E: Environment> fmt::Display for Node<E>
where
    E::Action: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self:?}")?;
        for (mov, node) in &*self.children {
            writeln!(f, "{mov}\t{node:?}")?;
        }
        Ok(())
    }
}
