use std::fmt;

use ordered_float::OrderedFloat;

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
        let mut children: Vec<_> = self.improved_policy().zip(self.children.iter()).collect();
        children.sort_by_key(|(p, _)| OrderedFloat(*p));
        for (improved_policy, (mov, node)) in children {
            writeln!(f, "{mov}\t{improved_policy:.6}\t{node:?}")?;
        }
        Ok(())
    }
}
