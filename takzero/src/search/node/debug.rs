use std::fmt;

use ordered_float::OrderedFloat;

use super::{
    super::{env::Environment, eval::Eval},
    Node,
};

// TODO: Improve this
impl<E: Environment> fmt::Display for Node<E>
where
    E::Action: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut action_info = self.action_info(0.0);
        action_info.sort_by_key(|a| OrderedFloat(a.improved_policy));
        writeln!(
            f,
            "root: c:{: >8} v:{:+.4} e:{:+.4}",
            self.visit_count, self.variance, self.evaluation
        )?;
        for a in action_info {
            writeln!(f, "{a}")?;
        }
        Ok(())
    }
}

impl<E: Environment> Node<E> {
    #[must_use]
    pub fn action_info(&self, beta: f32) -> Vec<ActionInfo<E::Action>> {
        self.improved_policy(beta)
            .zip(self.children.iter())
            .map(|(improved_policy, (action, child))| ActionInfo {
                action: action.clone(),
                visit_count: child.visit_count,
                policy: child.policy,
                improved_policy,
                eval: child.evaluation,
                variance: child.variance,
            })
            .collect()
    }
}

pub struct ActionInfo<A> {
    action: A,
    visit_count: u32,
    policy: f32,
    improved_policy: f32,
    variance: f32,
    eval: Eval,
}

impl<A: fmt::Display> fmt::Display for ActionInfo<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{: >8} c:{: >8} p:{:+.4} i:{:+.4} v:{:.4} e:{:+.4?}",
            self.action,
            self.visit_count,
            self.policy,
            self.improved_policy,
            self.variance,
            self.eval
        )
    }
}
