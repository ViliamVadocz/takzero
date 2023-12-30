use std::fmt;

use ordered_float::NotNan;

use super::{
    super::{env::Environment, eval::Eval},
    policy::upper_confidence_bound,
    Node,
};

// TODO: Improve this
impl<E: Environment> fmt::Display for Node<E>
where
    E::Action: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut action_info = self.action_info(0.0);
        action_info.sort_by_key(|a| a.visit_count);
        writeln!(
            f,
            "[root]   c:{: >8} v:{:+.4} e:{:+.4}",
            self.visit_count, self.std_dev, self.evaluation
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
                logit: child.logit,
                probability: child.probability,
                improved_policy,
                puct: upper_confidence_bound(
                    self.visit_count as f32,
                    child.visit_count as f32,
                    child.probability.into_inner(),
                ),
                eval: child.evaluation,
                std_dev: child.std_dev,
            })
            .collect()
    }
}

pub struct ActionInfo<A> {
    action: A,
    visit_count: u32,
    logit: NotNan<f32>,
    probability: NotNan<f32>,
    improved_policy: NotNan<f32>,
    puct: f32,
    std_dev: NotNan<f32>,
    eval: Eval,
}

impl<A: fmt::Display> fmt::Display for ActionInfo<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{: >8} count:{: >8} logit:{:+.4} prob:{:+.4} impol:{:+.4} puct:{:+.4} stdev:{:.4} \
             eval:{:+.4?}",
            self.action.to_string(),
            self.visit_count,
            self.logit.into_inner(),
            self.probability.into_inner(),
            self.improved_policy.into_inner(),
            self.puct,
            self.std_dev,
            self.eval
        )
    }
}
