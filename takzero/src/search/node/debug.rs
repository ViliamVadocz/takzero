use std::fmt;

use ordered_float::NotNan;

use super::{
    super::{env::Environment, eval::Eval},
    policy::upper_confidence_bound_with_predictor,
    Node,
};

impl<E: Environment> fmt::Display for Node<E>
where
    E::Action: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut action_info = self.action_info();
        action_info.sort_by_key(|a| a.visit_count);
        if self.needs_initialization() {
            writeln!(f, "--- This node still needs to be initialized! ---")?;
        } else {
            for a in action_info {
                writeln!(f, "{a}")?;
            }
            // Header for action info
            writeln!(
                f,
                "[ action ] [ count ] [ logit ] [ proba ] [ impol ] [ puct ] [ stdev ] [ \
                 evaluation ]"
            )?;
        }
        writeln!(
            f,
            "((node))  [count: {}]  [std_dev: {:.4}]  [eval: {:+.4}]",
            self.visit_count, self.std_dev, self.evaluation
        )
    }
}

impl<E: Environment> Node<E> {
    #[must_use]
    pub fn action_info(&self) -> Vec<ActionInfo<E::Action>> {
        self.improved_policy(self.most_visited_count())
            .zip(self.children.iter())
            .map(|(improved_policy, (action, child))| ActionInfo {
                action: action.clone(),
                visit_count: child.visit_count,
                logit: child.logit,
                probability: child.probability,
                improved_policy,
                puct: upper_confidence_bound_with_predictor(
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
            "{: ^10} {: ^9} {: ^9} {: ^9} {: ^9} {: ^8} {: ^9} {: ^14}",
            self.action.to_string(),
            self.visit_count,
            format!("{:+.4}", self.logit.into_inner()),
            format!("{:.4}", self.probability.into_inner()),
            format!("{:.4}", self.improved_policy.into_inner()),
            format!("{:.4}", self.puct),
            format!("{:.4}", self.std_dev),
            format!("{:+.4}", self.eval)
        )
    }
}
