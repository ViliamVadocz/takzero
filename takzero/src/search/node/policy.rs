use ordered_float::NotNan;

use super::{super::env::Environment, Node};

/// Perform the softmax on an iterator.
///
/// # Panics
///
/// Panics if any exponent results in NaN.
pub fn softmax(
    logits: impl Iterator<Item = NotNan<f32>> + Clone,
) -> impl Iterator<Item = NotNan<f32>> + Clone {
    let max = logits.clone().max().unwrap_or_default();
    let exp = logits.map(move |x| {
        NotNan::new((x - max).exp()).expect("exponent should not create NaN in softmax")
    });
    let sum: NotNan<f32> = exp.clone().sum();
    exp.map(move |x| x / sum)
}

impl<E: Environment> Node<E> {
    #[must_use]
    pub fn most_visited_count(&self) -> f32 {
        self.children
            .iter()
            .map(|(_, node)| node.visit_count)
            .max()
            .unwrap_or_default() as f32
    }

    /// Get the improved policy for this node.
    ///
    /// # Panics
    ///
    /// Panics if the evaluation is NaN.
    pub fn improved_policy(
        &self,
        #[cfg(not(feature = "baseline"))] beta: f32,
    ) -> impl Iterator<Item = NotNan<f32>> + '_ {
        let most_visited_count = self.most_visited_count();
        let p = self.children.iter().map(move |(_, node)| -> NotNan<f32> {
            let completed_value: NotNan<f32> = NotNan::new(
                if node.needs_initialization() {
                    self.evaluation
                } else {
                    node.evaluation.negate()
                }
                .into(),
            )
            .expect("completed value should not be NaN");
            sigma(
                completed_value,
                #[cfg(not(feature = "baseline"))]
                node.variance,
                #[cfg(not(feature = "baseline"))]
                beta,
                most_visited_count,
            ) + node.logit
        });

        softmax(p)
    }

    /// Get index of child which maximizes the improved policy.
    #[allow(clippy::missing_panics_doc)]
    pub fn select_with_improved_policy(&mut self, beta: f32) -> usize {
        self.improved_policy(
            #[cfg(not(feature = "baseline"))]
            beta,
        )
        .zip(self.children.iter())
        .enumerate()
        // Prune only losing moves to preserve optimality.
        .filter(|(_, (_, (_, node)))| !node.evaluation.is_win())
        // Minimize mean-squared-error between visits and improved policy
        .max_by_key(|(_, (pi, (_, node)))| {
            pi - node.visit_count as f32 / ((self.visit_count + 1) as f32)
        })
        .map(|(i, _)| i)
        .expect("there should always be a child to simulate")
    }
}

pub const C_VISIT: f32 = 50.0; // Paper used 50, but 30 solves tests
pub const C_SCALE: f32 = 0.1; // Paper used 1, but 0.1 solves tests

#[must_use]
#[allow(clippy::suboptimal_flops)]
pub fn sigma(
    q: NotNan<f32>,
    #[cfg(not(feature = "baseline"))] variance: NotNan<f32>,
    #[cfg(not(feature = "baseline"))] beta: f32,
    visit_count: f32,
) -> NotNan<f32> {
    #[cfg(feature = "baseline")]
    return q * (C_VISIT + visit_count) * C_SCALE;
    #[cfg(not(feature = "baseline"))]
    return (q + variance.sqrt() * beta) * (C_VISIT + visit_count) * C_SCALE;
}
