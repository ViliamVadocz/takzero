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
    pub fn improved_policy(&self, visitations: f32) -> impl Iterator<Item = NotNan<f32>> + '_ {
        let p = self.children.iter().map(move |(_, node)| -> NotNan<f32> {
            let completed_value = if node.needs_initialization() {
                self.evaluation
            } else {
                node.evaluation.negate()
            }
            .into();
            sigma_improve(completed_value, node.std_dev, 0.0, visitations) + node.logit
        });

        softmax(p)
    }

    /// Get index of child which maximizes the improved policy.
    /// Losing actions are pruned unless this node is a proven loss.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    #[must_use]
    pub fn select_with_improved_policy(&self) -> usize {
        self.improved_policy(self.most_visited_count())
            .zip(self.children.iter())
            .enumerate()
            // Prune only losing moves to preserve optimality.
            .filter(|(_, (_, (_, child)))| self.evaluation.is_loss() || !child.evaluation.is_win())
            // Minimize mean-squared-error between visits and improved policy
            .max_by_key(|(_, (pi, (_, node)))| {
                pi - node.visit_count as f32 / ((self.visit_count + 1) as f32)
            })
            .map(|(i, _)| i)
            .expect("there should always be a child to simulate")
    }

    /// Get index of child which maximizes PUCT.
    /// Losing actions are pruned unless this node is a proven loss.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    #[must_use]
    pub fn select_with_puct(&self, beta: f32) -> usize {
        let parent_visit_count = self.visit_count as f32;
        self.children
            .iter()
            .enumerate()
            .filter(|(_, (_, child))| self.evaluation.is_loss() || !child.evaluation.is_win())
            .max_by_key(|(_, (_, child))| {
                let q = child.q_value();
                let puct = upper_confidence_bound_with_predictor(
                    parent_visit_count,
                    child.visit_count as f32,
                    child.probability.into_inner(),
                );
                q + puct + child.std_dev * beta
            })
            .map(|(i, _)| i)
            .expect("there should always be a child to simulate")
    }

    /// Get index of child which maximizes UCT.
    /// Losing actions are pruned unless this node is a proven loss.
    ///
    /// # Panics
    ///
    /// Panics if there are no children.
    #[must_use]
    pub fn select_with_uct(&self, beta: f32) -> usize {
        let parent_visit_count = self.visit_count as f32;
        self.children
            .iter()
            .enumerate()
            .filter(|(_, (_, child))| self.evaluation.is_loss() || !child.evaluation.is_win())
            .max_by_key(|(_, (_, child))| {
                let q = child.q_value();
                let uct = upper_confidence_bound(parent_visit_count, child.visit_count as f32);
                q + uct + child.std_dev * beta
            })
            .map(|(i, _)| i)
            .expect("there should always be a child to simulate")
    }
}

#[must_use]
pub fn sigma_select(
    q: NotNan<f32>,
    std_dev: NotNan<f32>,
    beta: f32,
    visit_count: f32,
) -> NotNan<f32> {
    (q + std_dev * beta) * (50.0 + visit_count)
}

#[must_use]
pub fn sigma_improve(
    q: NotNan<f32>,
    std_dev: NotNan<f32>,
    beta: f32,
    visit_count: f32,
) -> NotNan<f32> {
    (q + std_dev * beta) * visit_count.sqrt()
}

const EXPLORATION_BASE: f32 = 500.0;
const EXPLORATION_INIT: f32 = 4.0;

fn exploration_rate(visit_count: f32) -> f32 {
    ((1.0 + visit_count + EXPLORATION_BASE) / EXPLORATION_BASE).ln() + EXPLORATION_INIT
}

/// U(s, a) = C(s) * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
#[must_use]
pub fn upper_confidence_bound_with_predictor(
    parent_visit_count: f32,
    visit_count: f32,
    probability: f32,
) -> f32 {
    exploration_rate(parent_visit_count) * probability * parent_visit_count.sqrt()
        / (1.0 + visit_count)
}

const EXPLORATION_COEFFICIENT: f32 = 1.0;

/// U(s, a) = C * sqrt(ln(N(s)) / N(s, a))
#[must_use]
pub fn upper_confidence_bound(parent_visit_count: f32, visit_count: f32) -> f32 {
    EXPLORATION_COEFFICIENT * (parent_visit_count.ln() / visit_count).sqrt()
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::softmax;

    #[test]
    fn softmax_works() {
        let iter = [1, 2, 3, 4, 5]
            .into_iter()
            .map(|x| NotNan::new(x as f32).unwrap());

        softmax(iter)
            .zip([
                0.011_656_231,
                0.031_684_92,
                0.086_128_55,
                0.234_121_65,
                0.636_408_6,
            ])
            .for_each(|(a, b)| assert!((a - b).abs() < f32::EPSILON, "{a} should equal {b}"));
    }
}
