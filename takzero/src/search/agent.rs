use std::ops::Index;

use super::env::Environment;

pub trait Agent<E: Environment> {
    type Policy: Index<E::Action, Output = f32> + Send + Sync;
    type Context;

    #[cfg(feature = "baseline")]
    fn policy_value(
        &self,
        env_batch: &[E],
        actions_batch: &[Vec<E::Action>],
        mask: &[bool],
        context: &mut Self::Context,
    ) -> Vec<(Self::Policy, f32)>;

    #[cfg(not(feature = "baseline"))]
    /// Always batched. Mask is true for environments which need eval and for
    /// those the context should be updated. The length of the output should
    /// correspond to the number of true values in the mask.
    fn policy_value_uncertainty(
        &self,
        env_batch: &[E],
        actions_batch: &[Vec<E::Action>],
        mask: &[bool],
        context: &mut Self::Context,
    ) -> Vec<(Self::Policy, f32, f32)>;
}

pub mod dummy {
    use std::ops::Index;

    use super::{super::env::Environment, Agent};

    pub struct Dummy;

    impl<E: Environment> Agent<E> for Dummy {
        type Context = ();
        type Policy = Policy;

        #[cfg(feature = "baseline")]
        fn policy_value(
            &self,
            env_batch: &[E],
            actions_batch: &[Vec<<E as Environment>::Action>],
            mask: &[bool],
            _context: &mut Self::Context,
        ) -> Vec<(Self::Policy, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            actions_batch
                .iter()
                .zip(mask)
                .filter(|(_, mask)| **mask)
                .map(|(actions, _)| (Policy(1.0 / actions.len() as f32), 0.0))
                .collect()
        }

        #[cfg(not(feature = "baseline"))]
        fn policy_value_uncertainty(
            &self,
            env_batch: &[E],
            actions_batch: &[Vec<<E as Environment>::Action>],
            mask: &[bool],
            _context: &mut Self::Context,
        ) -> Vec<(Self::Policy, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            actions_batch
                .iter()
                .zip(mask)
                .filter(|(_, mask)| **mask)
                .map(|(actions, _)| (Policy(1.0 / actions.len() as f32), 0.0, 0.0))
                .collect()
        }
    }

    pub struct Policy(f32);

    impl<T> Index<T> for Policy {
        type Output = f32;

        fn index(&self, _: T) -> &Self::Output {
            &self.0
        }
    }
}
