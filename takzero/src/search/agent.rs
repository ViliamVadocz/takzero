use std::ops::Index;

use super::env::Environment;

pub trait Agent<E: Environment> {
    type Policy: Index<E::Action, Output = f32> + Send + Sync;
    type Context;

    // Always batched.
    fn policy_value_uncertainty(
        &self,
        env_batch: &[E],
        actions_batch: &[Vec<E::Action>],
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

        fn policy_value_uncertainty(
            &self,
            env_batch: &[E],
            actions_batch: &[Vec<<E as Environment>::Action>],
            _context: &mut Self::Context,
        ) -> Vec<(Self::Policy, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            actions_batch
                .iter()
                .map(|actions| (Policy(1.0 / actions.len() as f32), 0.0, 0.0))
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
