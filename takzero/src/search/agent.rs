use std::ops::Index;

use super::env::Environment;

pub trait Agent<E: Environment> {
    type Policy: Index<E::Action, Output = f32>;
    fn policy_value(&self, env: &E, actions: &[E::Action]) -> (Self::Policy, f32);
}

pub mod dummy {
    use std::ops::Index;

    use super::{super::env::Environment, Agent};

    pub struct Dummy;

    impl<E: Environment> Agent<E> for Dummy {
        type Policy = Policy;

        fn policy_value(&self, _: &E, actions: &[E::Action]) -> (Self::Policy, f32) {
            (Policy(1.0 / actions.len() as f32), 0.0)
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
