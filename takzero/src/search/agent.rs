use std::ops::Index;

use super::env::Environment;

pub trait Agent<E: Environment> {
    type Policy: Index<E::Action, Output = f32>;
    fn value(&self, env: &E) -> f32;
    fn policy(&self, env: &E) -> Self::Policy;
}

// #[cfg(test)]
pub mod dummy {
    use std::ops::Index;

    use super::{super::env::Environment, Agent};

    pub struct Dummy;

    impl<E: Environment> Agent<E> for Dummy {
        type Policy = Policy;

        fn value(&self, _: &E) -> f32 {
            0.0
        }

        fn policy(&self, env: &E) -> Self::Policy {
            let mut actions = Vec::new();
            env.populate_actions(&mut actions);
            Policy(1.0 / actions.len() as f32)
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
