use std::ops::Index;

use crate::env::Environment;

pub trait Agent<E: Environment> {
    type Policy: Index<E::Action, Output = f32>;
    fn value(&self, env: &E) -> f32;
    fn policy(&self, env: &E) -> Self::Policy;
}
