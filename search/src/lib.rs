#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![warn(clippy::nursery)]
// https://github.com/rust-lang/rust-clippy/issues/8538
#![allow(clippy::iter_with_drain)]

pub mod agent;
pub mod env;
pub mod eval;

