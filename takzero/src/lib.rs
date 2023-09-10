#![warn(clippy::pedantic, clippy::style, clippy::nursery)]
// https://github.com/rust-lang/rust-clippy/issues/8538
#![allow(clippy::iter_with_drain)]
// Just let me cast in peace
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
// This one is just annoying
#![allow(clippy::module_name_repetitions)]

pub mod network;
pub mod search;
