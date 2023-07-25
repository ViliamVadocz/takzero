#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![warn(clippy::nursery)]
// https://github.com/rust-lang/rust-clippy/issues/8538
#![allow(clippy::iter_with_drain)]
// Just let me cast in peace
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
// We're going nightly boys
#![feature(extract_if)]

pub mod model;
pub mod repr;
pub mod search;
