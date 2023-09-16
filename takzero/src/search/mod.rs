pub mod agent;
pub mod env;
pub mod eval;
pub mod node;

// Steps for TD-learning.
pub const STEP: usize = 5;
// Discount, also known as gamma.
pub const DISCOUNT_FACTOR: f32 = 0.997;
pub const SERIES_DISCOUNT: f32 = 1.0 / (1.0 - DISCOUNT_FACTOR * DISCOUNT_FACTOR);
