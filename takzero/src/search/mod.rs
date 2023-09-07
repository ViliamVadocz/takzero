pub mod agent;
pub mod env;
pub mod eval;
pub mod node;

// Steps for TD-learning.
pub const STEP: usize = 5;
// Discount, also known as gamma.
pub const DISCOUNT_FACTOR: f32 = 0.99;
pub const GEOMETRIC_SUM_DISCOUNT: f32 = {
    // Calculate DISCOUNT_FACTOR^STEP.
    let mut exp = STEP;
    let mut pow = 1.0;
    while exp > 0 {
        pow *= DISCOUNT_FACTOR;
        exp -= 1;
    }
    // Geometric sum.
    1.0 - pow / (1.0 - DISCOUNT_FACTOR * DISCOUNT_FACTOR)
};
