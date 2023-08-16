use arrayvec::ArrayVec;
use takzero::search::env::Environment;

use crate::STEP;

pub struct Replay<E: Environment> {
    pub env: E,
    pub actions: ArrayVec<E::Action, STEP>,
}

pub struct Target<E: Environment> {
    pub state: E,                        // s_t
    pub policy: Box<[(E::Action, f32)]>, // \pi'(s_t)
    pub value: f32,                      // discounted N-step value
}
