use fast_tak::{takparse::Move, Game};

pub struct Replay<const N: usize, const HALF_KOMI: i8> {
    state: Game<N, HALF_KOMI>,
    action: Move,
}

pub struct Target<const N: usize, const HALF_KOMI: i8> {
    state: Game<N, HALF_KOMI>,  // s_t
    policy: Box<[(Move, f32)]>, // \pi'(s_t)
    value: f32,                 // V(s_{t+1})
}
