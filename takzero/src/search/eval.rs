use std::cmp::Ordering;

use super::env::Terminal;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Eval {
    Value(f32),
    Win(u32),
    Loss(u32),
    Draw(u32),
}

impl Eval {
    #[must_use]
    pub fn negate(&self) -> Self {
        match *self {
            Self::Value(value) => Self::Value(-value),
            Self::Win(ply) => Self::Loss(ply + 1),
            Self::Draw(ply) => Self::Draw(ply + 1),
            Self::Loss(ply) => Self::Win(ply + 1),
        }
    }

    #[must_use]
    pub const fn is_win(&self) -> bool {
        matches!(self, Self::Win(_))
    }

    #[must_use]
    pub const fn is_draw(&self) -> bool {
        matches!(self, Self::Draw(_))
    }

    #[must_use]
    pub const fn is_loss(&self) -> bool {
        matches!(self, Self::Loss(_))
    }

    #[must_use]
    pub const fn ply(&self) -> Option<u32> {
        match self {
            Self::Value(_) => None,
            Self::Win(ply) | Self::Draw(ply) | Self::Loss(ply) => Some(*ply),
        }
    }
}

impl Default for Eval {
    fn default() -> Self {
        Self::Value(Default::default())
    }
}

impl From<Eval> for f32 {
    fn from(value: Eval) -> Self {
        match value {
            Eval::Value(x) => x,
            Eval::Win(_) => 1.0,
            Eval::Loss(_) => -1.0,
            Eval::Draw(_) => 0.0,
        }
    }
}

impl From<Terminal> for Eval {
    fn from(value: Terminal) -> Self {
        match value {
            Terminal::Win => Self::Win(0),
            Terminal::Loss => Self::Loss(0),
            Terminal::Draw => Self::Draw(0),
        }
    }
}

pub const CONTEMPT: f32 = -0.05;

impl PartialOrd for Eval {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self {
            Self::Value(left) => match other {
                Self::Value(right) => left.partial_cmp(right),
                Self::Win(_) => Some(Ordering::Less),
                Self::Draw(_) => left.partial_cmp(&CONTEMPT),
                Self::Loss(_) => Some(Ordering::Greater),
            },
            Self::Win(left) => match other {
                Self::Win(right) => right.partial_cmp(left),
                _ => Some(Ordering::Greater),
            },
            Self::Draw(left) => match other {
                Self::Value(right) => CONTEMPT.partial_cmp(right),
                Self::Win(_) => Some(Ordering::Less),
                Self::Draw(right) => left.partial_cmp(right),
                Self::Loss(_) => Some(Ordering::Greater),
            },
            Self::Loss(left) => match other {
                Self::Loss(right) => left.partial_cmp(right),
                _ => Some(Ordering::Less),
            },
        }
    }
}

impl Eq for Eval {}

impl Ord for Eval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{Eval, CONTEMPT};

    #[test]
    fn eval_order() {
        let mut evals = [
            Eval::Value(1.0),
            Eval::Value(CONTEMPT + 0.1),
            Eval::Value(-1.0),
            Eval::Win(5),
            Eval::Win(10),
            Eval::Draw(5),
            Eval::Draw(10),
            Eval::Loss(5),
            Eval::Loss(10),
        ];
        evals.sort();
        assert_eq!(evals, [
            Eval::Loss(5),
            Eval::Loss(10),
            Eval::Value(-1.0),
            Eval::Draw(5),
            Eval::Draw(10),
            Eval::Value(CONTEMPT + 0.1),
            Eval::Value(1.0),
            Eval::Win(10),
            Eval::Win(5),
        ]);
    }
}
