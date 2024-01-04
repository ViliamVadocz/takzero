use std::{cmp::Ordering, fmt};

use ordered_float::{FloatIsNan, NotNan};

use super::{env::Terminal, DISCOUNT_FACTOR};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Eval {
    Value(NotNan<f32>),
    Win(u32),
    Loss(u32),
    Draw(u32),
}

impl fmt::Display for Eval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(not_nan) => not_nan.into_inner().fmt(f),
            Self::Win(ply) => write!(f, "Win({ply})"),
            Self::Loss(ply) => write!(f, "Loss({ply})"),
            Self::Draw(ply) => write!(f, "Draw({ply})"),
        }
    }
}

impl Eval {
    /// # Errors
    ///
    /// Errors if the value is NaN.
    pub fn new_value(value: f32) -> Result<Self, FloatIsNan> {
        NotNan::new(value).map(Self::Value)
    }

    #[must_use]
    pub const fn new_not_nan_value(value: NotNan<f32>) -> Self {
        Self::Value(value)
    }

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
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Win(_) | Self::Draw(_) | Self::Loss(_))
    }

    #[must_use]
    pub const fn ply(&self) -> Option<u32> {
        match self {
            Self::Value(_) => None,
            Self::Win(ply) | Self::Draw(ply) | Self::Loss(ply) => Some(*ply),
        }
    }

    /// # Panics
    ///
    /// Panics if the return value of the function is NaN.
    #[must_use]
    pub fn map<F: Fn(NotNan<f32>) -> NotNan<f32>>(self, f: F) -> Self {
        match self {
            Self::Value(x) => Self::Value(f(x)),
            eval => eval,
        }
    }
}

impl Default for Eval {
    fn default() -> Self {
        Self::Value(NotNan::default())
    }
}

impl From<Eval> for f32 {
    fn from(value: Eval) -> Self {
        DISCOUNT_FACTOR.powi(value.ply().unwrap_or_default() as i32)
            * match value {
                Eval::Value(x) => x.into(),
                Eval::Win(_) => 1.0,
                Eval::Loss(_) => -1.0,
                Eval::Draw(_) => 0.0,
            }
    }
}

impl From<Eval> for NotNan<f32> {
    fn from(eval: Eval) -> Self {
        match eval {
            Eval::Value(x) => x,
            Eval::Win(_) | Eval::Draw(_) | Eval::Loss(_) => {
                Self::new(f32::from(eval)).expect("known evaluations cannot give NaN")
            }
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

pub const CONTEMPT: NotNan<f32> = unsafe { NotNan::new_unchecked(-0.05) };

impl PartialOrd for Eval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Eval {}

impl Ord for Eval {
    fn cmp(&self, other: &Self) -> Ordering {
        match self {
            Self::Value(left) => match other {
                Self::Value(right) => left.cmp(right),
                Self::Win(_) => Ordering::Less,
                Self::Draw(_) => left.cmp(&CONTEMPT),
                Self::Loss(_) => Ordering::Greater,
            },
            Self::Win(left) => match other {
                Self::Win(right) => right.cmp(left),
                _ => Ordering::Greater,
            },
            Self::Draw(left) => match other {
                Self::Value(right) => CONTEMPT.cmp(right),
                Self::Win(_) => Ordering::Less,
                Self::Draw(right) => right.cmp(left),
                Self::Loss(_) => Ordering::Greater,
            },
            Self::Loss(left) => match other {
                Self::Loss(right) => left.cmp(right),
                _ => Ordering::Less,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Eval, CONTEMPT};

    #[test]
    fn eval_order() {
        let mut evals = [
            Eval::new_value(1.0).unwrap(),
            Eval::Value(CONTEMPT + 0.1),
            Eval::new_value(-1.0).unwrap(),
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
            Eval::new_value(-1.0).unwrap(),
            Eval::Draw(10),
            Eval::Draw(5),
            Eval::Value(CONTEMPT + 0.1),
            Eval::new_value(1.0).unwrap(),
            Eval::Win(10),
            Eval::Win(5),
        ]);
    }
}
