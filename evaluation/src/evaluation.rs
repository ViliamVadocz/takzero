use std::{iter::Sum, ops::AddAssign};

#[derive(Debug, Default)]
pub struct Evaluation {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl AddAssign for Evaluation {
    fn add_assign(&mut self, rhs: Self) {
        self.wins += rhs.wins;
        self.losses += rhs.losses;
        self.draws += rhs.draws;
    }
}

impl Sum for Evaluation {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut a, b| {
            a += b;
            a
        })
    }
}

impl Evaluation {
    pub fn win_rate(&self) -> f32 {
        // TODO: Think about whether we should ignore draws or not.
        #![allow(clippy::cast_precision_loss)]
        self.wins as f32 / (self.wins + self.draws + self.losses) as f32
    }

    pub fn win() -> Self {
        Self {
            wins: 1,
            ..Default::default()
        }
    }

    pub fn loss() -> Self {
        Self {
            losses: 1,
            ..Default::default()
        }
    }

    pub fn draw() -> Self {
        Self {
            draws: 1,
            ..Default::default()
        }
    }
}
