use std::{fmt, str::FromStr};

use arrayvec::ArrayVec;
use rand::prelude::*;
use takzero::{
    fast_tak::{
        takparse::{ParseMoveError, ParseTpsError, Tps},
        Game,
        Reserves,
        Symmetry,
    },
    search::env::Environment,
};
use thiserror::Error;

use crate::STEP;

#[derive(Clone)]
pub struct Replay<E: Environment> {
    pub env: E,                             // s_t
    pub actions: ArrayVec<E::Action, STEP>, // a_t:t+n
}

pub struct Target<E: Environment> {
    pub env: E,                          // s_t
    pub policy: Box<[(E::Action, f32)]>, // \pi'(s_t)
    pub value: f32,                      // discounted N-step value
}

pub trait Augment {
    fn augment(&self, rng: &mut impl Rng) -> Self;
}

impl<const N: usize, const HALF_KOMI: i8> Augment for Replay<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn augment(&self, rng: &mut impl Rng) -> Self {
        let index = rng.gen_range(0..8);
        Self {
            env: self.env.symmetries().into_iter().nth(index).unwrap(),
            actions: self
                .actions
                .iter()
                .map(|a| Symmetry::<N>::symmetries(a)[index])
                .collect(),
        }
    }
}

impl<const N: usize, const HALF_KOMI: i8> fmt::Display for Replay<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tps: Tps = self.env.clone().into();
        let actions = self
            .actions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        writeln!(f, "{tps};{actions}")
    }
}

#[derive(Error, Debug)]
pub enum ParseReplayError {
    #[error("missing delimiter between TPS and moves")]
    MissingDelimiter,
    #[error("{0}")]
    Tps(#[from] ParseTpsError),
    #[error("{0}")]
    Actions(#[from] ParseMoveError),
}

impl<const N: usize, const HALF_KOMI: i8> FromStr for Replay<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    type Err = ParseReplayError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (tps, actions) = s
            .split_once(';')
            .ok_or(ParseReplayError::MissingDelimiter)?;
        let tps: Tps = tps.parse()?;
        let actions = actions
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        Ok(Self {
            env: tps.into(),
            actions,
        })
    }
}
