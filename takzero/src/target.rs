use std::{
    collections::VecDeque,
    fmt,
    fs::OpenOptions,
    io::{BufRead, BufReader},
    num::ParseFloatError,
    path::Path,
    str::FromStr,
};

use fast_tak::{
    takparse::{GameResult, ParseMoveError, ParsePtnError, ParseTpsError, Ptn, Tps},
    Game,
    PlayError,
    Reserves,
    Symmetry,
};
use ordered_float::{FloatIsNan, NotNan};
use rand::prelude::*;
use thiserror::Error;

use crate::search::{env::Environment, node::Node};

#[derive(Debug, PartialEq)]
pub struct Target<E: Environment> {
    pub env: E,                                  // s_t
    pub policy: Box<[(E::Action, NotNan<f32>)]>, // \pi'(s_t)
    pub value: f32,                              // discounted N-step value
    pub ube: f32,                                // sum of RND + discounted N-step UBE
}

pub trait Augment {
    #[must_use]
    fn augment(&self, rng: &mut impl Rng) -> Self;
}

impl<const N: usize, const HALF_KOMI: i8> Augment for Target<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn augment(&self, rng: &mut impl Rng) -> Self {
        let index = rng.gen_range(0..8);
        Self {
            env: self.env.symmetries().into_iter().nth(index).unwrap(),
            value: self.value,
            ube: self.ube,
            policy: self
                .policy
                .iter()
                .map(|(mov, p)| (Symmetry::<N>::symmetries(mov)[index], *p))
                .collect(),
        }
    }
}

impl<const N: usize, const HALF_KOMI: i8> fmt::Display for Target<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tps: Tps = self.env.clone().into();
        let value = self.value;
        let ube = self.ube;
        let policy = self
            .policy
            .iter()
            .map(|(mov, p)| format!("{mov}:{p}"))
            .collect::<Vec<_>>()
            .join(",");

        writeln!(f, "{tps};{value};{ube};{policy}")
    }
}

#[derive(Error, Debug)]
pub enum ParseTargetError {
    #[error("missing TPS")]
    MissingTps,
    #[error("missing value")]
    MissingValue,
    #[error("missing UBE")]
    MissingUbe,
    #[error("missing policy")]
    MissingPolicy,
    #[error("policy format is wrong")]
    WrongPolicyFormat,
    #[error("{0}")]
    Tps(#[from] ParseTpsError),
    #[error("{0}")]
    Action(#[from] ParseMoveError),
    #[error("{0}")]
    Float(#[from] ParseFloatError),
    #[error("{0}")]
    PolicyNan(#[from] FloatIsNan),
    #[error("the policy does not contain the right actions")]
    PolicyWrongActions,
}

impl<const N: usize, const HALF_KOMI: i8> FromStr for Target<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    type Err = ParseTargetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        //{tps};{value};{ube};{policy}
        let mut iter = s.trim().split(';');
        let tps: Tps = iter.next().ok_or(ParseTargetError::MissingTps)?.parse()?;
        let value = iter.next().ok_or(ParseTargetError::MissingValue)?.parse()?;
        let ube = iter.next().ok_or(ParseTargetError::MissingUbe)?.parse()?;
        let policy: Box<_> = iter
            .next()
            .ok_or(ParseTargetError::MissingPolicy)?
            .split(',')
            .map(|s| {
                s.split_once(':')
                    .ok_or(ParseTargetError::WrongPolicyFormat)
                    .and_then(|(a, p)| Ok((a.parse()?, NotNan::new(p.parse()?)?)))
            })
            .collect::<Result<_, _>>()?;
        let env: Game<N, HALF_KOMI> = tps.into();

        // Check that all actions that should be in the policy are in the policy,
        // and that there are no extras.
        let mut actions = vec![];
        env.populate_actions(&mut actions);
        if actions.len() != policy.len() {
            return Err(ParseTargetError::PolicyWrongActions);
        }
        for action in actions {
            if !policy.iter().any(|(a, _)| *a == action) {
                return Err(ParseTargetError::PolicyWrongActions);
            }
        }

        Ok(Self {
            env,
            policy,
            value,
            ube,
        })
    }
}

/// Create an improved policy target of proportional visit counts.
///
/// # Panics
///
/// Panics if the target policy for any move is NaN.
#[must_use]
pub fn policy_target_from_proportional_visits<E: Environment>(
    node: &Node<E>,
) -> Box<[(E::Action, NotNan<f32>)]> {
    node.children
        .iter()
        .map(|(action, child)| {
            (
                action.clone(),
                NotNan::new(child.visit_count as f32 / node.visit_count as f32)
                    .expect("target policy should not be NaN"),
            )
        })
        .collect()
}

#[derive(Debug, PartialEq, Clone)]
pub struct Replay<E: Environment> {
    pub env: E,
    pub actions: VecDeque<E::Action>,
}

impl<E: Environment> Replay<E> {
    /// Start a new replay from an initial position.
    pub const fn new(env: E) -> Self {
        Self {
            env,
            actions: VecDeque::new(),
        }
    }

    /// Add an action to the replay.
    pub fn push(&mut self, action: E::Action) {
        self.actions.push_back(action);
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Advance the replay state by `steps`.
    ///
    /// # Panics
    ///
    /// Panics if there are fewer actions in the replay than requested steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.env.step(self.actions.pop_front().unwrap());
        }
    }
}

impl<const N: usize, const HALF_KOMI: i8> fmt::Display for Replay<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[TPS \"{}\"]", Tps::from(self.env.clone()),)?;
        let mut env = self.env.clone();
        for action in &self.actions {
            write!(f, " {action}")?;
            env.step(*action);
        }
        if let Ok(result) = GameResult::try_from(env.result()) {
            writeln!(f, " {result}")
        } else {
            writeln!(f)
        }
    }
}

#[derive(Error, Debug)]
pub enum ParseReplayError {
    #[error("{0}")]
    Ptn(#[from] ParsePtnError),
    #[error("missing TPS")]
    MissingTps,
    #[error("{0}")]
    Tps(#[from] ParseTpsError),
    #[error("{0}")]
    Action(#[from] ParseMoveError),
    #[error("invalid action")]
    Invalid(#[from] PlayError),
}

impl<const N: usize, const HALF_KOMI: i8> FromStr for Replay<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    type Err = ParseReplayError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ptn: Ptn = s.parse()?;
        let env: Game<N, HALF_KOMI> = ptn.tps().ok_or(ParseReplayError::MissingTps)?.into();

        let actions: VecDeque<_> = ptn.moves().iter().copied().collect();

        // Verify that the actions are valid.
        let mut test_env = env.clone();
        for &action in &actions {
            test_env.play(action)?;
        }

        Ok(Self { env, actions })
    }
}

/// Open a file and parse all the replays (stored one per line).
///
/// # Errors
///
/// Returns an error if the file cannot be opened.
pub fn get_replays<const N: usize, const HALF_KOMI: i8>(
    path: impl AsRef<Path>,
) -> Result<impl Iterator<Item = Replay<Game<N, HALF_KOMI>>>, std::io::Error>
where
    Reserves<N>: Default,
{
    Ok(BufReader::new(OpenOptions::new().read(true).open(path)?)
        .lines()
        .filter_map(|line| line.ok()?.parse::<Replay<Game<N, HALF_KOMI>>>().ok()))
}

/// Open a file and parse all the targets (stored one per line).
///
/// # Errors
///
/// Returns an error if the file cannot be opened.
pub fn get_targets<const N: usize, const HALF_KOMI: i8>(
    path: impl AsRef<Path>,
) -> Result<impl Iterator<Item = Target<Game<N, HALF_KOMI>>>, std::io::Error>
where
    Reserves<N>: Default,
{
    Ok(BufReader::new(OpenOptions::new().read(true).open(path)?)
        .lines()
        .filter_map(|line| line.ok()?.parse::<Target<Game<N, HALF_KOMI>>>().ok()))
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use ordered_float::NotNan;
    use rand::{seq::IteratorRandom, Rng, SeedableRng};

    use crate::{
        search::env::Environment,
        target::{Replay, Target},
    };

    #[test]
    fn target_consistency() {
        const SEED: u64 = 123;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut env: Game<5, 4> = Game::default();
        let mut actions = Vec::new();
        while env.terminal().is_none() {
            env.populate_actions(&mut actions);
            let target = Target {
                env: {
                    let mut c = env.clone();
                    c.reversible_plies = 0;
                    c
                },
                policy: actions
                    .iter()
                    .map(|a| (*a, NotNan::new(rng.gen()).unwrap()))
                    .collect(),
                value: rng.gen(),
                ube: rng.gen(),
            };
            let string = target.to_string();
            println!("{string}");

            let recovered: Target<_> = string.parse().unwrap();
            let string_again = recovered.to_string();

            assert_eq!(target, recovered);
            assert_eq!(string, string_again);

            env.step(actions.drain(..).choose(&mut rng).unwrap());
        }
    }

    #[test]
    fn replay_consistency() {
        const SEED: u64 = 123;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut actions = Vec::new();
        for _ in 0..100 {
            let mut env: Game<5, 4> = Game::new_opening(&mut rng, &mut actions);
            let mut replay = Replay::new(env.clone());

            loop {
                env.populate_actions(&mut actions);
                let action = actions.drain(..).choose(&mut rng).unwrap();
                replay.push(action);
                env.step(action);

                let string = replay.to_string();
                println!("{string}");

                let recovered: Replay<_> = string.parse().unwrap();
                let string_again = recovered.to_string();

                assert_eq!(replay, recovered);
                assert_eq!(string, string_again);

                if env.terminal().is_some() {
                    break;
                }
            }
        }
    }
}
