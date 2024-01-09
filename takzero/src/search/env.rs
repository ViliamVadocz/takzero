use std::fmt;

use fast_tak::{takparse::Move, Game, Reserves};
use rand::{seq::IteratorRandom, Rng};

pub trait Environment: Send + Sync + Clone + Default {
    type Action: Send + Sync + Clone + PartialEq + fmt::Debug;

    fn populate_actions(&self, actions: &mut Vec<Self::Action>);
    fn step(&mut self, action: Self::Action);
    fn terminal(&self) -> Option<Terminal>;
    fn steps(&self) -> u16;

    fn new_opening(rng: &mut impl Rng, actions: &mut Vec<Self::Action>) -> Self;
}

pub enum Terminal {
    Win,
    Loss,
    Draw,
}

impl<const N: usize, const HALF_KOMI: i8> Environment for Game<N, HALF_KOMI>
where
    Reserves<N>: Default,
{
    type Action = Move;

    fn populate_actions(&self, actions: &mut Vec<Self::Action>) {
        self.possible_moves(actions);
    }

    fn step(&mut self, action: Self::Action) {
        self.play(action).expect("Action should be valid");
    }

    fn terminal(&self) -> Option<Terminal> {
        match self.result() {
            fast_tak::GameResult::Winner { color, .. } => {
                if color == self.to_move {
                    Some(Terminal::Win)
                } else {
                    Some(Terminal::Loss)
                }
            }
            fast_tak::GameResult::Draw { .. } => Some(Terminal::Draw),
            fast_tak::GameResult::Ongoing => None,
        }
    }

    fn steps(&self) -> u16 {
        self.ply
    }

    fn new_opening(rng: &mut impl Rng, actions: &mut Vec<Move>) -> Self {
        let mut env = Self::default();
        env.populate_actions(actions);
        env.step(actions.drain(..).choose(rng).unwrap());
        env.populate_actions(actions);
        env.step(actions.drain(..).choose(rng).unwrap());
        env
    }
}

impl From<Terminal> for f32 {
    fn from(value: Terminal) -> Self {
        match value {
            Terminal::Win => 1.0,
            Terminal::Loss => -1.0,
            Terminal::Draw => 0.0,
        }
    }
}

#[cfg(test)]
pub mod safecrack {
    use ordered_float::NotNan;

    use super::{Environment, Terminal};
    use crate::search::agent::Agent;

    #[derive(Clone)]
    pub struct SafeCrack {
        key: Vec<u8>,
        tried: Vec<u8>,
        active: bool,
    }

    impl Default for SafeCrack {
        fn default() -> Self {
            Self::new(vec![1, 2, 3, 4])
        }
    }

    impl SafeCrack {
        #[must_use]
        pub fn new(key: Vec<u8>) -> Self {
            Self {
                key,
                tried: Vec::default(),
                active: true,
            }
        }

        fn solved(&self) -> bool {
            self.tried.starts_with(&self.key)
        }
    }

    impl Environment for SafeCrack {
        type Action = Option<u8>;

        fn populate_actions(&self, actions: &mut Vec<Self::Action>) {
            if self.active {
                for i in 0..=9 {
                    actions.push(Some(i));
                }
            } else {
                actions.push(None);
            }
        }

        fn step(&mut self, action: Self::Action) {
            if self.active {
                self.tried
                    .push(action.expect("All actions should be Some()"));
            } else {
                assert_eq!(action, None);
            }

            self.active = !self.active;
        }

        fn terminal(&self) -> Option<Terminal> {
            None // The game never ends.
        }

        fn steps(&self) -> u16 {
            unimplemented!("not necessary for the test");
        }

        fn new_opening(_rng: &mut impl rand::prelude::Rng, _actions: &mut Vec<Option<u8>>) -> Self {
            unimplemented!("not necessary for the test");
        }
    }

    pub struct SafeCracker;

    impl Agent<SafeCrack> for SafeCracker {
        type Context = ();

        fn policy_value_uncertainty(
            &self,
            env_batch: &[SafeCrack],
            actions_batch: &[Vec<<SafeCrack as Environment>::Action>],
            _context: &mut Self::Context,
        ) -> impl Iterator<Item = (Vec<(Option<u8>, NotNan<f32>)>, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            env_batch.iter().zip(actions_batch).map(|(env, actions)| {
                (
                    actions
                        .iter()
                        .map(|a| (*a, NotNan::new(1.0).unwrap()))
                        .collect(),
                    if env.active { 1.0 } else { -1.0 } * f32::from(env.solved()),
                    0.0,
                )
            })
        }
    }
}
