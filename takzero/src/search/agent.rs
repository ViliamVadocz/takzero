use std::ops::Index;

use super::env::Environment;

pub trait Agent<E: Environment> {
    type Policy: Index<E::Action, Output = f32> + Send + Sync;
    type Context;

    /// Always batched.
    /// The policy does not have to be normalized (returning logits).
    fn policy_value_uncertainty(
        &self,
        env_batch: &[E],
        actions_batch: &[Vec<E::Action>],
        context: &mut Self::Context,
    ) -> impl Iterator<Item = (Self::Policy, f32, f32)>;
}

pub mod dummy {
    use std::ops::Index;

    use super::{super::env::Environment, Agent};

    pub struct Dummy;

    impl<E: Environment> Agent<E> for Dummy {
        type Context = ();
        type Policy = Policy;

        fn policy_value_uncertainty(
            &self,
            env_batch: &[E],
            actions_batch: &[Vec<<E as Environment>::Action>],
            _context: &mut Self::Context,
        ) -> impl Iterator<Item = (Self::Policy, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            actions_batch.iter().map(|_| (Policy, 0.0, 0.0))
        }
    }

    pub struct Policy;

    impl<T> Index<T> for Policy {
        type Output = f32;

        fn index(&self, _: T) -> &Self::Output {
            &1.0
        }
    }
}

pub mod simple {
    use std::ops::Index;

    use fast_tak::{
        takparse::{Color, Move, MoveKind, Piece},
        Game,
        Reserves,
    };

    use super::Agent;

    pub struct Simple;

    impl<const N: usize, const HALF_KOMI: i8> Agent<Game<N, HALF_KOMI>> for Simple
    where
        Reserves<N>: Default,
    {
        type Context = ();
        type Policy = Policy;

        fn policy_value_uncertainty(
            &self,
            env_batch: &[Game<N, HALF_KOMI>],
            actions_batch: &[Vec<Move>],
            _context: &mut Self::Context,
        ) -> impl Iterator<Item = (Self::Policy, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            env_batch.iter().map(|env| {
                let mut fcd = f32::from(env.board.flat_diff() - HALF_KOMI / 2) / (N * N) as f32;
                if env.to_move == Color::Black {
                    fcd = -fcd;
                }
                (Policy, fcd, 0.0)
            })
        }
    }

    pub struct Policy;

    impl Index<Move> for Policy {
        type Output = f32;

        fn index(&self, action: Move) -> &Self::Output {
            match action.kind() {
                MoveKind::Place(Piece::Flat) => &4.0,
                MoveKind::Place(Piece::Cap) => &3.0,
                MoveKind::Place(Piece::Wall) => &2.0,
                MoveKind::Spread(..) => &1.0,
            }
        }
    }
}
