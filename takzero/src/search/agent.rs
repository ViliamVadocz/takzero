use std::ops::Index;

use super::env::Environment;

pub trait Agent<E: Environment> {
    type Policy: Index<E::Action, Output = f32> + Send + Sync;
    type Context;

    /// Always batched. Mask is true for environments which need eval and for
    /// those the context should be updated. The length of the output should
    /// correspond to the number of true values in the mask.
    /// The policy does not have to be normalized (returning logits).
    fn policy_value_uncertainty(
        &self,
        env_batch: &[E],
        actions_batch: &[Vec<E::Action>],
        mask: &[bool],
        context: &mut Self::Context,
    ) -> Vec<(Self::Policy, f32, f32)>;
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
            mask: &[bool],
            _context: &mut Self::Context,
        ) -> Vec<(Self::Policy, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            actions_batch
                .iter()
                .zip(mask)
                .filter(|(_, mask)| **mask)
                .map(|_| (Policy, 0.0, 0.0))
                .collect()
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
        takparse::{Move, MoveKind, Piece},
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
            mask: &[bool],
            _context: &mut Self::Context,
        ) -> Vec<(Self::Policy, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            actions_batch
                .iter()
                .zip(env_batch)
                .zip(mask)
                .filter(|(_, mask)| **mask)
                .map(|((_, env), _)| {
                    let fcd = f32::from(env.board.flat_diff() - HALF_KOMI / 2) / (N * N) as f32;
                    (Policy, fcd, 0.0)
                })
                .collect()
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
