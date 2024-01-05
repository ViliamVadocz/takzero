use ordered_float::NotNan;

use super::env::Environment;

pub trait Agent<E: Environment> {
    type Context;

    /// Always batched.
    /// The policy does not have to be normalized (returning logits).
    fn policy_value_uncertainty(
        &self,
        env_batch: &[E],
        actions_batch: &[Vec<E::Action>],
        context: &mut Self::Context,
    ) -> impl Iterator<Item = (Vec<(E::Action, NotNan<f32>)>, f32, f32)>;
}

pub mod dummy {
    use ordered_float::NotNan;

    use super::{super::env::Environment, Agent};

    pub struct Dummy;

    impl<E: Environment> Agent<E> for Dummy {
        type Context = ();

        fn policy_value_uncertainty(
            &self,
            env_batch: &[E],
            actions_batch: &[Vec<<E as Environment>::Action>],
            _context: &mut Self::Context,
        ) -> impl Iterator<Item = (Vec<(E::Action, NotNan<f32>)>, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            actions_batch.iter().map(|actions| {
                (
                    actions
                        .iter()
                        .map(|a| (a.clone(), NotNan::new(1.0).unwrap()))
                        .collect(),
                    0.0,
                    0.0,
                )
            })
        }
    }
}

pub mod simple {
    use fast_tak::{
        takparse::{Color, Move, MoveKind, Piece},
        Game,
        Reserves,
    };
    use ordered_float::NotNan;

    use super::Agent;

    pub struct Simple;

    impl<const N: usize, const HALF_KOMI: i8> Agent<Game<N, HALF_KOMI>> for Simple
    where
        Reserves<N>: Default,
    {
        type Context = ();

        fn policy_value_uncertainty(
            &self,
            env_batch: &[Game<N, HALF_KOMI>],
            actions_batch: &[Vec<Move>],
            _context: &mut Self::Context,
        ) -> impl Iterator<Item = (Vec<(Move, NotNan<f32>)>, f32, f32)> {
            debug_assert_eq!(env_batch.len(), actions_batch.len());
            env_batch.iter().zip(actions_batch).map(|(env, actions)| {
                let mut fcd = f32::from(env.board.flat_diff() - HALF_KOMI / 2) / (N * N) as f32;
                if env.to_move == Color::Black {
                    fcd = -fcd;
                }
                let policy = actions
                    .iter()
                    .map(|a| {
                        let p = match a.kind() {
                            MoveKind::Place(Piece::Flat) => 4.0,
                            MoveKind::Place(Piece::Cap) => 3.0,
                            MoveKind::Place(Piece::Wall) => 2.0,
                            MoveKind::Spread(..) => 1.0,
                        };
                        (*a, NotNan::new(p).unwrap())
                    })
                    .collect();
                (policy, fcd, 0.0)
            })
        }
    }
}
