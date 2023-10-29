use std::ops::Index;

use fast_tak::{takparse::Move, Game};
use rayon::prelude::*;
use tch::{
    nn::{self, ModuleT},
    Device,
    Kind,
    Reduction,
    Tensor,
};

use super::{
    repr::{game_to_tensor, input_channels, input_size, move_index, output_channels, output_size},
    residual::ResidualBlock,
    Network,
};
use crate::{
    network::repr::move_mask,
    search::{agent::Agent, SERIES_DISCOUNT},
};

const N: usize = 5;
const FILTERS: i64 = 128;
const CORE_RES_BLOCKS: u32 = 10;

#[derive(Debug)]
pub struct Net5 {
    vs: nn::VarStore,
    core: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head: nn::SequentialT,
    #[cfg(not(feature = "baseline"))]
    ube_head: nn::SequentialT,
    #[cfg(not(feature = "baseline"))]
    rnd: Rnd,
}

#[cfg(not(feature = "baseline"))]
#[derive(Debug)]
struct Rnd {
    target: nn::SequentialT,
    learning: nn::SequentialT,
}

fn core(path: &nn::Path) -> nn::SequentialT {
    let mut core = nn::seq_t()
        .add(nn::conv2d(
            path / "input_conv2d",
            input_channels::<N>() as i64,
            FILTERS,
            3,
            nn::ConvConfig {
                stride: 1,
                padding: 1,
                bias: false,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(
            path / "batch_norm",
            FILTERS,
            nn::BatchNormConfig::default(),
        ))
        .add_fn(Tensor::relu);
    for n in 0..CORE_RES_BLOCKS {
        core = core.add(ResidualBlock::new(
            &(path / format!("res_block_{n}")),
            FILTERS,
            FILTERS,
        ));
    }
    core
}

fn policy_head(path: &nn::Path) -> nn::SequentialT {
    nn::seq_t()
        .add(ResidualBlock::new(&(path / "res_block"), FILTERS, FILTERS))
        .add(nn::conv2d(
            path / "conv2d",
            FILTERS,
            output_channels::<N>() as i64,
            3,
            nn::ConvConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
        ))
}

fn value_head(path: &nn::Path) -> nn::SequentialT {
    nn::seq_t()
        .add(ResidualBlock::new(&(path / "res_block"), FILTERS, FILTERS))
        .add(nn::conv2d(path / "conv2d", FILTERS, 1, 1, nn::ConvConfig {
            stride: 1,
            ..Default::default()
        }))
        .add_fn(Tensor::relu)
        .add_fn(|x| x.view([-1, (N * N) as i64]))
        .add(nn::linear(
            path / "linear",
            (N * N) as i64,
            1,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::tanh)
}

#[cfg(not(feature = "baseline"))]
fn ube_head(path: &nn::Path) -> nn::SequentialT {
    nn::seq_t()
        .add(ResidualBlock::new(&(path / "res_block"), FILTERS, FILTERS))
        .add(nn::conv2d(path / "conv2d", FILTERS, 1, 1, nn::ConvConfig {
            stride: 1,
            ..Default::default()
        }))
        .add_fn(Tensor::relu)
        .add_fn(|x| x.view([-1, (N * N) as i64]))
        .add(nn::linear(
            path / "linear",
            (N * N) as i64,
            1,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::square)
}

#[cfg(not(feature = "baseline"))]
fn rnd(path: &nn::Path) -> nn::SequentialT {
    const HIDDEN_LAYER: i64 = 1024;
    const OUTPUT: i64 = 512;
    nn::seq_t()
        .add_fn(|x| x.view([-1, input_size::<N>() as i64]))
        .add(nn::linear(
            path / "input_linear",
            input_size::<N>() as i64,
            HIDDEN_LAYER,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            path / "hidden_linear",
            HIDDEN_LAYER,
            HIDDEN_LAYER,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            path / "final_linear",
            HIDDEN_LAYER,
            OUTPUT,
            nn::LinearConfig::default(),
        ))
}

impl Network for Net5 {
    fn new(device: Device, seed: Option<i64>) -> Self {
        if let Some(seed) = seed {
            tch::manual_seed(seed);
        }

        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let core = core(&(&root / "core"));
        let policy_head = policy_head(&(&root / "policy"));
        let value_head = value_head(&(&root / "value"));
        #[cfg(not(feature = "baseline"))]
        let ube_head = ube_head(&(&root / "ube"));
        #[cfg(not(feature = "baseline"))]
        let rnd = Rnd {
            learning: rnd(&(&root / "rnd_learning")),
            target: rnd(&(&root / "rnd_target")),
        };

        Self {
            vs,
            core,
            policy_head,
            value_head,
            #[cfg(not(feature = "baseline"))]
            ube_head,
            #[cfg(not(feature = "baseline"))]
            rnd,
        }
    }

    fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    #[cfg(feature = "baseline")]
    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let s = self.core.forward_t(xs, train);
        (
            self.policy_head.forward_t(&s, train),
            self.value_head.forward_t(&s, train),
        )
    }

    #[cfg(not(feature = "baseline"))]
    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor, Tensor) {
        let s = self.core.forward_t(xs, train);
        (
            self.policy_head.forward_t(&s, train),
            self.value_head.forward_t(&s, train),
            self.ube_head.forward_t(&s, train),
        )
    }

    #[cfg(not(feature = "baseline"))]
    fn forward_rnd(&self, xs: &Tensor, train: bool) -> Tensor {
        self.rnd
            .learning
            .forward_t(xs, train)
            .mse_loss(
                &self
                    .rnd
                    .target
                    .forward_t(&xs.set_requires_grad(false), false)
                    .detach(),
                Reduction::None,
            )
            .sum_dim_intlist(1, false, None)
    }
}

type Env = Game<N, 4>;

impl Agent<Env> for Net5 {
    type Context = RndNormalizationContext;
    type Policy = Policy;

    #[cfg(feature = "baseline")]
    fn policy_value(
        &self,
        env_batch: &[Env],
        actions_batch: &[Vec<<Env as crate::search::env::Environment>::Action>],
        env_mask: &[bool],
        _context: &mut Self::Context,
    ) -> Vec<(Self::Policy, f32)> {
        debug_assert_eq!(env_batch.len(), actions_batch.len());
        debug_assert_eq!(env_batch.len(), env_mask.len());
        let device = self.vs.device();

        let xs = Tensor::cat(
            &env_batch
                .par_iter()
                .zip(env_mask)
                .map(|(env, mask)| {
                    if *mask {
                        game_to_tensor(env, device)
                    } else {
                        Tensor::zeros(
                            [1, input_channels::<N>() as i64, N as i64, N as i64],
                            (Kind::Float, device),
                        )
                    }
                })
                .collect::<Vec<_>>(),
            0,
        );
        let move_mask = Tensor::cat(
            &actions_batch
                .par_iter()
                .zip(env_mask)
                .map(|(m, mask)| {
                    if *mask {
                        move_mask::<N>(m, device)
                    } else {
                        Tensor::zeros(
                            [1, output_channels::<N>() as i64, N as i64, N as i64],
                            (Kind::Bool, device),
                        )
                    }
                })
                .collect::<Vec<_>>(),
            0,
        );
        // let env_mask_tensor = Tensor::from_slice(env_mask).to(device);

        let (policy, values) = self.forward_t(&xs, false);
        let masked_policy: Vec<Vec<_>> = policy
            .masked_fill(&move_mask, f64::from(f32::MIN))
            .view([-1, output_size::<N>() as i64])
            .softmax(1, Kind::Float)
            .try_into()
            .unwrap();
        let values: Vec<_> = values.view([-1]).try_into().unwrap();

        masked_policy
            .into_iter()
            .map(Policy)
            .zip(values)
            .zip(env_mask)
            .filter(|(_, mask)| **mask)
            .map(|((p, v), _)| (p, v))
            .collect()
    }

    #[cfg(not(feature = "baseline"))]
    fn policy_value_uncertainty(
        &self,
        env_batch: &[Env],
        actions_batch: &[Vec<<Env as crate::search::env::Environment>::Action>],
        env_mask: &[bool],
        context: &mut Self::Context,
    ) -> Vec<(Self::Policy, f32, f32)> {
        debug_assert_eq!(env_batch.len(), actions_batch.len());
        debug_assert_eq!(env_batch.len(), env_mask.len());
        let device = self.vs.device();

        let xs = Tensor::cat(
            &env_batch
                .par_iter()
                .zip(env_mask)
                .map(|(env, mask)| {
                    if *mask {
                        game_to_tensor(env, device)
                    } else {
                        Tensor::zeros(
                            [1, input_channels::<N>() as i64, N as i64, N as i64],
                            (Kind::Float, device),
                        )
                    }
                })
                .collect::<Vec<_>>(),
            0,
        );
        let move_mask = Tensor::cat(
            &actions_batch
                .par_iter()
                .zip(env_mask)
                .map(|(m, mask)| {
                    if *mask {
                        move_mask::<N>(m, device)
                    } else {
                        Tensor::zeros(
                            [1, output_channels::<N>() as i64, N as i64, N as i64],
                            (Kind::Bool, device),
                        )
                    }
                })
                .collect::<Vec<_>>(),
            0,
        );
        // let env_mask_tensor = Tensor::from_slice(env_mask).to(device);

        let (policy, values, ube_uncertainties) = self.forward_t(&xs, false);
        let masked_policy: Vec<Vec<_>> = policy
            .masked_fill(&move_mask, f64::from(f32::MIN))
            .view([-1, output_size::<N>() as i64])
            .softmax(1, Kind::Float)
            .try_into()
            .unwrap();
        let values: Vec<_> = values.view([-1]).try_into().unwrap();

        // Uncertainty.
        let rnd_uncertainties = context.normalize(&self.forward_rnd(&xs, false));
        let uncertainties: Vec<_> = ube_uncertainties
            .maximum(&(SERIES_DISCOUNT * rnd_uncertainties))
            .clip(0.0, 1.0)
            .view([-1])
            .try_into()
            .unwrap();

        masked_policy
            .into_iter()
            .map(Policy)
            .zip(values)
            .zip(uncertainties)
            .zip(env_mask)
            .filter(|(_, mask)| **mask)
            .map(|(((p, v), u), _)| (p, v, u))
            .collect()
    }
}

pub struct Policy(Vec<f32>);

impl Index<Move> for Policy {
    type Output = f32;

    fn index(&self, index: Move) -> &Self::Output {
        &self.0[move_index::<N>(&index)]
    }
}

pub struct RndNormalizationContext {
    last_training_loss: f64,
}

impl RndNormalizationContext {
    #[must_use]
    pub const fn new(last_training_loss: f64) -> Self {
        Self { last_training_loss }
    }

    pub fn normalize(&mut self, rnd: &Tensor) -> Tensor {
        (rnd - self.last_training_loss).clamp(0.0, 1.0)
    }
}

#[cfg(not(feature = "baseline"))]
#[cfg(test)]
mod tests {
    use std::array;

    use fast_tak::Game;
    use tch::Device;

    use super::{Env, Net5, RndNormalizationContext};
    use crate::{
        network::Network,
        search::{agent::Agent, env::Environment},
    };

    #[test]
    fn evaluate() {
        let net = Net5::new(Device::cuda_if_available(), Some(123));
        let game: Env = Game::default();
        let mut moves = Vec::new();
        let mut context = RndNormalizationContext::new(1.0);
        game.possible_moves(&mut moves);
        let (_policy, _value, _uncertainty) = net
            .policy_value_uncertainty(&[game], &[moves], &[true], &mut context)
            .pop()
            .unwrap();
    }

    #[test]
    fn evaluate_batch() {
        const BATCH_SIZE: usize = 128;
        let net = Net5::new(Device::cuda_if_available(), Some(456));
        let mut games: [Env; BATCH_SIZE] = array::from_fn(|_| Game::default());
        let mut actions_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        let mut context = RndNormalizationContext::new(1.0);
        let mask = [[false; BATCH_SIZE / 2], [true; BATCH_SIZE / 2]].concat();
        games
            .iter_mut()
            .zip(&mut actions_batch)
            .for_each(|(game, actions)| game.populate_actions(actions));
        let output = net.policy_value_uncertainty(&games, &actions_batch, &mask, &mut context);
        assert_eq!(output.len(), BATCH_SIZE / 2);
    }
}
