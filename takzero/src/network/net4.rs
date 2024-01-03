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

pub const N: usize = 4;
pub const HALF_KOMI: i8 = 4;
pub type Env = Game<N, HALF_KOMI>;
const FILTERS: i64 = 128;

#[derive(Debug)]
pub struct Net {
    vs: nn::VarStore,
    policy_net: nn::SequentialT,
    value_net: nn::SequentialT,
    ube_net: nn::SequentialT,
    rnd: Rnd,
}

#[derive(Debug)]
struct Rnd {
    target: nn::SequentialT,
    learning: nn::SequentialT,
}

fn core(path: &nn::Path) -> nn::SequentialT {
    const CORE_RES_BLOCKS: u32 = 16;
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

fn policy_net(path: &nn::Path) -> nn::SequentialT {
    core(&(path / "core")).add(nn::conv2d(
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

fn value_net(path: &nn::Path) -> nn::SequentialT {
    core(&(path / "core"))
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

fn ube_net(path: &nn::Path) -> nn::SequentialT {
    core(&(path / "core"))
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
        .add_fn(Tensor::exp)
}

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

impl Network for Net {
    fn new(device: Device, seed: Option<i64>) -> Self {
        if let Some(seed) = seed {
            tch::manual_seed(seed);
        }

        let vs = nn::VarStore::new(device);
        let root = vs.root();
        Self {
            policy_net: policy_net(&(&root / "policy")),
            value_net: value_net(&(&root / "value")),
            ube_net: ube_net(&(&root / "ube")),
            rnd: Rnd {
                learning: rnd(&(&root / "rnd_learning")),
                target: rnd(&(&root / "rnd_target")),
            },
            vs,
        }
    }

    fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor, Tensor) {
        (
            self.policy_net.forward_t(xs, train),
            self.value_net.forward_t(xs, train),
            self.ube_net.forward_t(xs, train),
        )
    }

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

impl Agent<Env> for Net {
    type Context = RndNormalizationContext;
    type Policy = Policy;

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

        let (policy, values, ube_uncertainties) = self.forward_t(&xs, false);
        let masked_policy: Vec<Vec<_>> = policy
            .masked_fill(&move_mask, f64::from(f32::MIN))
            .view([-1, output_size::<N>() as i64])
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

#[cfg(test)]
mod tests {
    use std::array;

    use fast_tak::Game;
    use tch::Device;

    use super::{Env, Net, RndNormalizationContext};
    use crate::{
        network::Network,
        search::{agent::Agent, env::Environment},
    };

    #[test]
    fn evaluate() {
        let net = Net::new(Device::cuda_if_available(), Some(123));
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
        let net = Net::new(Device::cuda_if_available(), Some(456));
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
