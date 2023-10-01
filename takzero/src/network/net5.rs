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
    ube_head: nn::SequentialT,
    rnd: Rnd,
}

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
        let ube_head = ube_head(&(&root / "ube"));
        let rnd = Rnd {
            learning: rnd(&(&root / "rnd_learning")),
            target: rnd(&(&root / "rnd_target")),
        };

        Self {
            vs,
            core,
            policy_head,
            value_head,
            ube_head,
            rnd,
        }
    }

    fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor, Tensor) {
        let s = self.core.forward_t(xs, train);
        (
            self.policy_head.forward_t(&s, train),
            self.value_head.forward_t(&s, train),
            self.ube_head.forward_t(&s, train),
        )
    }

    fn forward_rnd(&self, xs: &Tensor, train: bool) -> Tensor {
        self.rnd.learning.forward_t(xs, train).mse_loss(
            &self
                .rnd
                .target
                .forward_t(&xs.set_requires_grad(false), false)
                .detach(),
            Reduction::None,
        )
    }
}

type Env = Game<N, 4>;

impl Agent<Env> for Net5 {
    type Context = RndNormalizationContext;
    type Policy = Policy;

    fn policy_value_uncertainty(
        &self,
        env_batch: &[Env],
        actions_batch: &[Vec<<Env as crate::search::env::Environment>::Action>],
        context: &mut Self::Context,
    ) -> Vec<(Self::Policy, f32, f32)> {
        debug_assert_eq!(env_batch.len(), actions_batch.len());
        if env_batch.is_empty() {
            return Vec::new();
        }
        let device = self.vs.device();

        let xs = Tensor::cat(
            &env_batch
                .par_iter()
                .map(|env| game_to_tensor(env, device))
                .collect::<Vec<_>>(),
            0,
        );
        let mask = Tensor::cat(
            &actions_batch
                .par_iter()
                .map(|m| move_mask::<N>(m, device))
                .collect::<Vec<_>>(),
            0,
        );

        let (policy, values, ube_uncertainties) = self.forward_t(&xs, false);
        let masked_policy: Vec<Vec<_>> = policy
            .masked_fill(&mask, f64::from(f32::MIN))
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
            .map(|((p, v), u)| (p, v, u))
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
    min: Tensor,
    max: Tensor,
}

impl RndNormalizationContext {
    fn new(batch_size: i64, device: Device) -> Self {
        let mut this = Self {
            min: Tensor::zeros([batch_size], (Kind::Float, device)),
            max: Tensor::zeros([batch_size], (Kind::Float, device)),
        };
        this.reset();
        this
    }

    fn reset(&mut self) {
        self.min = self.min.fill(f64::from(f32::MAX));
        self.max = self.max.fill(f64::from(f32::MIN));
    }

    fn update(&mut self, rnd: &Tensor) {
        self.min = self.min.minimum(rnd);
        self.max = self.max.maximum(rnd);
    }

    pub fn normalize(&mut self, rnd: &Tensor) -> Tensor {
        self.update(rnd);
        let diff = &self.max - &self.min;
        let mask = diff.gt(0.0);
        let norm = (rnd - &self.min) / diff;
        norm * &mask + rnd * mask.logical_not()
    }
}

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
        let mut context = RndNormalizationContext::new(1, Device::cuda_if_available());
        game.possible_moves(&mut moves);
        let (_policy, _value, _uncertainty) = net
            .policy_value_uncertainty(&[game], &[moves], &mut context)
            .pop()
            .unwrap();
    }

    #[test]
    fn evaluate_batch() {
        const BATCH_SIZE: usize = 128;
        let net = Net5::new(Device::cuda_if_available(), Some(456));
        let mut games: [Env; BATCH_SIZE] = array::from_fn(|_| Game::default());
        let mut actions_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        let mut context =
            RndNormalizationContext::new(BATCH_SIZE as i64, Device::cuda_if_available());
        games
            .iter_mut()
            .zip(&mut actions_batch)
            .for_each(|(game, actions)| game.populate_actions(actions));
        let output = net.policy_value_uncertainty(&games, &actions_batch, &mut context);
        assert_eq!(output.len(), BATCH_SIZE);
    }
}
