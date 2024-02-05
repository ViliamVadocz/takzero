use fast_tak::{takparse::Move, Game};
use ordered_float::NotNan;
use tch::{
    nn::{self, ModuleT},
    Device,
    Tensor,
};

use super::{
    repr::{game_to_tensor, input_channels, input_size, move_index, output_channels},
    residual::ResidualBlock,
    Network,
};
use crate::{
    network::repr::output_size,
    search::{agent::Agent, SERIES_DISCOUNT},
};

pub const N: usize = 4;
pub const HALF_KOMI: i8 = 4;
pub type Env = Game<N, HALF_KOMI>;
const FILTERS: i64 = 256;

// Value is [-1, 1], which is size 2, so variance can be 2*2 = 4.
pub const MAXIMUM_VARIANCE: f64 = 4.0;

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
    // Normalization variables
    min: Tensor,
    max: Tensor,
}

fn core(path: &nn::Path) -> nn::SequentialT {
    const CORE_RES_BLOCKS: u32 = 10;
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
    const RND_INPUT_SCALE: f64 = 50.0;
    nn::seq_t()
        .add_fn(|x| x.view([-1, input_size::<N>() as i64]))
        .add_fn(|x| RND_INPUT_SCALE * x / x.square().sum_dim_intlist(1, true, None))
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
                min: root.var("min", &[1], nn::Init::Const(0.0)),
                // TODO: Think about a good default
                max: root.var("max", &[1], nn::Init::Const(1.0)),
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
        let policy = self.policy_net.forward_t(xs, train);
        let value = self.value_net.forward_t(xs, train);
        let ube = self.ube_net.forward_t(xs, train);
        (policy, value, ube)
    }

    fn forward_rnd(&self, xs: &Tensor, train: bool) -> Tensor {
        let learning = self
            .rnd
            .learning
            .forward_t(&xs.set_requires_grad(false), train);
        let target = self
            .rnd
            .target
            .forward_t(&xs.set_requires_grad(false), false)
            .detach();
        (learning - target).square().sum_dim_intlist(1, false, None)
    }

    fn normalized_rnd(&self, xs: &Tensor) -> Tensor {
        let min = self.rnd.min.detach();
        let max = self.rnd.max.detach();
        let normalized = (self.forward_rnd(xs, false) - &min) / (max - min);
        normalized.clamp(0.0, 1.0) * MAXIMUM_VARIANCE
    }

    fn update_rnd_normalization(&mut self, min: &Tensor, max: &Tensor) {
        log::debug!("Updating RND normalization to min: {min:?} and max: {max:?}");
        self.rnd.min.set_data(min);
        self.rnd.max.set_data(max);
    }
}

impl Agent<Env> for Net {
    fn policy_value_uncertainty(
        &self,
        env_batch: &[Env],
        actions_batch: &[Vec<<Env as crate::search::env::Environment>::Action>],
    ) -> impl Iterator<Item = (Vec<(Move, NotNan<f32>)>, f32, f32)> {
        assert_eq!(env_batch.len(), actions_batch.len());
        assert!(!env_batch.is_empty());
        let device = self.vs.device();

        let xs = Tensor::cat(
            &env_batch
                .iter()
                .map(|env| game_to_tensor(env, device))
                .collect::<Vec<_>>(),
            0,
        );
        let (policy, values, ube_uncertainties) = self.forward_t(&xs, false);
        let policy = policy.view([-1, output_size::<N>() as i64]);
        let max_actions = actions_batch.iter().map(Vec::len).max().unwrap_or_default();
        let index = Tensor::from_slice2(
            &actions_batch
                .iter()
                .map(|actions| {
                    actions
                        .iter()
                        .map(|a| move_index::<N>(a) as i64)
                        .chain(std::iter::repeat(0))
                        .take(max_actions)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
        .to(device);

        let indexed_policy = actions_batch
            .iter()
            .zip(
                Vec::<Vec<_>>::try_from(policy.gather(1, &index, false))
                    .expect("tensor should have two dimensions"),
            )
            .map(|(actions, p)| {
                actions
                    .iter()
                    .zip(p)
                    .map(|(a, p)| (*a, NotNan::new(p).expect("logit should not be NaN")))
                    .collect()
            });
        let values: Vec<_> = values.view([-1]).try_into().unwrap();

        // Uncertainty.
        let rnd_uncertainties = self.normalized_rnd(&xs);
        let uncertainties: Vec<_> = ube_uncertainties
            .maximum(&(SERIES_DISCOUNT * rnd_uncertainties))
            .clamp(0.0, MAXIMUM_VARIANCE)
            .view([-1])
            .try_into()
            .unwrap();

        indexed_policy
            .zip(values)
            .zip(uncertainties)
            .map(|((p, v), u)| (p, v, u))
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use fast_tak::Game;
    use tch::Device;

    use super::{Env, Net};
    use crate::{
        network::Network,
        search::{agent::Agent, env::Environment},
    };

    #[test]
    fn evaluate() {
        let net = Net::new(Device::cuda_if_available(), Some(123));
        let game: Env = Game::default();
        let mut moves = Vec::new();
        game.possible_moves(&mut moves);
        let (_policy, _value, _uncertainty) = net
            .policy_value_uncertainty(&[game], &[moves])
            .next()
            .unwrap();
    }

    #[test]
    fn evaluate_batch() {
        const BATCH_SIZE: usize = 128;
        let net = Net::new(Device::cuda_if_available(), Some(456));
        let mut games: [Env; BATCH_SIZE] = array::from_fn(|_| Game::default());
        let mut actions_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        games
            .iter_mut()
            .zip(&mut actions_batch)
            .for_each(|(game, actions)| game.populate_actions(actions));
        let output = net.policy_value_uncertainty(&games, &actions_batch);
        assert_eq!(output.count(), BATCH_SIZE);
    }

    #[test]
    fn update_rnd_persistance() {
        const NEW_MIN: f32 = 123.456;
        const NEW_MAX: f32 = 789.987;

        let mut net = Net::new(Device::cuda_if_available(), Some(456));
        println!("init: {:?} {:?}", net.rnd.min, net.rnd.max);
        assert!(f32::try_from(&net.rnd.min).unwrap().abs() < f32::EPSILON);
        assert!((f32::try_from(&net.rnd.max).unwrap() - 1.0).abs() < f32::EPSILON);

        net.update_rnd_normalization(&NEW_MIN.into(), &NEW_MAX.into());
        println!("set: {:?} {:?}", net.rnd.min, net.rnd.max);
        assert!((f32::try_from(&net.rnd.min).unwrap() - NEW_MIN).abs() < f32::EPSILON);
        assert!((f32::try_from(&net.rnd.max).unwrap() - NEW_MAX).abs() < f32::EPSILON);

        net.save("temp-remove-me.ot").unwrap();
        drop(net);

        let net = Net::load("temp-remove-me.ot", Device::cuda_if_available()).unwrap();
        assert!((f32::try_from(&net.rnd.min).unwrap() - NEW_MIN).abs() < f32::EPSILON);
        assert!((f32::try_from(&net.rnd.max).unwrap() - NEW_MAX).abs() < f32::EPSILON);
    }
}
