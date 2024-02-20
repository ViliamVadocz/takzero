use fast_tak::{takparse::Move, Game};
use ordered_float::NotNan;
use tch::{
    nn::{self, ModuleT},
    Device,
    Tensor,
};

use super::{
    repr::{game_to_tensor, input_channels, move_index, output_channels},
    residual::ResidualBlock,
    EnsembleNetwork,
    Network,
};
use crate::{network::repr::output_size, search::agent::Agent};

pub const N: usize = 4;
pub const HALF_KOMI: i8 = 4;
pub type Env = Game<N, HALF_KOMI>;
const FILTERS: i64 = 256;
const ENSEMBLE_SIZE: usize = 16;

// Value is [-1, 1], which is size 2, so variance can be 2*2 = 4.
pub const MAXIMUM_VARIANCE: f64 = 4.0;

#[derive(Debug)]
pub struct Net {
    vs: nn::VarStore,
    core: nn::SequentialT,
    policy_net: nn::SequentialT,
    value_net: nn::SequentialT,
    ube_net: nn::SequentialT,
    ensemble: Ensemble,
}

#[derive(Debug)]
struct Ensemble([nn::SequentialT; ENSEMBLE_SIZE]);

fn core(path: &nn::Path) -> nn::SequentialT {
    const CORE_RES_BLOCKS: u32 = 20;
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
    nn::seq_t().add(nn::conv2d(
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
    nn::seq_t()
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
    nn::seq_t()
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
}

impl Network for Net {
    fn new(device: Device, seed: Option<i64>) -> Self {
        if let Some(seed) = seed {
            tch::manual_seed(seed);
        }

        let vs = nn::VarStore::new(device);
        let root = vs.root();
        Self {
            core: core(&(&root / "core")),
            policy_net: policy_net(&(&root / "policy")),
            value_net: value_net(&(&root / "value")),
            ube_net: ube_net(&(&root / "ube")),
            ensemble: Ensemble(std::array::from_fn(|i| {
                value_net(&(&root / format!("ensemble head {i}")))
            })),
            vs,
        }
    }

    fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }
}

impl EnsembleNetwork for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor, Tensor, Tensor) {
        let core = self.core.forward_t(xs, train);
        let policy = self.policy_net.forward_t(&core, train);
        let value = self.value_net.forward_t(&core, train);
        // Detached UBE so it does not mess with baseline
        let ube = self.ube_net.forward_t(&core.detach(), train);
        let ensemble = self.forward_ensemble(&core.detach(), train);
        (policy, value, ube, ensemble)
    }

    fn forward_ensemble(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        Tensor::concat(
            &self
                .ensemble
                .0
                .iter()
                .map(|nn| nn.forward_t(xs, train))
                .collect::<Vec<_>>(),
            1,
        )
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
                .map(|env| game_to_tensor(env, Device::Cpu))
                .collect::<Vec<_>>(),
            0,
        )
        .to(device);
        let (policy, values, ube_uncertainties, ensemble) = self.forward_t(&xs, false);
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
        let ensemble_variances = ensemble.var_dim(1i64, false, false);
        let uncertainties: Vec<_> = ube_uncertainties
            .exp() // Exponent because UBE prediction is log(variance)
            .maximum(&ensemble_variances)
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
    use tch::{Device, Tensor};

    use super::{Env, Net};
    use crate::{
        network::{net4_ensemble::ENSEMBLE_SIZE, repr::game_to_tensor, EnsembleNetwork, Network},
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
    fn batch_ensemble_size() {
        const BATCH_SIZE: usize = 128;
        let net = Net::new(Device::cuda_if_available(), Some(456));
        let games: [Env; BATCH_SIZE] = array::from_fn(|_| Game::default());
        let xs = Tensor::cat(
            &games
                .iter()
                .map(|env| game_to_tensor(env, Device::cuda_if_available()))
                .collect::<Vec<_>>(),
            0,
        );

        let ensemble = net.forward_t(&xs, false).3;
        assert_eq!(ensemble.size(), [BATCH_SIZE as i64, ENSEMBLE_SIZE as i64]);
    }
}
