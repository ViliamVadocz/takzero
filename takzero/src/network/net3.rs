use std::ops::Index;

use fast_tak::{takparse::Move, Game};
use tch::{
    nn::{self, ModuleT},
    Device,
    Kind,
    Tensor,
};

use super::{
    repr::{game_to_tensor, input_channels, move_index, output_channels, output_size},
    residual::ResidualBlock,
    Network,
};
use crate::{network::repr::move_mask, search::agent::Agent};

const N: usize = 3;

#[derive(Debug)]
pub struct Net3 {
    vs: nn::VarStore,
    core: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head: nn::SequentialT,
}

impl Network for Net3 {
    fn new(device: Device, seed: Option<i64>) -> Self {
        const FILTERS: i64 = 128;
        const CORE_RES_BLOCKS: u32 = 8;

        if let Some(seed) = seed {
            tch::manual_seed(seed);
        }

        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let mut core = nn::seq_t()
            .add(nn::conv2d(
                &root,
                input_channels::<N>() as i64,
                FILTERS,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add(nn::batch_norm2d(
                &root,
                FILTERS,
                nn::BatchNormConfig::default(),
            ))
            .add_fn(Tensor::relu);
        for _ in 0..CORE_RES_BLOCKS {
            core = core.add(ResidualBlock::new(&root, FILTERS, FILTERS));
        }

        let policy_head = nn::seq_t()
            .add(ResidualBlock::new(&root, FILTERS, FILTERS))
            .add(nn::conv2d(
                &root,
                FILTERS,
                output_channels::<N>() as i64,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
            ));

        let value_head = nn::seq_t()
            .add(ResidualBlock::new(&root, FILTERS, FILTERS))
            .add(nn::conv2d(&root, FILTERS, 1, 1, nn::ConvConfig {
                stride: 1,
                ..Default::default()
            }))
            .add_fn(Tensor::relu)
            .add_fn(|x| x.view([-1, (N * N) as i64]))
            .add(nn::linear(
                &root,
                (N * N) as i64,
                1,
                nn::LinearConfig::default(),
            ))
            .add_fn(Tensor::tanh);

        Self {
            vs,
            core,
            policy_head,
            value_head,
        }
    }

    fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let s = self.core.forward_t(xs, train);
        (
            self.policy_head.forward_t(&s, train),
            self.value_head.forward_t(&s, train),
        )
    }
}

type Env = Game<N, 0>;

// TODO: gather policy from tensor on gpu (i.e. pass moves and select the policy
// instead of masking)?
// - https://pytorch.org/docs/stable/generated/torch.gather.html
// - https://pytorch.org/docs/stable/generated/torch.index_select.html
impl Agent<Env> for Net3 {
    type Policy = Policy;

    fn policy_value(
        &self,
        env_batch: &[Env],
        actions_batch: &[Vec<Move>],
    ) -> Vec<(Self::Policy, f32)> {
        debug_assert_eq!(env_batch.len(), actions_batch.len());
        if env_batch.is_empty() {
            return Vec::new();
        }
        let device = self.vs.device();

        let xs = Tensor::cat(
            &env_batch
                .iter()
                .map(|env| game_to_tensor(env, device))
                .collect::<Vec<_>>(),
            0,
        );
        let mask = Tensor::cat(
            &actions_batch
                .iter()
                .map(|m| move_mask::<N>(m, device))
                .collect::<Vec<_>>(),
            0,
        );

        let (policy, values) = self.forward_t(&xs, false);
        let masked_policy: Vec<Vec<_>> = policy
            .masked_fill(&mask, f64::from(f32::MIN))
            .view([-1, output_size::<3>() as i64])
            .softmax(1, Kind::Float)
            .try_into()
            .unwrap();
        let values: Vec<_> = values.view([-1]).try_into().unwrap();

        masked_policy.into_iter().map(Policy).zip(values).collect()
    }
}

pub struct Policy(Vec<f32>);

impl Index<Move> for Policy {
    type Output = f32;

    fn index(&self, index: Move) -> &Self::Output {
        &self.0[move_index::<N>(&index)]
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use fast_tak::Game;
    use tch::Device;

    use super::Net3;
    use crate::{
        network::Network,
        search::{agent::Agent, env::Environment},
    };

    #[test]
    fn evaluate() {
        let net = Net3::new(Device::cuda_if_available(), Some(123));
        let game: Game<3, 0> = Game::default();
        let mut moves = Vec::new();
        game.possible_moves(&mut moves);
        let (_policy, _value) = net.policy_value(&[game], &[moves]).pop().unwrap();
    }

    #[test]
    fn evaluate_batch() {
        const BATCH_SIZE: usize = 128;
        let net = Net3::new(Device::cuda_if_available(), Some(456));
        let mut games: [Game<3, 0>; BATCH_SIZE] = array::from_fn(|_| Game::default());
        let mut actions_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        games
            .iter_mut()
            .zip(&mut actions_batch)
            .for_each(|(game, actions)| game.populate_actions(actions));
        let output = net.policy_value(&games, &actions_batch);
        assert_eq!(output.len(), BATCH_SIZE);
    }
}
