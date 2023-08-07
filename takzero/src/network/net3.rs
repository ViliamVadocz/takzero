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

#[derive(Debug)]
pub struct Net3 {
    vs: nn::VarStore,
    core: nn::SequentialT,
    policy_head: nn::SequentialT,
    value_head: nn::SequentialT,
}

impl Network for Net3 {
    fn new(device: Device, seed: Option<i64>) -> Self {
        const FILTERS: i64 = 8;
        const CORE_RES_BLOCKS: u32 = 2;
        const N: usize = 3;

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

impl Agent<Game<3, 0>> for Net3 {
    type Policy = Policy;

    fn policy_value(&self, env: &Game<3, 0>, moves: &[Move]) -> (Self::Policy, f32) {
        let device = Device::cuda_if_available();

        let tensor = game_to_tensor(env, device);
        let (policy, value) = self.forward_t(&tensor, false);
        let masked_policy = policy
            .masked_fill(&move_mask::<3>(moves).to(device), 0.0)
            .softmax(1, Kind::Float)
            .view([output_size::<3>() as i64])
            .try_into()
            .unwrap();
        (Policy(masked_policy), value.try_into().unwrap())
    }
}

pub struct Policy(Vec<f32>);
impl Index<Move> for Policy {
    type Output = f32;

    fn index(&self, m: Move) -> &Self::Output {
        &self.0[move_index::<3>(&m)]
    }
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use tch::Device;

    use super::Net3;
    use crate::{network::Network, search::agent::Agent};

    #[test]
    fn evaluate() {
        let net = Net3::new(Device::cuda_if_available(), Some(123));
        let game: Game<3, 0> = Game::default();
        let mut moves = Vec::new();
        game.possible_moves(&mut moves);
        let (_policy, _value) = net.policy_value(&game, &moves);
    }
}
