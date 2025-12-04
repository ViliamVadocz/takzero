use bitvec::prelude::*;
use fast_tak::{takparse::Move, Game};
use ordered_float::NotNan;
use tch::{
    nn::{self, ModuleT},
    Device,
    Kind,
    TchError,
    Tensor,
};

use super::{
    repr::{game_to_tensor, input_channels, move_index, output_channels},
    residual::ResidualBlock,
    HashNetwork,
    Network,
};
use crate::{network::repr::output_size, search::agent::Agent};

pub const N: usize = 4;
pub const HALF_KOMI: i8 = 4;
pub type Env = Game<N, HALF_KOMI>;
const FILTERS: i64 = 256;
const HASH_BITS: usize = 32;

// Value is [-1, 1], which is size 2, so variance can be 2*2 = 4.
pub const MAXIMUM_VARIANCE: f64 = 4.0;

#[derive(Debug)]
pub struct Net {
    vs: nn::VarStore,
    core: nn::SequentialT,
    policy_net: nn::SequentialT,
    value_net: nn::SequentialT,
    ube_net: nn::SequentialT,
    lcghash_init: Tensor,
    lcghash_set: BitBox,
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
            lcghash_init: root.uniform(
                "lcghash_init",
                &[input_channels::<N>() as i64, N as i64, N as i64],
                -100.0,
                100.0,
            ),
            lcghash_set: bitbox![0; 1 << HASH_BITS],
            vs,
        }
    }

    fn vs(&self) -> &nn::VarStore {
        &self.vs
    }

    fn vs_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    #[allow(clippy::missing_errors_doc)]
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), TchError> {
        self.vs().save(&path)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(
                path.as_ref()
                    .parent()
                    .map(std::path::Path::to_path_buf)
                    .unwrap_or_default()
                    .join("bitvec.bin"),
            )?;
        std::io::copy(
            &mut bytemuck::cast_slice(self.lcghash_set.as_raw_slice()),
            &mut file,
        )?;
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    fn load(path: impl AsRef<std::path::Path>, device: Device) -> Result<Self, TchError> {
        let mut nn = Self::new(device, None);
        nn.vs_mut().load(&path)?;

        let mut file = std::fs::OpenOptions::new().read(true).open(
            path.as_ref()
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_default()
                .join("bitvec.bin"),
        )?;
        std::io::copy(
            &mut file,
            &mut bytemuck::cast_slice_mut::<_, u8>(nn.lcghash_set.as_raw_mut_slice()),
        )?;

        Ok(nn)
    }
}

impl HashNetwork<Env> for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor, Tensor) {
        let core = self.core.forward_t(xs, train);
        let policy = self.policy_net.forward_t(&core, train);
        let value = self.value_net.forward_t(&core, train);
        // Detached UBE so it does not mess with baseline
        let ube = self.ube_net.forward_t(&core.detach(), train);
        (policy, value, ube)
    }

    fn get_indices(&self, xs: &Tensor) -> Vec<usize> {
        // Hash using a linear congruential generator (LCG).
        // https://stackoverflow.com/a/77213071
        const MULTIPLIER: i64 = 6_364_136_223_846_793_005;
        const INCREMENT: i64 = 1;

        let options = (Kind::Int64, self.vs().device());

        let (batch_size, channels, rows, _cols) = xs.size4().unwrap();

        let xs = (xs * self.lcghash_init.detach())
            .view_dtype(Kind::Int)
            .split(1, 3);
        let mut acc = Tensor::zeros([batch_size, channels, rows], options);
        for x in xs {
            acc *= MULTIPLIER;
            acc += INCREMENT;
            acc += x.squeeze();
        }
        let xs = acc.split(1, 2);
        acc = Tensor::zeros([batch_size, channels], options);
        for x in xs {
            acc *= MULTIPLIER;
            acc += INCREMENT;
            acc += x.squeeze();
        }
        let xs = acc.split(1, 1);
        acc = Tensor::zeros([batch_size], options);
        for x in xs {
            acc *= MULTIPLIER;
            acc += INCREMENT;
            acc += x.squeeze();
        }

        let unshifted: Vec<i64> = acc.try_into().unwrap();
        #[allow(clippy::cast_sign_loss)]
        unshifted
            .into_iter()
            .map(|x| (x.abs() >> (63 - HASH_BITS)) as usize)
            .collect()
    }

    fn update_counts(&mut self, xs: &Tensor) {
        let indices = tch::no_grad(|| self.get_indices(xs));
        for index in indices {
            self.lcghash_set.set(index, true);
        }
    }

    fn forward_hash(&self, xs: &Tensor) -> Tensor {
        let indices = tch::no_grad(|| self.get_indices(xs));
        let counts: Vec<_> = indices
            .into_iter()
            .map(|index| {
                if self.lcghash_set[index] {
                    0.0
                } else {
                    MAXIMUM_VARIANCE
                }
            })
            .collect();
        Tensor::from_slice(&counts).to(self.vs().device())
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
        let local_uncertainties = self.forward_hash(&xs);
        let uncertainties: Vec<_> = ube_uncertainties
            .exp() // Exponent because UBE prediction is log(variance)
            .maximum(&local_uncertainties)
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
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use tch::{Device, Tensor};

    use super::{Env, Net, HALF_KOMI, N};
    use crate::{
        network::{repr::game_to_tensor, HashNetwork, Network},
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
        const SEED: u64 = 456;
        let mut rng = StdRng::seed_from_u64(SEED);
        let net = Net::new(Device::cuda_if_available(), Some(rng.random()));
        let mut games: [Env; BATCH_SIZE] =
            array::from_fn(|_| Game::new_opening_with_random_steps(&mut rng, &mut vec![], 10));
        let mut actions_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        games
            .iter_mut()
            .zip(&mut actions_batch)
            .for_each(|(game, actions)| game.populate_actions(actions));
        let output = net.policy_value_uncertainty(&games, &actions_batch);
        assert_eq!(output.count(), BATCH_SIZE);
    }

    #[test]
    fn counts_work() {
        const BATCH_SIZE: usize = 128;
        const SEED: u64 = 456;
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut net = Net::new(Device::cuda_if_available(), Some(rng.random()));

        let xs = Tensor::cat(
            &(0..BATCH_SIZE)
                .map(|_| {
                    game_to_tensor(
                        &Game::<N, HALF_KOMI>::new_opening_with_random_steps(
                            &mut rng,
                            &mut vec![],
                            5,
                        ),
                        Device::cuda_if_available(),
                    )
                })
                .collect::<Vec<_>>(),
            0,
        );

        let before = net.forward_hash(&xs);
        println!("{before}");
        net.update_counts(&xs);
        let after = net.forward_hash(&xs);
        println!("{after}");
        assert!(bool::try_from(before.gt_tensor(&after).all()).unwrap());
    }

    #[test]
    fn saving_works() {
        const BATCH_SIZE: usize = 128;
        const SEED: u64 = 456;
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut net = Net::new(Device::cuda_if_available(), Some(rng.random()));

        let xs = Tensor::cat(
            &(0..BATCH_SIZE)
                .map(|_| {
                    game_to_tensor(
                        &Game::<N, HALF_KOMI>::new_opening_with_random_steps(
                            &mut rng,
                            &mut vec![],
                            5,
                        ),
                        Device::cuda_if_available(),
                    )
                })
                .collect::<Vec<_>>(),
            0,
        );

        net.update_counts(&xs);
        net.save("delete-me.ot").unwrap();
        drop(net);

        let net = Net::load("delete-me.ot", Device::Cuda(0)).unwrap();
        let after = net.forward_hash(&xs);
        assert!(bool::try_from(after.eq(0.0).all()).unwrap());
    }
}
