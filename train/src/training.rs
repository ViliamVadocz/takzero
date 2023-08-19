use std::{path::Path, sync::atomic::Ordering};

use crossbeam::channel::Receiver;
use takzero::{
    fast_tak::{Game, Reserves},
    network::{
        repr::{game_to_tensor, move_mask, policy_tensor},
        Network,
    },
    search::agent::Agent,
};
use tch::{
    nn::{Adam, OptimizerConfig},
    Device,
    Kind,
    Reduction,
    Tensor,
};

use crate::{file_name, target::Target, BetaNet};

const WEIGHT_DECAY: f64 = 1e-4;
const LEARNING_RATE: f64 = 1e-3;
const BATCHES_PER_STEP: u64 = 16;
const STEPS_BETWEEN_PUBLISH: u64 = 2;
const PUBLISHES_BETWEEN_SAVE: u64 = 10;

// TODO: Consider learning rate scheduler: https://pytorch.org/docs/stable/optim.html

/// Improve the network by training on batches from the re-analyze thread.
/// Save checkpoints and distribute the newest model.
#[allow(clippy::needless_pass_by_value)]
pub fn run<const N: usize, const HALF_KOMI: i8, NET: Network + Agent<Game<N, HALF_KOMI>>>(
    device: Device,
    beta_net: &BetaNet,
    rx: Receiver<Vec<Target<Game<N, HALF_KOMI>>>>,
    model_path: &Path,
) where
    Reserves<N>: Default,
{
    let mut alpha_net = NET::new(device, None);
    alpha_net
        .vs_mut()
        .copy(&beta_net.1.read().unwrap())
        .unwrap();

    let mut opt = Adam {
        wd: WEIGHT_DECAY,
        ..Default::default()
    }
    .build(alpha_net.vs_mut(), LEARNING_RATE)
    .unwrap();

    let mut training_steps = 0;
    let mut batches = 0;

    while let Ok(batch) = rx.recv() {
        let batch_size = batch.len();

        let mut inputs = Vec::new();
        let mut policy_targets = Vec::new();
        let mut value_targets = Vec::new();
        let mut masks = Vec::new();
        for target in batch {
            inputs.push(game_to_tensor(&target.env, device));
            policy_targets.push(policy_tensor::<N>(&target.policy, device));
            value_targets.push(target.value);
            masks.push(move_mask::<N>(
                &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
                device,
            )); // FIXME: efficiency
        }

        // Get network output.
        let input = Tensor::cat(&inputs, 0);
        let mask = Tensor::cat(&masks, 0);
        let (policy, values) = alpha_net.forward_t(&input, true);
        let policy = policy.masked_fill(&mask, 0.0).log_softmax(1, Kind::Float);

        // Get the target.
        let p = Tensor::stack(&policy_targets, 0).view(policy.size().as_slice());
        let z = Tensor::from_slice(&value_targets).unsqueeze(1).to(device);

        // Calculate loss.
        let loss_p = policy.kl_div(&p, Reduction::Sum, false) / i64::try_from(batch_size).unwrap();
        let loss_z = (z - values).square().mean(Kind::Float);
        println!("p={loss_p:?}\t z={loss_z:?}");
        let total_loss = loss_z + loss_p;

        // Do multiple backwards batches before making a step.
        total_loss.backward();
        batches += 1;
        if batches % BATCHES_PER_STEP == 0 {
            println!("Taking step!");
            opt.step();
            opt.zero_grad();
            training_steps += 1;

            if training_steps % STEPS_BETWEEN_PUBLISH == 0 {
                beta_net.1.write().unwrap().copy(alpha_net.vs()).unwrap();
                beta_net.0.fetch_add(1, Ordering::Relaxed);
                // Save checkpoint.
                if (training_steps / STEPS_BETWEEN_PUBLISH) % PUBLISHES_BETWEEN_SAVE == 0 {
                    // FIXME: This will stall until write is complete, which might be a long time
                    // because we are writing to a different computer.
                    alpha_net
                        .save(model_path.join(file_name(
                            training_steps / STEPS_BETWEEN_PUBLISH / PUBLISHES_BETWEEN_SAVE,
                        )))
                        .unwrap();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::{atomic::AtomicUsize, RwLock},
    };

    use rand::{Rng, SeedableRng};
    use takzero::{
        fast_tak::Game,
        network::{net3::Net3, Network},
    };
    use tch::Device;

    use super::run;
    use crate::{target::Target, BetaNet};

    #[test]
    fn training_works() {
        const SEED: u64 = 1234;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net3::new(Device::Cpu, Some(rng.gen()));
        let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

        let (batch_tx, batch_rx) = crossbeam::channel::unbounded();

        // Send a mock batch.
        batch_tx
            .send(vec![
                Target {
                    env: Game::default(),
                    policy: vec![("a1".parse().unwrap(), 0.2), ("a2".parse().unwrap(), 0.8)]
                        .into_boxed_slice(),
                    value: -0.4,
                },
                Target {
                    env: Game::default(),
                    policy: vec![("a3".parse().unwrap(), 0.4), ("a4".parse().unwrap(), 0.6)]
                        .into_boxed_slice(),
                    value: 0.3,
                },
            ])
            .unwrap();
        drop(batch_tx);

        run::<3, 0, Net3>(
            Device::cuda_if_available(),
            &beta_net,
            batch_rx,
            &PathBuf::default(),
        );
    }
}
