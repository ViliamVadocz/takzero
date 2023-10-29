use std::{
    path::Path,
    sync::{atomic::{AtomicU32, Ordering}, RwLock}, collections::VecDeque,
};

use crossbeam::channel::Receiver;
use takzero::{
    network::{
        repr::{game_to_tensor, move_mask, output_size, policy_tensor},
        Network,
    },
    target::{Target, Augment},
};
use tch::{
    nn::{Adam, OptimizerConfig},
    Device,
    Kind,
    Reduction,
    Tensor,
};
use rand::prelude::*;

use crate::{file_name, reanalyze, Env, Net, SharedNet, N};

pub const LEARNING_RATE: f64 = 1e-4;
pub const EFFECTIVE_BATCH_SIZE: usize = 512;
pub const EXPLOITATION_PARTS: usize = 3;
pub const STEPS_BETWEEN_PUBLISH: u32 = 100;
pub const PUBLISHES_BETWEEN_SAVE: u32 = 10;

const BATCH_SIZE: usize = reanalyze::BATCH_SIZE * (1 + EXPLOITATION_PARTS); 
#[allow(clippy::assertions_on_constants)]
const _: () = assert!(
    EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0,
    "EFFECTIVE_BATCH_SIZE should be divisible by training::BATCH_SIZE"
);
const BATCHES_PER_STEP: usize = EFFECTIVE_BATCH_SIZE.div_ceil(BATCH_SIZE);

// TODO: Consider learning rate scheduler: https://pytorch.org/docs/stable/optim.html

/// Improve the network by training on batches from the re-analyze thread.
/// Save checkpoints and distribute the newest model.
#[allow(clippy::needless_pass_by_value)]
pub fn run(
    device: Device,
    seed: u64,
    shared_net: &SharedNet,
    rx: Receiver<Vec<Target<Env>>>,
    exploitation_buffer: &RwLock<VecDeque<Target<Env>>>,
    training_steps: &AtomicU32,
    model_path: &Path,
) {
    log::debug!("started training thread");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(device, None);
    net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
    net.vs_mut().unfreeze();

    let mut latest_rnd_losses: Vec<f64> = Vec::new();

    let mut opt = Adam::default().build(net.vs_mut(), LEARNING_RATE).unwrap();
    let mut batches = 0;
    let mut accumulated_total_loss = Tensor::zeros([1], (Kind::Float, device));
    while let Ok(mut batch) = rx.recv() { 
        // Sample some targets from the exploitation buffer.
        batch.extend(
            exploitation_buffer.read().unwrap()
            .iter()
            .choose_multiple(&mut rng, reanalyze::BATCH_SIZE * EXPLOITATION_PARTS)
            .into_iter()
            .map(|target| target.augment(&mut rng))
        );
        assert!(batch.iter().all(|target| target.value.is_finite()));
        assert!(batch.iter().all(|target| target.ube.is_finite()));
        let batch_size = batch.len();
        log::info!("batch size is {batch_size}");

        let mut inputs = Vec::with_capacity(batch_size);
        let mut value_targets = Vec::with_capacity(batch_size);
        let mut ube_targets = Vec::with_capacity(batch_size);
        let mut policy_targets = Vec::with_capacity(batch_size);
        let mut masks = Vec::with_capacity(batch_size);
        for target in batch {
            inputs.push(game_to_tensor(&target.env, device));
            value_targets.push(target.value);
            ube_targets.push(target.ube);
            policy_targets.push(policy_tensor::<N>(&target.policy, device));
            masks.push(move_mask::<N>(
                &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
                device,
            ));
        }

        // Get network output.
        let input = Tensor::cat(&inputs, 0);
        let mask = Tensor::cat(&masks, 0);
        let (policy, values, ube_uncertainty) = net.forward_t(&input, true);
        #[allow(clippy::cast_possible_wrap)]
        let policy = policy
            .masked_fill(&mask, f64::from(f32::MIN))
            .view([-1, output_size::<N>() as i64])
            .log_softmax(1, Kind::Float);

        // Get the target.
        let p = Tensor::stack(&policy_targets, 0).view(policy.size().as_slice());
        let z = Tensor::from_slice(&value_targets).unsqueeze(1).to(device);
        let u = Tensor::from_slice(&ube_targets).unsqueeze(1).to(device);

        // Calculate loss.
        let loss_p = policy.kl_div(&p, Reduction::Sum, false) / i64::try_from(batch_size).unwrap();
        let loss_z = (z - values).square().mean(Kind::Float);
        let loss_u = (u - ube_uncertainty).square().mean(Kind::Float);
        log::info!("p={loss_p:?}\t z={loss_z:?}\t u={loss_u:?}"); // FIXME: This forces synchronization!
        accumulated_total_loss += loss_z + loss_p + loss_u;

        // RND
        let loss_rnd = net.forward_rnd(&input, true).mean(Kind::Float);
        log::info!("rnd={loss_rnd:?}");
        latest_rnd_losses.push((&loss_rnd).try_into().unwrap());
        accumulated_total_loss += loss_rnd;

        // Do multiple backwards batches before making a step.
        batches += 1;
        #[allow(clippy::modulo_one)]
        if batches % BATCHES_PER_STEP == 0 {
            log::info!("taking step, accumulated_loss = {accumulated_total_loss:?}"); // FIXME: Forces synchronization
            opt.backward_step(&(accumulated_total_loss / i64::try_from(BATCHES_PER_STEP).unwrap()));
            accumulated_total_loss = Tensor::zeros([1], (Kind::Float, device));
            let training_steps = 1 + training_steps.fetch_add(1, Ordering::Relaxed);

            #[allow(clippy::modulo_one)]
            if training_steps % STEPS_BETWEEN_PUBLISH == 0 {
                {
                    let mut lock = shared_net.1.write().unwrap();
                    lock.copy(net.vs()).unwrap();
                    lock.freeze();
                }
                shared_net.0.fetch_add(1, Ordering::Relaxed);
                let mean_rnd_loss_over_batch = {
                    let len = latest_rnd_losses.len();
                    latest_rnd_losses.drain(..).sum::<f64>() / f64::from(u32::try_from(len).unwrap())
                };
                *shared_net.2.write().unwrap() = mean_rnd_loss_over_batch;
                // Save checkpoint.
                if (training_steps / STEPS_BETWEEN_PUBLISH) % PUBLISHES_BETWEEN_SAVE == 0 {
                    // FIXME: This will stall until write is complete, which might be a long time
                    // because we are writing to a different computer.
                    let path = model_path.join(file_name(training_steps));
                    log::info!("Saving model at {}, with rnd_loss = {mean_rnd_loss_over_batch}", path.display());
                    net.save(path).unwrap();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::{
            atomic::{AtomicU32, AtomicUsize},
            RwLock,
        }, collections::VecDeque,
    };

    use fast_tak::Game;
    use rand::{Rng, SeedableRng};
    use takzero::{network::Network, target::Target};
    use tch::Device;

    use super::run;
    use crate::{Net, SharedNet};

    #[test]
    fn training_works() {
        const SEED: u64 = 1234;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net::new(Device::Cpu, Some(rng.gen()));
        let shared_net: SharedNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()), RwLock::new(0.0));

        let (batch_tx, batch_rx) = crossbeam::channel::unbounded();

        // Send a mock batch.
        batch_tx
            .send(vec![
                Target {
                    env: Game::default(),
                    policy: vec![("a1".parse().unwrap(), 0.2), ("a2".parse().unwrap(), 0.8)]
                        .into_boxed_slice(),
                    value: -0.4,
                    ube: 1.0,
                },
                Target {
                    env: Game::default(),
                    policy: vec![("a3".parse().unwrap(), 0.4), ("a4".parse().unwrap(), 0.6)]
                        .into_boxed_slice(),
                    value: 0.3,
                    ube: 0.05,
                },
            ])
            .unwrap();
        drop(batch_tx);

        run(
            Device::cuda_if_available(),
            123,
            &shared_net,
            batch_rx,
            &RwLock::new(VecDeque::new()),
            &AtomicU32::new(0),
            &PathBuf::default(),
        );
    }
}
