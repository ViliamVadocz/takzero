use std::{collections::HashSet, fmt::Write as FmtWrite, fs::OpenOptions, io::Write};

use rand::{seq::SliceRandom, Rng, SeedableRng};
use takzero::{
    network::{net4_lcghash::Net, repr::game_to_tensor, HashNetwork, Network},
    search::env::Environment,
    target::get_replays,
};
use tch::{Device, Tensor};
use utils::reference_batches;

const STEPS: usize = 200_000;
const BATCH_SIZE: usize = 128;
const DEVICE: Device = Device::Cuda(0);
const FORCED_USES: u32 = 4;
const PERIOD: usize = 100;

mod utils;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(432);

    let mut net = Net::new(DEVICE, Some(rng.gen()));

    let mut positions = Vec::new();
    let mut seen_positions = HashSet::new();
    let mut unique_positions = Vec::new();
    for replay in get_replays("_replays/undirected_01_replay.txt")
        .unwrap()
        .take(STEPS * BATCH_SIZE / 10)
    {
        let mut env = replay.env;
        for action in replay.actions {
            positions.push(env.clone());
            if seen_positions.insert(env.clone().canonical()) {
                unique_positions.push(env.clone());
            }
            env.step(action);
        }
    }

    let (
        (early_game, early_tensor),
        (late_game, late_tensor),
        (random_early_batch, random_early_tensor),
        (random_late_batch, random_late_tensor),
        impossible_early_tensor,
    ) = reference_batches(&unique_positions, &mut rng, BATCH_SIZE);

    for (game, hash) in early_game.iter().zip(net.get_indices(&early_tensor)) {
        let tps: fast_tak::takparse::Tps = game.clone().into();
        println!("{tps} \t:\t {hash}");
    }

    let mut buffer = Vec::with_capacity(2048);
    let mut positions = positions.into_iter().filter(|s| {
        let c = s.clone().canonical();
        !early_game.contains(&c)
            && !late_game.contains(&c)
            && !random_early_batch.contains(&c)
            && !random_late_batch.contains(&c)
    });

    let mut current: Vec<f64> = Vec::new();
    let mut after: Vec<f64> = Vec::new();
    let mut early_losses: Vec<f64> = Vec::new();
    let mut late_losses: Vec<f64> = Vec::new();
    let mut random_early_losses: Vec<f64> = Vec::new();
    let mut random_late_losses: Vec<f64> = Vec::new();
    let mut impossible_losses: Vec<f64> = Vec::new();

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("eee_data.csv")
        .unwrap();

    for step in 0..=STEPS {
        if step % PERIOD == 0 {
            println!("step: {step: >8}");
            let content = current
                .drain(..)
                .zip(after.drain(..))
                .zip(early_losses.drain(..))
                .zip(late_losses.drain(..))
                .zip(random_early_losses.drain(..))
                .zip(random_late_losses.drain(..))
                .zip(impossible_losses.drain(..))
                .enumerate()
                .fold(
                    String::new(),
                    |mut s,
                     (
                        i,
                        (
                            (((((current, after), early), late), random_early), random_late),
                            impossible_early,
                        ),
                    )| {
                        writeln!(
                            &mut s,
                            "{},{current},{after},{early},{late},{random_early},{random_late},\
                             {impossible_early}",
                            step + i - PERIOD
                        )
                        .unwrap();
                        s
                    },
                );
            file.write_all(content.as_bytes()).unwrap();
        }

        // Add replays to buffer until we have enough.
        while buffer.len() < 1024 {
            let position = positions.next().unwrap();
            buffer.push((position, FORCED_USES));
        }

        // Sample a batch.
        buffer.shuffle(&mut rng);
        let batch = buffer.split_off(buffer.len() - BATCH_SIZE);
        let tensor = Tensor::concat(
            &batch
                .iter()
                .map(|(env, _)| game_to_tensor(env, Device::Cpu))
                .collect::<Vec<_>>(),
            0,
        )
        .to(DEVICE);
        buffer.extend(
            batch
                .into_iter()
                .filter(|(_, x)| *x <= 1)
                .map(|(s, x)| (s, x - 1)),
        );

        let var = net.forward_hash(&early_tensor).mean(None);
        early_losses.push(var.try_into().unwrap());
        let var = net.forward_hash(&late_tensor).mean(None);
        late_losses.push(var.try_into().unwrap());
        let var = net.forward_hash(&random_early_tensor).mean(None);
        random_early_losses.push(var.try_into().unwrap());
        let var = net.forward_hash(&random_late_tensor).mean(None);
        random_late_losses.push(var.try_into().unwrap());
        let var = net.forward_hash(&impossible_early_tensor).mean(None);
        impossible_losses.push(var.try_into().unwrap());

        let var = net.forward_hash(&tensor).mean(None);
        current.push(var.try_into().unwrap());

        net.update_counts(&tensor);

        let var = net.forward_hash(&tensor).mean(None);
        after.push(var.try_into().unwrap());
    }

    println!("Done.");
}
