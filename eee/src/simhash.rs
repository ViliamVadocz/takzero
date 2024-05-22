use std::{collections::HashSet, fmt::Write as FmtWrite, fs::OpenOptions, io::Write};

use fast_tak::takparse::Move;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
    SeedableRng,
};
use takzero::{
    network::{
        net4_simhash::{Env, Net, N},
        repr::{game_to_tensor, input_channels},
        HashNetwork,
        Network,
    },
    search::env::Environment,
    target::get_replays,
};
use tch::{Device, Tensor};

const STEPS: usize = 45_000;
const BATCH_SIZE: usize = 256;
const DEVICE: Device = Device::Cuda(0);
const FORCED_USES: u32 = 4;

fn random_env(ply: usize, actions: &mut Vec<Move>, rng: &mut impl Rng) -> Env {
    let mut env = Env::default();
    for _ in 0..ply {
        env.populate_actions(actions);
        let Some(action) = actions.drain(..).choose(rng) else {
            break;
        };
        env.step(action);
    }
    env
}

fn reference_envs(ply: usize, actions: &mut Vec<Move>, rng: &mut impl Rng) -> (Vec<Env>, Tensor) {
    let games: Vec<_> = (0..BATCH_SIZE)
        .map(|_| random_env(ply, actions, rng))
        .collect();
    let tensor = Tensor::cat(
        &games
            .iter()
            .map(|g| game_to_tensor(g, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);
    (games, tensor)
}

#[allow(clippy::type_complexity)]
fn reference_batches(
    unique_positions: &[Env],
    rng: &mut impl Rng,
) -> (
    (Vec<Env>, Tensor),
    (Vec<Env>, Tensor),
    (Vec<Env>, Tensor),
    (Vec<Env>, Tensor),
    Tensor,
) {
    let early_game = unique_positions
        .iter()
        .filter(|s| s.ply == 8)
        .cloned()
        .choose_multiple(rng, BATCH_SIZE);
    let early_tensor = Tensor::concat(
        &early_game
            .iter()
            .map(|s| game_to_tensor(s, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);
    let late_game = unique_positions
        .iter()
        .filter(|s| s.ply == 60)
        .cloned()
        .choose_multiple(rng, BATCH_SIZE);
    let late_tensor = Tensor::concat(
        &late_game
            .iter()
            .map(|s| game_to_tensor(s, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);

    let mut actions = vec![];
    let (random_early_batch, random_early_tensor) = reference_envs(8, &mut actions, rng);
    let (random_late_batch, random_late_tensor) = reference_envs(60, &mut actions, rng);

    let (_, impossible_early_tensor) = reference_envs(8, &mut actions, rng);
    let impossible_early_tensor = impossible_early_tensor.index_select(
        1,
        &Tensor::from_slice(
            &[6, 7, 4, 5, 2, 3, 0, 1]
                .into_iter()
                .chain(8..input_channels::<N>())
                .map(|x| x as i64)
                .collect::<Vec<_>>(),
        )
        .to(DEVICE),
    );

    (
        (early_game, early_tensor),
        (late_game, late_tensor),
        (random_early_batch, random_early_tensor),
        (random_late_batch, random_late_tensor),
        impossible_early_tensor,
    )
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(432);

    let mut net = Net::new(DEVICE, Some(rng.gen()));

    let mut positions = Vec::new();
    let mut seen_positions = HashSet::new();
    let mut unique_positions = Vec::new();
    for replay in get_replays("4x4_neurips_undirected_00_replays.txt")
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
    ) = reference_batches(&unique_positions, &mut rng);

    let mut buffer = Vec::with_capacity(2048);
    let mut positions = positions.into_iter().filter(|s| {
        !early_game.contains(s)
            && !late_game.contains(s)
            && !random_early_batch.contains(s)
            && !random_late_batch.contains(s)
    });

    let mut current: Vec<f64> = Vec::with_capacity(STEPS);
    let mut after: Vec<f64> = Vec::with_capacity(STEPS);
    let mut early_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut late_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut random_early_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut random_late_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut impossible_losses: Vec<f64> = Vec::with_capacity(STEPS);

    for step in 0..STEPS {
        if step % 100 == 0 {
            println!("step: {step: >8}");
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

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("eee_data.csv")
        .unwrap();
    let content = current
        .into_iter()
        .zip(after)
        .zip(early_losses)
        .zip(late_losses)
        .zip(random_early_losses)
        .zip(random_late_losses)
        .zip(impossible_losses)
        .enumerate()
        .fold(
            "step,current,after,early,late,random_early,random_late,impossible_early\n".to_string(),
            |mut s,
             (
                step,
                (
                    (((((current, after), early), late), random_early), random_late),
                    impossible_early,
                ),
            )| {
                writeln!(
                    &mut s,
                    "{step},{current},{after},{early},{late},{random_early},{random_late},\
                     {impossible_early}"
                )
                .unwrap();
                s
            },
        );
    file.write_all(content.as_bytes()).unwrap();

    println!("Done.");
}
