use std::{
    collections::HashSet,
    fmt::Write as FmtWrite,
    fs::OpenOptions,
    io::{BufRead, BufReader, Write as IoWrite},
    path::Path,
};

use fast_tak::takparse::Move;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
    SeedableRng,
};
use takzero::{
    network::{
        net4_neurips::{Env, N},
        repr::{game_to_tensor, input_channels},
        residual::{ResidualBlock, SmallBlock},
    },
    search::env::Environment,
    target::{Replay, Target},
};
use tch::{
    nn::{self, Adam, ModuleT, OptimizerConfig},
    Device,
    Tensor,
};

const STEPS: usize = 45_000;
const BATCH_SIZE: usize = 256;
const DEVICE: Device = Device::Cuda(0);
const LEARNING_RATE: f64 = 1e-4;
const FORCED_USES: u32 = 4;

fn get_replays(path: impl AsRef<Path>) -> impl Iterator<Item = Replay<Env>> {
    BufReader::new(OpenOptions::new().read(true).open(path).unwrap())
        .lines()
        .filter_map(|line| line.ok()?.parse::<Replay<Env>>().ok())
}

#[allow(unused)]
fn get_targets(path: impl AsRef<Path>) -> impl Iterator<Item = Target<Env>> {
    BufReader::new(OpenOptions::new().read(true).open(path).unwrap())
        .lines()
        .filter_map(|line| line.ok()?.parse::<Target<Env>>().ok())
}

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

fn rnd(path: &nn::Path, target: bool) -> nn::SequentialT {
    let res_blocks = 4;
    let filters = 32;
    let mut net = nn::seq_t()
        .add(nn::layer_norm(
            path / "layer_norm",
            vec![
                BATCH_SIZE as i64,
                input_channels::<N>() as i64,
                N as i64,
                N as i64,
            ],
            nn::LayerNormConfig::default(),
        ))
        // .add_fn(|x| x * 2)
        .add(nn::conv2d(
            path / "input_conv2d",
            input_channels::<N>() as i64,
            filters,
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
            filters,
            nn::BatchNormConfig::default(),
        ))
        .add_fn(Tensor::relu);
    for n in 0..res_blocks {
        net = net.add(ResidualBlock::new(
            &(path / format!("res_block_{n}")),
            filters,
            filters,
        ));
    }
    net.add(SmallBlock::new(&(path / "last_small_block"), filters, 32))
        .add_fn(|x| x.flatten(1, 3))
}

#[allow(clippy::too_many_lines)]
fn main() {
    let seed: u64 = 432;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    tch::manual_seed(rng.gen());
    let vs = nn::VarStore::new(DEVICE);
    let root = vs.root();
    let target = rnd(&(&root / "target"), true);
    let predictor = rnd(&(&root / "predictor"), false);
    // for (name, tensor) in &mut vs.variables() {
    //     if name.contains("target") {
    //         *tensor = tensor.set_requires_grad(false);
    //         *tensor *= 1.5;
    //     }
    // }

    let mut opt = Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut positions = Vec::new();
    let mut seen_positions = HashSet::new();
    let mut unique_positions = Vec::new();
    for replay in get_replays("4x4_old_directed_01_replays.txt") {
        let mut env = replay.env;
        for action in replay.actions {
            positions.push(env.clone());
            if seen_positions.insert(env.clone().canonical()) {
                unique_positions.push(env.clone());
            }
            env.step(action);
        }
    }

    let early_game = unique_positions
        .iter()
        .filter(|s| s.ply < 10)
        .cloned()
        .choose_multiple(&mut rng, BATCH_SIZE);
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
        .filter(|s| s.ply >= 60)
        .cloned()
        .choose_multiple(&mut rng, BATCH_SIZE);
    let late_tensor = Tensor::concat(
        &late_game
            .iter()
            .map(|s| game_to_tensor(s, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);

    let mut actions = vec![];
    let (random_early_batch, random_early_tensor) = reference_envs(8, &mut actions, &mut rng);
    let (random_late_batch, random_late_tensor) = reference_envs(120, &mut actions, &mut rng);

    let (_, impossible_early_tensor) = reference_envs(8, &mut actions, &mut rng);
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

    let mut buffer = Vec::with_capacity(2048);
    let mut positions = positions.into_iter().filter(|s| {
        !early_game.contains(s)
            && !late_game.contains(s)
            && !random_early_batch.contains(s)
            && !random_late_batch.contains(s)
    });

    let mut losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut early_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut late_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut random_early_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut random_late_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut impossible_losses: Vec<f64> = Vec::with_capacity(STEPS);

    // let mut running_mean = early_tensor.zeros_like();
    // let mut running_sum_squares = running_mean.ones_like() * 1e-3;
    // let mut running_sum_squares_output = 0.0;

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

        // Update normalization statistics.
        // let new_running_mean = &running_mean + (&tensor - &running_mean) / (step + 1) as i64;
        // running_sum_squares += (&tensor - &running_mean) * (&tensor - &new_running_mean);
        // running_mean = new_running_mean;
        // let running_variance = &running_sum_squares / (step + 1) as i64; // Normalize.

        // Compute loss for early batch.
        let input = &early_tensor; // ((&early_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        early_losses.push(loss.try_into().unwrap());

        // Compute loss for late batch.
        let input = &late_tensor; //  ((&late_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        late_losses.push(loss.try_into().unwrap());

        // Compute loss for random early batch.
        let input = &random_early_tensor; //  ((&random_early_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        random_early_losses.push(loss.try_into().unwrap());

        // Compute loss for random late batch.
        let input = &random_late_tensor; //  ((&random_late_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        random_late_losses.push(loss.try_into().unwrap());

        // Compute loss for impossible early batch
        let input = &impossible_early_tensor;
            // ((& impossible_early_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        impossible_losses.push(loss.try_into().unwrap());

        // Do a training step.
        let input = tensor; // ((tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, true);
        let loss = (target_out - predictor_out).square().mean(None);
        opt.backward_step(&loss);

        // Save the normalized loss.
        losses.push(loss.try_into().unwrap());
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("rnd_data.csv")
        .unwrap();
    let content =
        losses
            .into_iter()
            .zip(early_losses)
            .zip(late_losses)
            .zip(random_early_losses)
            .zip(random_late_losses)
            .zip(impossible_losses)
            .enumerate()
            .fold(
                "step,loss,early,late,random_early,random_late,impossible_early\n".to_string(),
                |mut s,
                 (
                    step,
                    (((((loss, early), late), random_early), random_late), impossible_early),
                )| {
                    writeln!(
                        &mut s,
                        "{step},{loss},{early},{late},{random_early},{random_late},\
                         {impossible_early}"
                    )
                    .unwrap();
                    s
                },
            );
    file.write_all(content.as_bytes()).unwrap();

    println!("Done.");
}
