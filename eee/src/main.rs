use std::{
    collections::HashSet,
    fmt::Write as FmtWrite,
    fs::OpenOptions,
    io::{BufRead, BufReader, Write as IoWrite},
    path::Path,
};

use fast_tak::takparse::{Move, Tps};
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

const STEPS: usize = 25_000;
const BATCH_SIZE: usize = 128;
const DEVICE: Device = Device::Cuda(0);
const LEARNING_RATE: f64 = 1e-4;

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

fn rnd(path: &nn::Path) -> nn::SequentialT {
    const RES_BLOCKS: u32 = 8;
    const FILTERS: i64 = 64;
    let mut net = nn::seq_t()
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
    for n in 0..RES_BLOCKS {
        net = net.add(ResidualBlock::new(
            &(path / format!("res_block_{n}")),
            FILTERS,
            FILTERS,
        ));
    }
    net.add(SmallBlock::new(&(path / "last_small_block"), FILTERS, 32))
        .add_fn(|x| x.flatten(1, 3))
}

#[allow(clippy::too_many_lines)]
fn main() {
    let seed: u64 = 12345;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let vs = nn::VarStore::new(DEVICE);
    let root = vs.root();
    let target = rnd(&(&root / "target"));
    let predictor = rnd(&(&root / "predictor"));

    let mut opt = Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut replays = get_replays("4x4_old_directed_01_replays.txt");

    let mut seen_positions = HashSet::new();
    let mut unique_positions = Vec::new();

    for replay in replays {
        let mut env = replay.env;
        for action in replay.actions {
            if seen_positions.insert(env.clone().canonical()) {
                unique_positions.push(env.clone());
            }
            env.step(action);
        }
    }

    unique_positions.truncate(STEPS * BATCH_SIZE);
    println!("{}", unique_positions.len());

    let early_batch = unique_positions[..1_000_000].choose_multiple(&mut rng, 128);
    let early_tensor = Tensor::concat(
        &early_batch
            .into_iter()
            .map(|s| game_to_tensor(s, Device::Cpu))
            .collect::<Vec<_>>(),
        1,
    )
    .to(DEVICE);
    let late_batch =
        unique_positions[unique_positions.len() - 1_000_000..].choose_multiple(&mut rng, 128);
    let late_tensor = Tensor::concat(
        &late_batch
            .into_iter()
            .map(|s| game_to_tensor(s, Device::Cpu))
            .collect::<Vec<_>>(),
        1,
    )
    .to(DEVICE);

    let mut unique_positions_iter = unique_positions.into_iter();
    let mut buffer = Vec::with_capacity(2048);

    let mut losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut early_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut late_losses: Vec<f64> = Vec::with_capacity(STEPS);

    let mut running_mean = early_tensor.zeros_like();
    let mut running_sum_squares = running_mean.ones_like() * 0.0001;
    // let mut running_sum_squares_output = 0.0;

    for step in 0..STEPS {
        if step % 100 == 0 {
            println!("step: {step: >8}");
        }

        // Add replays to buffer until we have enough.
        while buffer.len() < 1024 {
            let position = unique_positions_iter.next().unwrap();
            buffer.push(position);
        }

        // Sample a batch.
        buffer.shuffle(&mut rng);
        let batch = buffer.split_off(buffer.len() - BATCH_SIZE);
        let tensor = if (5_000..5_004).contains(&step) {
            early_tensor.copy()
        } else if (20_000..20_004).contains(&step) {
            late_tensor.copy()
        } else {
            Tensor::concat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor(&env, Device::Cpu))
                    .collect::<Vec<_>>(),
                0,
            )
            .to(DEVICE)
        };

        // Update normalization statistics.
        let new_running_mean = &running_mean + (&tensor - &running_mean) / (step + 1) as i64;
        running_sum_squares += (&tensor - &running_mean) * (&tensor - &new_running_mean);
        running_mean = new_running_mean;
        let running_variance = &running_sum_squares / (step + 1) as i64; // Normalize.

        // Compute loss for early batch.
        let input = ((&early_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        early_losses.push(loss.try_into().unwrap());

        // Compute loss for late batch.
        let input = ((&late_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        late_losses.push(loss.try_into().unwrap());

        // Do a training step.
        let input = ((tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, true);
        let loss = (target_out - predictor_out).square().mean(None);
        opt.backward_step(&loss);

        // Save the normalized loss.
        losses.push(loss.try_into().unwrap());
    }

    println!("{running_mean}");
    println!("{running_mean:?}");
    println!("{running_sum_squares}");
    println!("{running_sum_squares:?}");

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("rnd_data.csv")
        .unwrap();
    let content = losses
        .into_iter()
        .zip(early_losses)
        .zip(late_losses)
        .enumerate()
        .fold(
            "step,loss,early,late\n".to_string(),
            |mut s, (step, ((loss, early), late))| {
                writeln!(&mut s, "{step},{loss},{early},{late}").unwrap();
                s
            },
        );
    file.write_all(content.as_bytes()).unwrap();

    println!("Done.");
}
