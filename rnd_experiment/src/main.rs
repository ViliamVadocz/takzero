use std::{
    collections::HashSet,
    fs::OpenOptions,
    io::{BufRead, BufReader, Write},
    iter::repeat,
};

use charming::{
    component::{Axis, Title},
    series::Line,
    theme::Theme,
    Chart,
    HtmlRenderer,
};
use data::{ORIGINAL, STARTING, TEST, TRAINING, UNLIKELY};
use fast_tak::{
    takparse::{Color, Piece, Square, Tps},
    Game,
};
use rand::{
    rngs::StdRng,
    seq::{IteratorRandom, SliceRandom},
    Rng,
    SeedableRng,
};
use rayon::prelude::*;
use takzero::{
    network::repr::{game_to_tensor, input_size},
    search::env::Environment,
    target::Replay,
};
use tch::{
    nn::{self, Adam, ModuleT, OptimizerConfig, VarStore},
    Device,
    Tensor,
};

mod data;

const N: usize = 5;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;

const BATCH_SIZE: usize = 512;

fn target_rnd(path: &nn::Path) -> nn::SequentialT {
    nn::seq_t()
        .add_fn(|x| x.view([-1, input_size::<N>() as i64]))
        .add(nn::linear(
            path,
            input_size::<N>() as i64,
            1024,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(path, 1024, 1024, nn::LinearConfig::default()))
        .add_fn(Tensor::relu)
        .add(nn::linear(path, 1024, 512, nn::LinearConfig::default()))
}

fn improving_rnd(path: &nn::Path) -> nn::SequentialT {
    nn::seq_t()
        .add_fn(|x| x.view([-1, input_size::<N>() as i64]))
        .add(nn::linear(
            path,
            input_size::<N>() as i64,
            1024,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(path, 1024, 1024, nn::LinearConfig::default()))
        .add_fn(Tensor::relu)
        .add(nn::linear(path, 1024, 512, nn::LinearConfig::default()))
}

const DEVICE: Device = Device::Cuda(0);
const REPLAY_LIMIT: usize = 2_000_000;
const STARTING_REPLAY_BUFFER_SIZE: usize = 10_000;
const NEW_REPLAYS_PER_STEP: usize = 1;
const STEPS: usize = 2_000;
const SKIP: usize = 250;

fn main() {
    draw_chart(
        TRAINING.to_owned(),
        TEST.to_owned(),
        ORIGINAL.to_owned(),
        STARTING.to_owned(),
        UNLIKELY.to_owned(),
    );

    experiment()
}

fn experiment() {
    let mut rng = StdRng::seed_from_u64(70);

    let starting_positions: Vec<_> = (0..25 * 24)
        .into_par_iter()
        .map(|i| {
            let mut actions = Vec::new();
            let mut game = Env::default();
            game.populate_actions(&mut actions);
            let len = actions.len();
            game.step(actions.drain(..).nth(i % len).unwrap());
            game.populate_actions(&mut actions);
            let len2 = actions.len();
            game.step(actions.drain(..).nth((i / len) % len2).unwrap());
            game
        })
        .collect();
    let unlikely_positions: Vec<_> = (0..BATCH_SIZE)
        .map(|_| generate_unlikely_position(&mut rng))
        .collect();
    for p in &unlikely_positions {
        let tps: Tps = p.clone().into();
        println!("{tps}");
    }

    let file = OpenOptions::new()
        .read(true)
        .open(".\\replays_final.txt")
        .unwrap();
    let replay_buffer: HashSet<_> = BufReader::new(file)
        .lines()
        .choose_multiple(&mut rng, REPLAY_LIMIT)
        .into_par_iter()
        .map(|line| {
            let replay: Replay<Env> = line.unwrap().parse().unwrap();
            replay.env // .canonical()
        })
        .collect();
    println!("loaded {} unique replays!", replay_buffer.len());

    let (mut training, mut simulated_buffer, test) = {
        let mut replay_buffer: Vec<_> = replay_buffer.into_iter().collect();
        replay_buffer.shuffle(&mut rng);
        let len = replay_buffer.len();
        let mut iter = replay_buffer.into_iter();
        let mut training = iter.by_ref().take(len / 2).collect::<Vec<_>>().into_iter();
        let simulated_buffer: Vec<_> = training
            .by_ref()
            .take(STARTING_REPLAY_BUFFER_SIZE)
            .collect();
        (training, simulated_buffer, iter.collect::<Vec<_>>())
    };
    let original = simulated_buffer.clone();

    let vs = VarStore::new(DEVICE);
    let improving = improving_rnd(&(vs.root() / "improving"));
    let target = target_rnd(&(vs.root() / "target"));

    let mut opt = Adam::default().build(&vs, 1e-3).unwrap();

    let mut training_losses = Vec::new();
    let mut test_losses = Vec::new();
    let mut original_losses = Vec::new();
    let mut starting_losses = Vec::new();
    let mut unlikely_losses = Vec::new();
    for step in 0..STEPS {
        print!("step: {step}, ");

        // Test
        tch::no_grad(|| {
            let batch = test.choose_multiple(&mut rng, BATCH_SIZE);
            let inputs = Tensor::cat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor::<N, HALF_KOMI>(env, DEVICE))
                    .collect::<Vec<_>>(),
                0,
            );

            let actual = target.forward_t(&inputs, false).detach();
            let predicted = improving.forward_t(&inputs, false);
            let loss = predicted
                .mse_loss(&actual, tch::Reduction::None)
                .sum_dim_intlist(1, false, None)
                .mean(None);

            test_losses.push(f32::try_from(loss).unwrap());
        });

        // Test on original
        tch::no_grad(|| {
            let batch = original.choose_multiple(&mut rng, BATCH_SIZE);
            let inputs = Tensor::cat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor::<N, HALF_KOMI>(env, DEVICE))
                    .collect::<Vec<_>>(),
                0,
            );

            let actual = target.forward_t(&inputs, false).detach();
            let predicted = improving.forward_t(&inputs, false);
            let loss = predicted
                .mse_loss(&actual, tch::Reduction::None)
                .sum_dim_intlist(1, false, None)
                .mean(None);

            original_losses.push(f32::try_from(loss).unwrap());
        });

        // Test on starting
        tch::no_grad(|| {
            let batch = starting_positions.choose_multiple(&mut rng, BATCH_SIZE);
            let inputs = Tensor::cat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor::<N, HALF_KOMI>(env, DEVICE))
                    .collect::<Vec<_>>(),
                0,
            );

            let actual = target.forward_t(&inputs, false).detach();
            let predicted = improving.forward_t(&inputs, false);
            let loss = predicted
                .mse_loss(&actual, tch::Reduction::None)
                .sum_dim_intlist(1, false, None)
                .mean(None);

            starting_losses.push(f32::try_from(loss).unwrap());
        });

        // Test on impossible
        tch::no_grad(|| {
            let batch = unlikely_positions.choose_multiple(&mut rng, BATCH_SIZE);
            let inputs = Tensor::cat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor::<N, HALF_KOMI>(env, DEVICE))
                    .collect::<Vec<_>>(),
                0,
            );

            let actual = target.forward_t(&inputs, false).detach();
            let predicted = improving.forward_t(&inputs, false);
            let loss = predicted
                .mse_loss(&actual, tch::Reduction::None)
                .sum_dim_intlist(1, false, None)
                .mean(None);

            unlikely_losses.push(f32::try_from(loss).unwrap());
        });

        // Training
        {
            println!("size of buffer: {}", simulated_buffer.len());
            let batch = simulated_buffer.choose_multiple(&mut rng, BATCH_SIZE);
            let inputs = Tensor::cat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor::<N, HALF_KOMI>(env, DEVICE))
                    .collect::<Vec<_>>(),
                0,
            );

            let actual = tch::no_grad(|| {
                target
                    .forward_t(&inputs.set_requires_grad(false), false)
                    .detach()
            });
            let predicted = improving.forward_t(&inputs, true);
            let loss = predicted
                .mse_loss(&actual, tch::Reduction::None)
                .sum_dim_intlist(1, false, None)
                .mean(None);
            opt.backward_step(&loss);

            training_losses.push(f32::try_from(loss).unwrap());
            simulated_buffer.extend(training.by_ref().take(NEW_REPLAYS_PER_STEP));
        }
    }

    let mut file = std::fs::File::create("data.rs").unwrap();
    #[rustfmt::skip]
    file.write_all(
        format!(
            "pub const TRAINING: &[f32] = &{training_losses:?};\n\
             pub const TEST: &[f32] = &{test_losses:?};\n\
             pub const ORIGINAL: &[f32] = &{original_losses:?};\n\
             pub const STARTING: &[f32] = &{starting_losses:?};\n\
             pub const UNLIKELY: &[f32] = &{unlikely_losses:?};\n"
        )
        .as_bytes(),
    )
    .unwrap();

    draw_chart(
        training_losses,
        test_losses,
        original_losses,
        starting_losses,
        unlikely_losses,
    );
}

fn draw_chart(
    training_losses: Vec<f32>,
    test_losses: Vec<f32>,
    original_losses: Vec<f32>,
    starting_losses: Vec<f32>,
    unlikely_losses: Vec<f32>,
) {
    #[allow(clippy::iter_skip_zero)]
    let chart = Chart::new()
        .title(
            Title::new()
                .text(format!(
                    "RND, replays={STARTING_REPLAY_BUFFER_SIZE}+{NEW_REPLAYS_PER_STEP}n, \
                     batch_size={BATCH_SIZE}"
                ))
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("training steps"))
        .y_axis(Axis::new().name("loss"))
        .series(
            Line::new().dataset_id("training").show_symbol(false).data(
                training_losses
                    .into_iter()
                    .enumerate()
                    .skip(SKIP)
                    .map(|(x, y)| vec![x as f64, y as f64])
                    .collect(),
            ),
        )
        .series(
            Line::new().dataset_id("test").show_symbol(false).data(
                test_losses
                    .into_iter()
                    .enumerate()
                    .skip(SKIP)
                    .map(|(x, y)| vec![x as f64, y as f64])
                    .collect(),
            ),
        )
        .series(
            Line::new().dataset_id("original").show_symbol(false).data(
                original_losses
                    .into_iter()
                    .enumerate()
                    .skip(SKIP)
                    .map(|(x, y)| vec![x as f64, y as f64])
                    .collect(),
            ),
        )
        .series(
            Line::new().dataset_id("starting").show_symbol(false).data(
                starting_losses
                    .into_iter()
                    .enumerate()
                    .skip(SKIP)
                    .map(|(x, y)| vec![x as f64, y as f64])
                    .collect(),
            ),
        )
        .series(
            Line::new().dataset_id("unlikely").show_symbol(false).data(
                unlikely_losses
                    .into_iter()
                    .enumerate()
                    .skip(SKIP)
                    .map(|(x, y)| vec![x as f64, y as f64])
                    .collect(),
            ),
        );

    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
    println!("graph saved");
}

fn generate_unlikely_position(rng: &mut impl Rng) -> Env {
    const MIN_FLATS: u8 = 1;
    const MAX_FLATS: u8 = 10;
    const MAX_WALLS: u8 = 3;
    const MAX_PLIES: u16 = 100;

    let mut env = Env::default();

    // pick how many flats will be placed
    let white = rng.gen_range(MIN_FLATS..=MAX_FLATS);
    let black = rng.gen_range(MIN_FLATS..=MAX_FLATS);
    let mut colors: Vec<_> = repeat(Color::White)
        .take(white as usize)
        .chain(repeat(Color::Black).take(black as usize))
        .collect();
    colors.shuffle(rng);

    // place flats in stacks
    for color in colors {
        env.board
            .get_mut(random_square(rng))
            .unwrap()
            .stack(Piece::Flat, color)
            .expect("All stacks should have flats on top, so stacking is allowed");
    }

    // pick how many walls will be attempted to be placed
    let white = rng.gen_range(0..=MAX_WALLS);
    let black = rng.gen_range(0..=MAX_WALLS);
    let mut colors: Vec<_> = repeat(Color::White)
        .take(white as usize)
        .chain(repeat(Color::Black).take(black as usize))
        .collect();
    colors.shuffle(rng);

    // place walls
    for color in colors {
        let _ = env
            .board
            .get_mut(random_square(rng))
            .unwrap()
            .stack(Piece::Wall, color);
    }

    // place capstones
    while {
        for color in [Color::White, Color::Black] {
            let _ = env
                .board
                .get_mut(random_square(rng))
                .unwrap()
                .stack(Piece::Cap, color);
        }
        rng.gen_bool(0.01) // small chance to generate more capstones
    } {}

    // randomize other data
    env.ply = rng.gen_range(0..MAX_PLIES);
    env.to_move = if rng.gen() {
        Color::White
    } else {
        Color::Black
    };
    env.white_reserves.stones = rng.gen_range(0..=env.white_reserves.stones);
    env.white_reserves.caps = rng.gen_range(0..=env.white_reserves.caps);
    env.black_reserves.stones = rng.gen_range(0..=env.black_reserves.stones);
    env.black_reserves.caps = rng.gen_range(0..=env.black_reserves.caps);

    env
}

fn random_square(rng: &mut impl Rng) -> Square {
    let row = rng.gen_range(0..N) as u8;
    let col = rng.gen_range(0..N) as u8;
    Square::new(col, row)
}
