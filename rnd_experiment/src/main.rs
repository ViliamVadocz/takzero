use std::{
    collections::HashSet,
    fs::OpenOptions,
    io::{BufRead, BufReader},
};

use charming::{
    component::{Axis, Title},
    series::Line,
    theme::Theme,
    Chart,
    HtmlRenderer,
};
use fast_tak::Game;
use rand::{
    rngs::StdRng,
    seq::{IteratorRandom, SliceRandom},
    SeedableRng,
};
use rayon::prelude::*;
use takzero::{
    network::repr::{game_to_tensor, input_size},
    target::Replay,
};
use tch::{
    nn::{self, Adam, ModuleT, OptimizerConfig, VarStore},
    Device,
    Tensor,
};

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
const NEW_REPLAYS_PER_STEP: usize = 10;
const STEPS: usize = 4_000;
const SKIP: usize = 250;

fn main() {
    let mut rng = StdRng::seed_from_u64(70);

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

            let actual = target
                .forward_t(&inputs.set_requires_grad(false), false)
                .detach();
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

            let actual = target
                .forward_t(&inputs.set_requires_grad(false), false)
                .detach();
            let predicted = improving.forward_t(&inputs, false);
            let loss = predicted
                .mse_loss(&actual, tch::Reduction::None)
                .sum_dim_intlist(1, false, None)
                .mean(None);

            original_losses.push(f32::try_from(loss).unwrap());
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

    let chart = Chart::new()
        .title(Title::new().text("RND").left("center").top(0))
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
        );

    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
    println!("graph saved");
}
