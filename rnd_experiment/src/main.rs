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
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;
use takzero::{
    network::repr::{game_to_tensor, input_channels},
    target::Replay,
};
use tch::{
    nn::{self, Adam, ModuleT, OptimizerConfig, VarStore},
    Device,
    Kind,
    Tensor,
};

const N: usize = 5;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;

const BATCH_SIZE: usize = 512;

// core
const FILTERS: i64 = 128;
// rnd
const BEFORE_LINEAR: i64 = N as i64 * N as i64 * FILTERS;
const LINEAR_SIZE: i64 = 1024;

fn rnd(path: &nn::Path) -> nn::SequentialT {
    nn::seq_t()
        .add(nn::conv2d(
            path,
            input_channels::<N>() as i64,
            FILTERS,
            3,
            nn::ConvConfig {
                stride: 1,
                padding: 1,
                ..Default::default()
            },
        ))
        .add_fn(Tensor::relu)
        .add_fn(|x| x.view([-1, BEFORE_LINEAR]))
        .add(nn::linear(
            path,
            BEFORE_LINEAR,
            LINEAR_SIZE,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            path,
            LINEAR_SIZE,
            LINEAR_SIZE,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            path,
            LINEAR_SIZE,
            LINEAR_SIZE,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            path,
            LINEAR_SIZE,
            LINEAR_SIZE,
            nn::LinearConfig::default(),
        ))
}

const DEVICE: Device = Device::Cuda(0);
const REPLAY_LIMIT: usize = 2_000_000;

fn main() {
    let mut rng = StdRng::seed_from_u64(70);

    let file = OpenOptions::new()
        .read(true)
        .open(".\\_data\\5x5\\1\\replays_final.txt")
        .unwrap();
    let replay_buffer: HashSet<_> = BufReader::new(file)
        .lines()
        .take(REPLAY_LIMIT) // FIXME: This is here just to make the tests go faster.
        .par_bridge()
        .map(|line| {
            let replay: Replay<Env> = line.unwrap().parse().unwrap();
            replay.env.canonical()
        })
        .collect();
    println!("loaded {} unique replays!", replay_buffer.len());

    let mut replay_buffer: Vec<_> = replay_buffer.into_iter().collect();
    replay_buffer.shuffle(&mut rng);
    let (training, test) = replay_buffer.split_at(replay_buffer.len() / 2);

    let vs = VarStore::new(DEVICE);
    let root = vs.root();
    let improving = rnd(&root);
    let target = rnd(&root);

    let mut opt = Adam {
        wd: 1e-3,
        ..Default::default()
    }
    .build(&vs, 1e-3)
    .unwrap();

    let mut training_losses = Vec::new();
    let mut test_losses = Vec::new();
    for epoch in 0..100 {
        println!("epoch: {epoch}");

        // Test
        {
            let batch = test.choose_multiple(&mut rng, BATCH_SIZE);
            let inputs = Tensor::cat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor::<N, HALF_KOMI>(env, DEVICE))
                    .collect::<Vec<_>>(),
                0,
            );

            let predicted = improving.forward_t(&inputs, false);
            let actual = target.forward_t(&inputs, false).detach();
            let loss = predicted.mse_loss(&actual, tch::Reduction::Mean);

            test_losses.push(loss.to_device_(Device::Cpu, Kind::Float, true, false));
        }

        // Training
        {
            let batch = training.choose_multiple(&mut rng, BATCH_SIZE);
            let inputs = Tensor::cat(
                &batch
                    .into_iter()
                    .map(|env| game_to_tensor::<N, HALF_KOMI>(env, DEVICE))
                    .collect::<Vec<_>>(),
                0,
            );

            let predicted = improving.forward_t(&inputs, true);
            let actual = target.forward_t(&inputs, false).detach();
            let loss = predicted.mse_loss(&actual, tch::Reduction::Mean);
            opt.backward_step(&loss);

            training_losses.push(loss.to_device_(Device::Cpu, Kind::Float, true, false));
        }
    }
    tch::Cuda::synchronize(0);

    let training_loss: Vec<f32> = Tensor::stack(&training_losses, 0).try_into().unwrap();
    let test_loss: Vec<f32> = Tensor::stack(&test_losses, 0).try_into().unwrap();

    let chart = Chart::new()
        .title(Title::new().text("RND").left("center").top(0))
        .x_axis(Axis::new().name("training steps"))
        .y_axis(Axis::new().name("loss"))
        .series(
            Line::new().dataset_id("training").show_symbol(false).data(
                training_loss
                    .into_iter()
                    .enumerate()
                    .map(|(x, y)| vec![x as f64, y as f64])
                    .collect(),
            ),
        )
        .series(
            Line::new().dataset_id("test").show_symbol(false).data(
                test_loss
                    .into_iter()
                    .enumerate()
                    .map(|(x, y)| vec![x as f64, y as f64])
                    .collect(),
            ),
        );

    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
    println!("graph saved");
}
