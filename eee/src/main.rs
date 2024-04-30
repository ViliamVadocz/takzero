use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::Path,
};

use charming::{
    component::{Axis, Grid, Legend, Title},
    element::Symbol,
    series::Line,
    theme::Theme,
    Chart,
    HtmlRenderer,
};
use fast_tak::takparse::Move;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
    SeedableRng,
};
use takzero::{
    network::{
        net4_neurips::{Env, Net, N},
        repr::{
            game_to_tensor,
            input_channels,
            input_size,
            move_mask,
            output_channels,
            output_size,
            policy_tensor,
        },
        EnsembleNetwork,
        Network,
        RndNetwork,
    },
    search::env::Environment,
    target::{Augment, Replay, Target},
};
use tch::{
    nn::{self, Adam, ModuleT, Optimizer, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

const STEPS: usize = 20_000;
const REFERENCE_PERIOD: usize = 500;
const BATCH_SIZE: usize = 128;
const DEVICE: Device = Device::Cuda(0);
const LEARNING_RATE: f64 = 1e-4;

fn get_replays(path: impl AsRef<Path>) -> impl Iterator<Item = Replay<Env>> {
    BufReader::new(OpenOptions::new().read(true).open(path).unwrap())
        .lines()
        .filter_map(|line| line.ok()?.parse::<Replay<Env>>().ok())
}

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

// struct Tensors {
//     input: Tensor,
//     mask: Tensor,
//     target_value: Tensor,
//     target_policy: Tensor,
// }

// impl Default for Tensors {
//     fn default() -> Self {
//         let policy = Tensor::zeros(
//             [
//                 BATCH_SIZE as i64,
//                 output_channels::<N>() as i64,
//                 N as i64,
//                 N as i64,
//             ],
//             (Kind::Float, DEVICE),
//         );
//         Self {
//             input: Tensor::zeros(
//                 [
//                     BATCH_SIZE as i64,
//                     input_channels::<N>() as i64,
//                     N as i64,
//                     N as i64,
//                 ],
//                 (Kind::Float, DEVICE),
//             ),
//             mask: Tensor::zeros_like(&policy).to_dtype(Kind::Bool, false,
// false),             target_value: Tensor::new(),
//             target_policy: policy,
//         }
//     }
// }

// fn create_input_and_target_tensors(
//     tensors: &mut Tensors,
//     batch: impl Iterator<Item = Target<Env>>,
//     rng: &mut impl Rng,
// ) {
//     // Create input tensors.
//     let mut inputs = Vec::with_capacity(BATCH_SIZE);
//     let mut policy_targets = Vec::with_capacity(BATCH_SIZE);
//     let mut masks = Vec::with_capacity(BATCH_SIZE);
//     let mut value_targets = Vec::with_capacity(BATCH_SIZE);
//     for target in batch {
//         let target = target.augment(rng);
//         inputs.push(game_to_tensor(&target.env, Device::Cpu));
//         policy_targets.push(policy_tensor::<N>(&target.policy, Device::Cpu));
//         masks.push(move_mask::<N>(
//             &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
//             Device::Cpu,
//         ));
//         value_targets.push(target.value);
//     }

//     // Get network output.
//     tensors.input.copy_(&Tensor::cat(&inputs, 0));
//     tensors.mask.copy_(&Tensor::cat(&masks, 0));
//     // Get the target.
//     tensors
//         .target_policy
//         .copy_(&Tensor::cat(&policy_targets, 0));
//     tensors.target_value =
// Tensor::from_slice(&value_targets).unsqueeze(1).to(DEVICE); }

// fn compute_loss_and_take_step(net: &Net, opt: &mut Optimizer, tensors:
// &Tensors) {     // Get network output.
//     let (policy, network_value, _, ensemble) = net.forward_t(&tensors.input,
// true);     let log_softmax_network_policy = policy
//         .masked_fill(&tensors.mask, f64::from(f32::MIN))
//         .view([-1, output_size::<N>() as i64])
//         .log_softmax(1, Kind::Float);

//     // Calculate loss.
//     let loss_policy = -(log_softmax_network_policy
//         * &tensors .target_policy .view([BATCH_SIZE as i64,
//           output_size::<N>() as i64]))
//         .sum(Kind::Float)
//         / i64::try_from(BATCH_SIZE).unwrap();
//     let loss_value = (&tensors.target_value - network_value)
//         .square()
//         .mean(Kind::Float);
//     let loss_ensemble = (&tensors.target_value - ensemble)
//         .square()
//         .mean(Kind::Float);
//     let loss = &loss_policy + &loss_value + &loss_ensemble;
//     // #[rustfmt::skip]
//     // println!(
//     //     "loss = {loss:?}\n\
//     //      loss_policy = {loss_policy:?}\n\
//     //      loss_value = {loss_value:?}\n\
//     //      loss_ensemble = {loss_ensemble:?}"
//     // );
//     // Take step.
//     opt.backward_step(&loss);
// }

fn rnd(path: &nn::Path) -> nn::SequentialT {
    const HIDDEN_LAYER: i64 = 1024;
    const OUTPUT: i64 = 512;
    nn::seq_t()
        .add_fn(|x| x.view([-1, input_size::<N>() as i64]))
        .add_fn(|x| x / x.square().sum_dim_intlist(1, true, None))
        .add(nn::linear(
            path / "input_linear",
            input_size::<N>() as i64,
            HIDDEN_LAYER,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            path / "hidden_linear",
            HIDDEN_LAYER,
            HIDDEN_LAYER,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            path / "final_linear",
            HIDDEN_LAYER,
            OUTPUT,
            nn::LinearConfig::default(),
        ))
}

fn main() {
    let seed: u64 = 12345;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let vs = nn::VarStore::new(DEVICE);
    let root = vs.root();
    let target = rnd(&(&root / "target"));
    let predictor = rnd(&(&root / "predictor"));

    let mut opt = Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut replays = get_replays("directed-replays-01.txt");
    let mut buffer = Vec::with_capacity(2048);
    let mut reference_batches = Vec::new();

    let mut losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut batch_losses: Vec<Vec<f64>> = Vec::new();
    for step in 0..STEPS {
        if step % 100 == 0 {
            println!("step: {step: >8}");
        }

        // Add replays to buffer until we have enough.
        while buffer.len() < 1024 {
            let replay = replays.next().unwrap();
            let mut env = replay.env;
            buffer.extend(replay.actions.into_iter().map(|a| {
                env.step(a);
                env.clone()
            }));
        }

        // Sample a batch.
        buffer.shuffle(&mut rng);
        let batch = buffer.split_off(buffer.len() - BATCH_SIZE);
        let tensor = Tensor::concat(
            &batch
                .into_iter()
                .map(|env| game_to_tensor(&env, Device::Cpu))
                .collect::<Vec<_>>(),
            0,
        )
        .to(DEVICE);

        // Do a training step.
        let target_out = target.forward_t(&tensor, false).detach();
        let predictor_out = predictor.forward_t(&tensor, true);
        let loss = (target_out - predictor_out).square().mean(None);
        opt.backward_step(&loss);
        // Save the loss.
        losses.push(loss.try_into().unwrap());

        // Save reference batch.
        if step % REFERENCE_PERIOD == 0 {
            reference_batches.push(tensor.copy());
            batch_losses.push(Vec::with_capacity(STEPS - step));
        }

        for (batch_loss, reference_batch) in batch_losses.iter_mut().zip(reference_batches.iter()) {
            // Compute loss for reference batch.
            let target_out = target.forward_t(reference_batch, false).detach();
            let predictor_out = predictor.forward_t(reference_batch, true);
            let loss = (target_out - predictor_out).square().mean(None);
            batch_loss.push(loss.try_into().unwrap());
        }
    }

    println!("Plotting.");

    let mut chart = Chart::new()
        .title(Title::new().text("RND Loss").left("center").top(0))
        .x_axis(Axis::new().name("Training steps"))
        .y_axis(Axis::new().name("Loss").min(0).max(0.0001))
        .grid(Grid::new())
        .legend(
            Legend::new()
                .data(
                    std::iter::once("Recent batch".to_string())
                        .chain(
                            (0..batch_losses.len())
                                .map(|i| format!("{} batch", i * REFERENCE_PERIOD)),
                        )
                        .collect(),
                )
                .bottom(10)
                .left(30),
        )
        .series(
            Line::new()
                .data(
                    losses
                        .into_iter()
                        .enumerate()
                        .map(|(i, l)| vec![i as f64, l])
                        .collect(),
                )
                .name("Recent batch")
                .symbol(Symbol::None),
        );
    for (i, batch_loss) in batch_losses.into_iter().enumerate() {
        let start = REFERENCE_PERIOD * i;
        chart = chart.series(
            Line::new()
                .data(
                    batch_loss
                        .into_iter()
                        .enumerate()
                        .map(|(i, l)| vec![(i + start) as f64, l])
                        .collect(),
                )
                .name(format!("{start} batch"))
                .symbol(Symbol::None),
        );
    }

    let mut renderer = HtmlRenderer::new("graph", 1400, 700).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();

    println!("Done.");
}

// fn ensemble_experiment() {
//     let mut selfplay =
// targets("targets-selfplay-no-explore-01.txt").skip(BATCH_SIZE * 15_000);
//     let mut reanalyze =
// targets("targets-reanalyze-no-explore-01.txt").skip(BATCH_SIZE * 10_000);
//     println!("loaded targets");

//     let mut buffer = Vec::with_capacity(16_384);
//     buffer.extend(selfplay.by_ref().take(5_000));
//     buffer.extend(reanalyze.by_ref().take(5_000));
//     println!("created buffer");

//     let seed: u64 = 12345;
//     let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

//     let mut net = Net::new(Device::Cuda(0), Some(rng.gen()));
//     let mut opt = Adam::default().build(net.vs_mut(),
// LEARNING_RATE).unwrap();     println!("created network");

//     // Create reference positions.
//     let mut actions = Vec::new();
//     let (_early_positions, early_tensor) = reference_envs(5, &mut actions,
// &mut rng);     let (_mid_positions, mid_tensor) = reference_envs(30, &mut
// actions, &mut rng);     let (_late_positions, late_tensor) =
// reference_envs(120, &mut actions, &mut rng);     let mut lines = [vec![],
// vec![], vec![]];

//     let mut tensors = Tensors::default();
//     for step in 1..=STEPS {
//         println!("{step} / {STEPS}");

//         buffer.extend(selfplay.by_ref().take(BATCH_SIZE / 2));
//         buffer.extend(reanalyze.by_ref().take(BATCH_SIZE / 2));
//         buffer.shuffle(&mut rng);

//         create_input_and_target_tensors(
//             &mut tensors,
//             buffer.drain(buffer.len() - BATCH_SIZE..),
//             &mut rng,
//         );
//         compute_loss_and_take_step(&net, &mut opt, &tensors);

//         if step % 100 == 1 {
//             let (_, value, _, ensemble) = net.forward_t(&early_tensor,
// false);             lines[0].push((
//                 step as f64,
//                 f64::try_from((ensemble.mean_dim(1, false, None) -
// value).mean(None)).unwrap(),
// f64::try_from(ensemble.var_dim(1i64, false, false).mean(None)).unwrap(),
//             ));
//             let (_, value, _, ensemble) = net.forward_t(&mid_tensor, false);
//             lines[1].push((
//                 step as f64,
//                 f64::try_from((ensemble.mean_dim(1, false, None) -
// value).mean(None)).unwrap(),
// f64::try_from(ensemble.var_dim(1i64, false, false).mean(None)).unwrap(),
//             ));
//             let (_, value, _, ensemble) = net.forward_t(&late_tensor, false);
//             lines[2].push((
//                 step as f64,
//                 f64::try_from((ensemble.mean_dim(1, false, None) -
// value).mean(None)).unwrap(),
// f64::try_from(ensemble.var_dim(1i64, false, false).mean(None)).unwrap(),
//             ));
//         }
//     }
//     plot(lines);
// }

// fn plot(mut lines: [Vec<(f64, f64, f64)>; 3]) {
//     let chart = Chart::new()
//         .title(
//             Title::new()
//                 .text("Ensemble Variance During Training")
//                 .left("center")
//                 .top(0),
//         )
//         .x_axis(Axis::new().name("Training steps"))
//         .y_axis(Axis::new().name("Variance"))
//         .grid(Grid::new())
//         .legend(
//             Legend::new()
//                 .data(vec!["Early (5 ply)", "Middle (30 ply)", "Late (<=120
// ply)"])                 .bottom(10)
//                 .left(30),
//         )
//         .series(
//             Line::new()
//                 .data(
//                     std::mem::take(&mut lines[0])
//                         .into_iter()
//                         .map(|(s, _, v)| vec![s, v])
//                         .collect(),
//                 )
//                 .name("Early (5 ply)")
//                 .symbol(Symbol::None),
//         )
//         .series(
//             Line::new()
//                 .data(
//                     std::mem::take(&mut lines[1])
//                         .into_iter()
//                         .map(|(s, _, v)| vec![s, v])
//                         .collect(),
//                 )
//                 .name("Middle (30 ply)")
//                 .symbol(Symbol::None),
//         )
//         .series(
//             Line::new()
//                 .data(
//                     std::mem::take(&mut lines[2])
//                         .into_iter()
//                         .map(|(s, _, v)| vec![s, v])
//                         .collect(),
//                 )
//                 .name("Late (<=120 ply)")
//                 .symbol(Symbol::None),
//         );

//     let mut renderer = HtmlRenderer::new("graph", 1400,
// 700).theme(Theme::Infographic);     renderer.save(&chart,
// "graph.html").unwrap(); }
