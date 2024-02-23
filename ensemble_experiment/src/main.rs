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
        net4_ensemble::{Env, Net, N},
        repr::{
            game_to_tensor,
            input_channels,
            move_mask,
            output_channels,
            output_size,
            policy_tensor,
        },
        EnsembleNetwork,
        Network,
    },
    search::env::Environment,
    target::{Augment, Target},
};
use tch::{
    nn::{Adam, Optimizer, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

const STEPS: usize = 20_000;
const BATCH_SIZE: usize = 128;
const DEVICE: Device = Device::Cuda(0);
const LEARNING_RATE: f64 = 1e-4;

fn targets(path: impl AsRef<Path>) -> impl Iterator<Item = Target<Env>> {
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

struct Tensors {
    input: Tensor,
    mask: Tensor,
    target_value: Tensor,
    target_policy: Tensor,
}

impl Default for Tensors {
    fn default() -> Self {
        let policy = Tensor::zeros(
            [
                BATCH_SIZE as i64,
                output_channels::<N>() as i64,
                N as i64,
                N as i64,
            ],
            (Kind::Float, DEVICE),
        );
        Self {
            input: Tensor::zeros(
                [
                    BATCH_SIZE as i64,
                    input_channels::<N>() as i64,
                    N as i64,
                    N as i64,
                ],
                (Kind::Float, DEVICE),
            ),
            mask: Tensor::zeros_like(&policy).to_dtype(Kind::Bool, false, false),
            target_value: Tensor::new(),
            target_policy: policy,
        }
    }
}

fn create_input_and_target_tensors(
    tensors: &mut Tensors,
    batch: impl Iterator<Item = Target<Env>>,
    rng: &mut impl Rng,
) {
    // Create input tensors.
    let mut inputs = Vec::with_capacity(BATCH_SIZE);
    let mut policy_targets = Vec::with_capacity(BATCH_SIZE);
    let mut masks = Vec::with_capacity(BATCH_SIZE);
    let mut value_targets = Vec::with_capacity(BATCH_SIZE);
    for target in batch {
        let target = target.augment(rng);
        inputs.push(game_to_tensor(&target.env, Device::Cpu));
        policy_targets.push(policy_tensor::<N>(&target.policy, Device::Cpu));
        masks.push(move_mask::<N>(
            &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
            Device::Cpu,
        ));
        value_targets.push(target.value);
    }

    // Get network output.
    tensors.input.copy_(&Tensor::cat(&inputs, 0));
    tensors.mask.copy_(&Tensor::cat(&masks, 0));
    // Get the target.
    tensors
        .target_policy
        .copy_(&Tensor::cat(&policy_targets, 0));
    tensors.target_value = Tensor::from_slice(&value_targets).unsqueeze(1).to(DEVICE);
}

fn compute_loss_and_take_step(net: &Net, opt: &mut Optimizer, tensors: &Tensors) {
    // Get network output.
    let (policy, network_value, _, ensemble) = net.forward_t(&tensors.input, true);
    let log_softmax_network_policy = policy
        .masked_fill(&tensors.mask, f64::from(f32::MIN))
        .view([-1, output_size::<N>() as i64])
        .log_softmax(1, Kind::Float);

    // Calculate loss.
    let loss_policy = -(log_softmax_network_policy
        * &tensors
            .target_policy
            .view([BATCH_SIZE as i64, output_size::<N>() as i64]))
        .sum(Kind::Float)
        / i64::try_from(BATCH_SIZE).unwrap();
    let loss_value = (&tensors.target_value - network_value)
        .square()
        .mean(Kind::Float);
    let loss_ensemble = (&tensors.target_value - ensemble)
        .square()
        .mean(Kind::Float);
    let loss = &loss_policy + &loss_value + &loss_ensemble;
    // #[rustfmt::skip]
    // println!(
    //     "loss = {loss:?}\n\
    //      loss_policy = {loss_policy:?}\n\
    //      loss_value = {loss_value:?}\n\
    //      loss_ensemble = {loss_ensemble:?}"
    // );
    // Take step.
    opt.backward_step(&loss);
}

fn main() {
    let mut selfplay = targets("targets-selfplay-no-explore-01.txt").skip(BATCH_SIZE * 15_000);
    let mut reanalyze = targets("targets-reanalyze-no-explore-01.txt").skip(BATCH_SIZE * 10_000);
    println!("loaded targets");

    let mut buffer = Vec::with_capacity(16_384);
    buffer.extend(selfplay.by_ref().take(5_000));
    buffer.extend(reanalyze.by_ref().take(5_000));
    println!("created buffer");

    let seed: u64 = 12345;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(Device::Cuda(0), Some(rng.gen()));
    let mut opt = Adam::default().build(net.vs_mut(), LEARNING_RATE).unwrap();
    println!("created network");

    // Create reference positions.
    let mut actions = Vec::new();
    let (_early_positions, early_tensor) = reference_envs(5, &mut actions, &mut rng);
    let (_mid_positions, mid_tensor) = reference_envs(30, &mut actions, &mut rng);
    let (_late_positions, late_tensor) = reference_envs(120, &mut actions, &mut rng);
    let mut lines = [vec![], vec![], vec![]];

    let mut tensors = Tensors::default();
    for step in 1..=STEPS {
        println!("{step} / {STEPS}");

        buffer.extend(selfplay.by_ref().take(BATCH_SIZE / 2));
        buffer.extend(reanalyze.by_ref().take(BATCH_SIZE / 2));
        buffer.shuffle(&mut rng);

        create_input_and_target_tensors(
            &mut tensors,
            buffer.drain(buffer.len() - BATCH_SIZE..),
            &mut rng,
        );
        compute_loss_and_take_step(&net, &mut opt, &tensors);

        if step % 100 == 1 {
            let (_, value, _, ensemble) = net.forward_t(&early_tensor, false);
            lines[0].push((
                step as f64,
                f64::try_from((ensemble.mean_dim(1, false, None) - value).mean(None)).unwrap(),
                f64::try_from(ensemble.var_dim(1i64, false, false).mean(None)).unwrap(),
            ));
            let (_, value, _, ensemble) = net.forward_t(&mid_tensor, false);
            lines[1].push((
                step as f64,
                f64::try_from((ensemble.mean_dim(1, false, None) - value).mean(None)).unwrap(),
                f64::try_from(ensemble.var_dim(1i64, false, false).mean(None)).unwrap(),
            ));
            let (_, value, _, ensemble) = net.forward_t(&late_tensor, false);
            lines[2].push((
                step as f64,
                f64::try_from((ensemble.mean_dim(1, false, None) - value).mean(None)).unwrap(),
                f64::try_from(ensemble.var_dim(1i64, false, false).mean(None)).unwrap(),
            ));
        }
    }
    plot(lines);
}

fn plot(mut lines: [Vec<(f64, f64, f64)>; 3]) {
    let chart = Chart::new()
        .title(
            Title::new()
                .text("Ensemble Variance During Training")
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("Training steps"))
        .y_axis(Axis::new().name("Variance"))
        .grid(Grid::new())
        .legend(
            Legend::new()
                .data(vec!["Early (5 ply)", "Middle (30 ply)", "Late (<=120 ply)"])
                .bottom(10)
                .left(30),
        )
        .series(
            Line::new()
                .data(
                    std::mem::take(&mut lines[0])
                        .into_iter()
                        .map(|(s, _, v)| vec![s, v])
                        .collect(),
                )
                .name("Early (5 ply)")
                .symbol(Symbol::None),
        )
        .series(
            Line::new()
                .data(
                    std::mem::take(&mut lines[1])
                        .into_iter()
                        .map(|(s, _, v)| vec![s, v])
                        .collect(),
                )
                .name("Middle (30 ply)")
                .symbol(Symbol::None),
        )
        .series(
            Line::new()
                .data(
                    std::mem::take(&mut lines[2])
                        .into_iter()
                        .map(|(s, _, v)| vec![s, v])
                        .collect(),
                )
                .name("Late (<=120 ply)")
                .symbol(Symbol::None),
        );

    let mut renderer = HtmlRenderer::new("graph", 1400, 700).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}
