use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader},
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
        net4_ensemble::{Env, Net, N},
        repr::{game_to_tensor, move_mask, output_size, policy_tensor},
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
        .filter_map(|line| line.unwrap().parse::<Target<Env>>().ok())
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

struct Tensors {
    input: Tensor,
    mask: Tensor,
    target_value: Tensor,
    target_policy: Tensor,
}

fn create_input_and_target_tensors(
    batch: impl Iterator<Item = Target<Env>>,
    rng: &mut impl Rng,
) -> Tensors {
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
    let input = Tensor::cat(&inputs, 0).to(DEVICE);
    let mask = Tensor::cat(&masks, 0).to(DEVICE);
    // Get the target.
    let target_policy = Tensor::stack(&policy_targets, 0)
        .view([BATCH_SIZE as i64, output_size::<N>() as i64])
        .to(DEVICE);
    let target_value = Tensor::from_slice(&value_targets).unsqueeze(1).to(DEVICE);

    Tensors {
        input,
        mask,
        target_value,
        target_policy,
    }
}

fn compute_loss_and_take_step(net: &Net, opt: &mut Optimizer, tensors: Tensors) {
    // Get network output.
    let (policy, network_value, _, ensemble) = net.forward_t(&tensors.input, true);
    let log_softmax_network_policy = policy
        .masked_fill(&tensors.mask, f64::from(f32::MIN))
        .view([-1, output_size::<N>() as i64])
        .log_softmax(1, Kind::Float);

    // Calculate loss.
    let loss_policy = -(log_softmax_network_policy * &tensors.target_policy).sum(Kind::Float)
        / i64::try_from(BATCH_SIZE).unwrap();
    let loss_value = (tensors.target_value - network_value)
        .square()
        .mean(Kind::Float);

    let loss = &loss_policy + &loss_value;
    #[rustfmt::skip]
    log::info!(
        "loss = {loss:?}\n\
         loss_policy = {loss_policy:?}\n\
         loss_value = {loss_value:?}"
    );

    // Take step.
    opt.backward_step(&loss);

    // TODO: Update ensemble
    todo!("{:?}", ensemble);
}

fn main() {
    let mut selfplay = targets("targets-selfplay-no-explore-01.txt").skip(BATCH_SIZE * 15_000);
    let mut reanalyze = targets("targets-reanalyze-no-explore-01.txt").skip(BATCH_SIZE * 10_000);

    let mut buffer = Vec::new();
    buffer.extend(selfplay.by_ref().take(5_000));
    buffer.extend(reanalyze.by_ref().take(5_000));

    let seed: u64 = 12345;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(Device::Cuda(0), Some(rng.gen()));
    let mut opt = Adam::default().build(net.vs_mut(), LEARNING_RATE).unwrap();

    // TODO: Create reference positions.
    let mut actions = Vec::new();
    let _early_positions: Vec<_> = (0..64)
        .map(|_| random_env(5, &mut actions, &mut rng))
        .collect();
    let _mid_positions: Vec<_> = (0..64)
        .map(|_| random_env(50, &mut actions, &mut rng))
        .collect();
    let _late_positions: Vec<_> = (0..64)
        .map(|_| random_env(120, &mut actions, &mut rng))
        .collect();

    for step in 1..=STEPS {
        print!("{step} / {STEPS}");
        buffer.extend(selfplay.by_ref().take(BATCH_SIZE / 2));
        buffer.extend(reanalyze.by_ref().take(BATCH_SIZE / 2));
        buffer.shuffle(&mut rng);

        let tensors =
            create_input_and_target_tensors(buffer.drain(buffer.len() - BATCH_SIZE..), &mut rng);
        compute_loss_and_take_step(&net, &mut opt, tensors);

        // TODO:
        // Keep track of variance and values on reference positions
        // - early, middle, late
        // - make many so that I can then pick out the interesting positions
        // Make graphs
    }
}
