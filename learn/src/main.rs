use std::{
    collections::VecDeque,
    fmt,
    fs::{read_dir, OpenOptions},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use clap::Parser;
use rand::prelude::*;
use takzero::{
    network::{
        net5::{Env, Net, N},
        repr::{game_to_tensor, move_mask, output_size, policy_tensor},
        Network,
    },
    search::{agent::Agent, env::Environment, eval::Eval},
    target::{Augment, Target},
};
use tch::{
    nn::{Adam, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

// The environment to learn.
#[rustfmt::skip] #[allow(dead_code)]
const fn assert_env<E: Environment>() where Target<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const DEVICE: Device = Device::Cuda(0);
const BATCH_SIZE: usize = 128;
const STEPS_PER_EPOCH: usize = 100;
const LEARNING_RATE: f64 = 1e-4;
const STEPS_BEFORE_REANALYZE: usize = 2000;
const INTIAL_RANDOM_TARGETS: usize = BATCH_SIZE * STEPS_PER_EPOCH;

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find targets
    /// and also where to save models.
    #[arg(long)]
    directory: PathBuf,
}

#[allow(clippy::too_many_lines)]
fn main() {
    env_logger::init();
    let args = Args::parse();

    let seed: u64 = rand::thread_rng().gen();
    log::info!("seed = {seed}");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let (mut net, mut steps) =
        if let Some((steps, model_path)) = get_model_path_with_most_steps(&args.directory) {
            log::info!("Resuming at {steps} steps with {}", model_path.display());
            (
                Net::load(model_path, DEVICE).expect("Model file should be loadable"),
                steps,
            )
        } else {
            log::info!("Creating new model");
            (Net::new(DEVICE, Some(rng.gen())), 0)
        };

    let mut opt = Adam::default().build(net.vs_mut(), LEARNING_RATE).unwrap();

    let mut exploitation_queue = VecDeque::new();
    let mut exploitation_targets_read = 0;
    let mut reanalyze_queue = VecDeque::new();
    let mut reanalyze_targets_read = 0;

    // Initialize exploitation buffer with random games.
    {
        let mut actions = Vec::new();
        let mut states = Vec::new();
        while exploitation_queue.len() < INTIAL_RANDOM_TARGETS {
            let mut game = Env::new_opening(&mut rng, &mut actions);
            while game.terminal().is_none() {
                states.push(game.clone());
                game.populate_actions(&mut actions);
                let action = actions.drain(..).choose(&mut rng).unwrap();
                game.step(action);
            }
            let mut value = Eval::from(game.terminal().unwrap());
            for env in states.drain(..).rev() {
                env.populate_actions(&mut actions);
                let p = 1.0 / actions.len() as f32;
                let policy = actions.drain(..).map(|a| (a, p)).collect();
                value = value.negate();
                exploitation_queue.push_back(Target {
                    env,
                    policy,
                    value: f32::from(value),
                    ube: 1.0,
                });
            }
        }
    }

    loop {
        for epoch_steps in 0..STEPS_PER_EPOCH {
            let using_reanalyze = steps >= STEPS_BEFORE_REANALYZE;

            // Make sure there are enough targets.
            loop {
                if let Err(err) = fill_queue_with_targets(
                    &mut exploitation_queue,
                    &mut exploitation_targets_read,
                    &args
                        .directory
                        .join(format!("targets-selfplay_{steps:0>6}.txt")),
                ) {
                    log::error!("Cannot fill exploitation queue: {err}");
                }
                if using_reanalyze {
                    if let Err(err) = fill_queue_with_targets(
                        &mut reanalyze_queue,
                        &mut reanalyze_targets_read,
                        &args
                            .directory
                            .join(format!("targets-reanalyze_{steps:0>6}.txt")),
                    ) {
                        log::error!("Cannot fill reanalyze queue: {err}");
                    }
                }

                let enough_for_combined_batch = exploitation_queue.len() >= BATCH_SIZE / 2
                    && reanalyze_queue.len() >= BATCH_SIZE / 2;
                if (using_reanalyze && enough_for_combined_batch)
                    || (!using_reanalyze && exploitation_queue.len() >= BATCH_SIZE)
                {
                    break;
                }
                let time = std::time::Duration::from_secs(30);
                #[rustfmt::skip]
                    log::info!(
                        "Not enough targets.\n\
                        Training steps: {}\n\
                        Exploitation queue size: {}\n\
                        Reanalyze queue size: {}\n\
                        Sleeping for {time:?}.",
                        steps + epoch_steps,
                        exploitation_queue.len(),
                        reanalyze_queue.len()
                    );
                std::thread::sleep(time);
            }

            // Create input and target tensors.
            let tensors = if using_reanalyze {
                create_input_and_target_tensors(
                    exploitation_queue
                        .drain(..BATCH_SIZE / 2)
                        .chain(reanalyze_queue.drain(..BATCH_SIZE / 2)),
                    &mut rng,
                )
            } else {
                create_input_and_target_tensors(exploitation_queue.drain(..BATCH_SIZE), &mut rng)
            };

            // Get network output.
            let (policy, network_value, _network_ube) = net.forward_t(&tensors.input, true);
            let log_softmax_network_policy = policy
                .masked_fill(&tensors.mask, f64::from(f32::MIN))
                .view([-1, output_size::<N>() as i64])
                .log_softmax(1, Kind::Float);

            // Calculate loss.
            let loss_policy = -(log_softmax_network_policy * &tensors.target_policy)
                .sum(Kind::Float)
                / i64::try_from(BATCH_SIZE).unwrap();
            let loss_value = (tensors.target_value - network_value)
                .square()
                .mean(Kind::Float);
            // TODO: Add UBE back later.
            // let loss_ube = (target_ube - network_ube).square().mean(Kind::Float);
            let loss = &loss_policy + &loss_value; //+ &loss_ube;
            log::info!(
                "loss = {loss:?}, loss_policy = {loss_policy:?}, loss_value = {loss_value:?}"
            );

            // Take step.
            opt.backward_step(&loss);
        }
        opt.zero_grad();

        log::info!("Saving model (end of epoch).");
        steps += STEPS_PER_EPOCH;
        net.save(args.directory.join(format!("model_{steps:0>6}.ot")))
            .unwrap();
        exploitation_targets_read = 0;
        reanalyze_targets_read = 0;
    }
}

/// Get the path to the model file (ending with ".ot")
/// which has the highest number of steps (number after '_')
/// in the given directory.
fn get_model_path_with_most_steps(directory: &PathBuf) -> Option<(usize, PathBuf)> {
    read_dir(directory)
        .unwrap()
        .filter_map(|res| res.ok().map(|entry| entry.path()))
        .filter(|p| p.extension().map(|ext| ext == "ot").unwrap_or_default())
        .filter_map(|p| {
            Some((
                p.file_stem()?
                    .to_str()?
                    .split_once('_')?
                    .1
                    .parse::<usize>()
                    .ok()?,
                p,
            ))
        })
        .max_by_key(|(s, _)| *s)
}

/// Add targets to queue from the given file, skipping the targets that have
/// already been read.
fn fill_queue_with_targets(
    queue: &mut VecDeque<Target<Env>>,
    targets_already_read: &mut usize,
    file_path: &Path,
) -> std::io::Result<()> {
    let before = queue.len();
    queue.extend(
        BufReader::new(OpenOptions::new().read(true).open(file_path)?)
            .lines()
            .skip(*targets_already_read)
            .filter_map(|line| line.unwrap().parse().ok()),
    );
    *targets_already_read += queue.len() - before;
    Ok(())
}

struct Tensors {
    input: Tensor,
    mask: Tensor,
    target_value: Tensor,
    target_policy: Tensor,
    #[allow(dead_code)]
    target_ube: Tensor,
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
    let mut ube_targets = Vec::with_capacity(BATCH_SIZE);
    for target in batch {
        let target = target.augment(rng);
        inputs.push(game_to_tensor(&target.env, DEVICE));
        policy_targets.push(policy_tensor::<N>(&target.policy, DEVICE));
        masks.push(move_mask::<N>(
            &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
            DEVICE,
        ));
        value_targets.push(target.value);
        ube_targets.push(target.ube);
    }

    // Get network output.
    let input = Tensor::cat(&inputs, 0).to(DEVICE);
    let mask = Tensor::cat(&masks, 0).to(DEVICE);
    // Get the target.
    let target_policy = Tensor::stack(&policy_targets, 0)
        .view([BATCH_SIZE as i64, output_size::<N>() as i64])
        .to(DEVICE);
    let target_value = Tensor::from_slice(&value_targets).unsqueeze(1).to(DEVICE);
    let target_ube = Tensor::from_slice(&ube_targets).unsqueeze(1).to(DEVICE);

    Tensors {
        input,
        mask,
        target_value,
        target_policy,
        target_ube,
    }
}
