use std::{
    cmp::Reverse,
    fmt,
    fs::{read_dir, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use clap::Parser;
use ordered_float::NotNan;
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
const STEPS_PER_EPOCH: usize = 1000;
const LEARNING_RATE: f64 = 1e-4;
const STEPS_BEFORE_REANALYZE: usize = 5000;
const MIN_EXPLOITATION_BUFFER_LEN: usize = 10_000;
const MAX_EXPLOITATION_BUFFER_LEN: usize = 100_000;
const MAX_REANALYZE_BUFFER_LEN: usize = 10_000;
const INITIAL_RANDOM_TARGETS: usize = MIN_EXPLOITATION_BUFFER_LEN + BATCH_SIZE * STEPS_PER_EPOCH;
const EXPLOITATION_TARGET_LIFETIME: u32 = 4;
const REANALYZE_TARGET_LIFETIME: u32 = 4;

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find targets
    /// and also where to save models.
    #[arg(long)]
    directory: PathBuf,
}

struct TargetWithLifeTime {
    target: Target<Env>,
    lifetime: u32,
}
impl TargetWithLifeTime {
    fn tap(mut self) -> Option<Self> {
        if self.lifetime > 1 {
            self.lifetime -= 1;
            Some(self)
        } else {
            None
        }
    }
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
            let net = Net::new(DEVICE, Some(rng.gen()));
            net.save(args.directory.join("model_000000.ot")).unwrap();
            (net, 0)
        };

    let mut opt = Adam::default().build(net.vs_mut(), LEARNING_RATE).unwrap();

    let mut exploitation_buffer = Vec::new();
    let mut exploitation_targets_read = 0;
    let mut reanalyze_buffer = Vec::new();
    let mut reanalyze_targets_read = 0;

    // Initialize exploitation buffer with random games.
    if steps == 0 {
        let mut actions = Vec::new();
        let mut states = Vec::new();
        while exploitation_buffer.len() < INITIAL_RANDOM_TARGETS {
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
                let p = NotNan::new(1.0 / actions.len() as f32)
                    .expect("there should always be at least one action");
                let policy = actions.drain(..).map(|a| (a, p)).collect();
                value = value.negate();
                exploitation_buffer.push(TargetWithLifeTime {
                    target: Target {
                        env,
                        policy,
                        value: f32::from(value),
                        ube: 1.0,
                    },
                    lifetime: 1,
                });
            }
        }
        // Save initial targets for inspection.
        let content: String = exploitation_buffer
            .iter()
            .map(|t| &t.target)
            .map(ToString::to_string)
            .collect();
        OpenOptions::new()
            .write(true)
            .create(true)
            .open(args.directory.join("targets-initial.txt"))
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
    }

    loop {
        for epoch_steps in 0..STEPS_PER_EPOCH {
            let using_reanalyze = steps >= STEPS_BEFORE_REANALYZE;

            // Make sure there are enough targets.
            loop {
                let _ = fill_buffer_with_targets(
                    &mut exploitation_buffer,
                    &mut exploitation_targets_read,
                    &args
                        .directory
                        .join(format!("targets-selfplay_{steps:0>6}.txt")),
                    EXPLOITATION_TARGET_LIFETIME,
                );
                if steps > 0 && exploitation_buffer.len() > MAX_EXPLOITATION_BUFFER_LEN {
                    log::debug!("Truncating exploitation buffer because it's too long");
                    exploitation_buffer.sort_unstable_by_key(|t| Reverse(t.lifetime));
                    exploitation_buffer.truncate(MAX_EXPLOITATION_BUFFER_LEN);
                }
                if using_reanalyze {
                    let _ = fill_buffer_with_targets(
                        &mut reanalyze_buffer,
                        &mut reanalyze_targets_read,
                        &args
                            .directory
                            .join(format!("targets-reanalyze_{steps:0>6}.txt")),
                        REANALYZE_TARGET_LIFETIME,
                    );
                    if exploitation_buffer.len() > MAX_REANALYZE_BUFFER_LEN {
                        log::debug!("Truncating reanalyze buffer because it's too long");
                        exploitation_buffer.sort_unstable_by_key(|t| Reverse(t.lifetime));
                        exploitation_buffer.truncate(MAX_REANALYZE_BUFFER_LEN);
                    }
                }
                let enough_in_exploitation =
                    exploitation_buffer.len() >= MIN_EXPLOITATION_BUFFER_LEN;
                let enough_for_combined_batch = exploitation_buffer.len() >= BATCH_SIZE / 2
                    && reanalyze_buffer.len() >= BATCH_SIZE / 2;
                if enough_in_exploitation
                    && ((using_reanalyze && enough_for_combined_batch)
                        || (!using_reanalyze && exploitation_buffer.len() >= BATCH_SIZE))
                {
                    break;
                }
                let time = std::time::Duration::from_secs(30);
                #[rustfmt::skip]
                    log::info!(
                        "Not enough targets.\n\
                        Training steps: {}\n\
                        Exploitation buffer size: {}\n\
                        Reanalyze buffer size: {}\n\
                        Sleeping for {time:?}.",
                        steps + epoch_steps,
                        exploitation_buffer.len(),
                        reanalyze_buffer.len()
                    );
                std::thread::sleep(time);
            }

            // Create input and target tensors.
            exploitation_buffer.shuffle(&mut rng);
            reanalyze_buffer.shuffle(&mut rng);
            let tensors = if using_reanalyze {
                let batch: Vec<_> = exploitation_buffer
                    .drain(exploitation_buffer.len() - BATCH_SIZE / 2..)
                    .chain(reanalyze_buffer.drain(reanalyze_buffer.len() - BATCH_SIZE / 2..))
                    .collect();
                let tensors =
                    create_input_and_target_tensors(batch.iter().map(|t| &t.target), &mut rng);
                let mut iter = batch.into_iter();
                exploitation_buffer.extend(
                    iter.by_ref()
                        .take(BATCH_SIZE / 2)
                        .filter_map(TargetWithLifeTime::tap),
                );
                reanalyze_buffer.extend(iter.filter_map(TargetWithLifeTime::tap));
                tensors
            } else {
                let batch: Vec<_> = exploitation_buffer
                    .drain(exploitation_buffer.len() - BATCH_SIZE..)
                    .collect();
                let tensors =
                    create_input_and_target_tensors(batch.iter().map(|t| &t.target), &mut rng);
                exploitation_buffer.extend(batch.into_iter().filter_map(TargetWithLifeTime::tap));
                tensors
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

        steps += STEPS_PER_EPOCH;
        net.save(args.directory.join(format!("model_{steps:0>6}.ot")))
            .unwrap();
        exploitation_targets_read = 0;
        reanalyze_targets_read = 0;

        #[rustfmt::skip]
        log::info!(
            "Saving model (end of epoch)\n\
             Training steps: {}\n\
             Exploitation buffer size: {}\n\
             Reanalyze buffer size: {}",
            steps,
            exploitation_buffer.len(),
            reanalyze_buffer.len()
        );
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

/// Add targets to the buffer from the given file, skipping the targets that
/// have already been read.
fn fill_buffer_with_targets(
    buffer: &mut Vec<TargetWithLifeTime>,
    targets_already_read: &mut usize,
    file_path: &Path,
    lifetime: u32,
) -> std::io::Result<()> {
    let before = buffer.len();
    buffer.extend(
        BufReader::new(OpenOptions::new().read(true).open(file_path)?)
            .lines()
            .skip(*targets_already_read)
            .filter_map(|line| line.unwrap().parse().ok())
            .map(|target| TargetWithLifeTime { target, lifetime }),
    );
    *targets_already_read += buffer.len() - before;
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

fn create_input_and_target_tensors<'a>(
    batch: impl Iterator<Item = &'a Target<Env>>,
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
