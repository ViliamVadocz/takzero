use std::{
    cmp::Reverse,
    fmt,
    fs::{read_dir, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use clap::Parser;
use ordered_float::NotNan;
use rand::prelude::*;
use takzero::{
    network::{
        net4_simhash::{Env, Net, MAXIMUM_VARIANCE, N},
        repr::{game_to_tensor, move_mask, output_size, policy_tensor},
        HashNetwork,
        Network,
    },
    search::{agent::Agent, env::Environment, eval::Eval},
    target::{Augment, Target},
};
use tch::{
    nn::{Adam, Optimizer, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

// use crate::rnd_normalization::{reference_games, update_rnd};
// mod rnd_normalization;

// The environment to learn.
#[rustfmt::skip] #[allow(dead_code)]
const fn assert_env<E: Environment>() where Target<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const DEVICE: Device = Device::Cuda(0);
const BATCH_SIZE: usize = 128;
const STEPS_PER_SAVE: usize = 100;
const STEPS_PER_CHECKPOINT: usize = 50_000;
const LEARNING_RATE: f64 = 1e-4;

// Pre-training
const INITIAL_RANDOM_TARGETS: usize = BATCH_SIZE * 2_000;
const PRE_TRAINING_STEPS: usize = 1_000;
const _: () = assert!(INITIAL_RANDOM_TARGETS >= PRE_TRAINING_STEPS * BATCH_SIZE);

// Buffers
const STEPS_BEFORE_REANALYZE: usize = 5000;
const MIN_SELFPLAY_BUFFER_LEN: usize = 10_000;
const _: () = assert!(MIN_SELFPLAY_BUFFER_LEN >= BATCH_SIZE);
const MIN_REANALYZE_BUFFER_LEN: usize = 2_000;
const _: () = assert!(MIN_REANALYZE_BUFFER_LEN >= BATCH_SIZE);
const SELFPLAY_TARGET_FORCED_USES: u32 = 4;
const REANALYZE_TARGET_FORCED_USES: u32 = 4;
const MIN_TIME_BETWEEN_BUFFER_READS: Duration = Duration::from_secs(10);
const SLEEP_WHEN_NOT_ENOUGH_TARGETS: Duration = Duration::from_secs(30);

// Target
const MINIMUM_UBE_TARGET: f64 = -10.0;

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find targets
    /// and also where to save models.
    #[arg(long)]
    directory: PathBuf,
    /// Targets to use for resuming after restart.
    #[arg(long)]
    restart_targets: Option<PathBuf>,
}

struct TargetWithContext {
    /// The target.
    target: Target<Env>,
    /// How many uses are available until you cannot use this target.
    forced_uses: u32,
    /// The model steps at the time of loading this target.
    model_steps: usize,
}

impl TargetWithContext {
    fn reuse(mut self) -> Option<Self> {
        if self.forced_uses > 1 {
            self.forced_uses -= 1;
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

    let (mut net, mut starting_steps) =
        if let Some((resume_steps, path)) = get_model_path_with_most_steps(&args.directory) {
            log::info!("Resuming with model at {}", path.display());
            (
                Net::load(path, DEVICE).expect("Could not load network model"),
                resume_steps,
            )
        } else {
            // Initialize a network.
            log::info!("Initializing a network model");
            let net = Net::new(DEVICE, Some(rng.gen()));
            net.save(args.directory.join("model_0000000.ot")).unwrap();
            (net, 0)
        };

    let mut opt = Adam::default().build(net.vs_mut(), LEARNING_RATE).unwrap();
    // Load RND reference games.
    // let (early_reference, late_reference) = reference_games(DEVICE, &mut rng);

    if let Some(target_file) = &args.restart_targets {
        // Resuming after restarting.
        let mut targets = BufReader::new(OpenOptions::new().read(true).open(target_file).unwrap())
            .lines()
            .filter_map(|line| line.ok()?.parse::<Target<Env>>().ok())
            .collect::<Vec<_>>();
        targets.shuffle(&mut rng);
        for batch in targets.chunks_exact(BATCH_SIZE) {
            let tensors = create_input_and_target_tensors(batch.iter(), &mut rng);
            compute_loss_and_take_step(
                &mut net, &mut opt, tensors,
                // &early_reference,
                // &late_reference,
                false,
            );
            starting_steps += 1;
        }
        net.save(
            args.directory
                .join(format!("model_{starting_steps:0>7}.ot")),
        )
        .unwrap();
    } else if starting_steps == 0 {
        // Pre-training.
        pre_training(
            &mut net,
            &mut opt,
            &mut rng,
            &args.directory,
            // &early_reference,
            // &late_reference,
        );
        starting_steps += PRE_TRAINING_STEPS;
        net.save(
            args.directory
                .join(format!("model_{starting_steps:0>7}.ot")),
        )
        .unwrap();
    }

    net.save(args.directory.join("model_latest.ot")).unwrap();

    // Initialize buffers.
    let mut exploitation_buffer: Vec<TargetWithContext> =
        Vec::with_capacity(2 * MIN_SELFPLAY_BUFFER_LEN);
    let mut exploitation_targets_seek = 0;
    let mut reanalyze_buffer: Vec<TargetWithContext> = Vec::new();
    let mut reanalyze_targets_seek = 0;

    // Main training loop.
    let mut last_loaded = Instant::now();
    for model_steps in (starting_steps + 1).. {
        let using_reanalyze =
            args.restart_targets.is_some() || model_steps >= STEPS_BEFORE_REANALYZE;

        // Make sure there are enough targets before sampling a batch.
        loop {
            if last_loaded.elapsed() >= MIN_TIME_BETWEEN_BUFFER_READS {
                fill_buffers(
                    &mut exploitation_buffer,
                    &mut exploitation_targets_seek,
                    &mut reanalyze_buffer,
                    &mut reanalyze_targets_seek,
                    &args.directory,
                    model_steps,
                    using_reanalyze,
                );
                last_loaded = Instant::now();
                // Write buffer sizes to file for synchronization.
                if let Ok(mut file) = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .create(true)
                    .open(args.directory.join("buffer_lengths.txt"))
                {
                    if let Err(err) = file.write_fmt(format_args!(
                        "{},{},{}",
                        exploitation_buffer.len(),
                        reanalyze_buffer.len(),
                        exploitation_buffer.len() + reanalyze_buffer.len(),
                    )) {
                        log::error!("Writing buffer sizes to file: {err}");
                    }
                }
            }

            // Create a batch and take a step if there are enough targets.
            let enough_exploitation_targets = exploitation_buffer.len() >= MIN_SELFPLAY_BUFFER_LEN;
            let enough_reanalyze_targets =
                !using_reanalyze || reanalyze_buffer.len() >= MIN_REANALYZE_BUFFER_LEN;
            if enough_exploitation_targets && enough_reanalyze_targets {
                break;
            }

            #[rustfmt::skip]
            log::info!(
                "Not enough targets.\n\
                 Waiting {SLEEP_WHEN_NOT_ENOUGH_TARGETS:?}.\n\
                 Training steps: {model_steps}\n\
                 Exploitation buffer size: {}\n\
                 Reanalyze buffer size: {}",
                exploitation_buffer.len(),
                reanalyze_buffer.len()
            );
            std::thread::sleep(SLEEP_WHEN_NOT_ENOUGH_TARGETS);
        }

        let tensors = create_batch(
            using_reanalyze,
            &mut exploitation_buffer,
            &mut reanalyze_buffer,
            &mut rng,
        );
        compute_loss_and_take_step(
            &mut net, &mut opt, tensors,
            // &early_reference,
            // &late_reference,
            true,
        );

        // Save latest model.
        if model_steps % STEPS_PER_SAVE == 0 {
            #[rustfmt::skip]
                log::info!(
                    "Saving model.\n\
                     Training steps: {model_steps}\n\
                     Exploitation buffer size: {}\n\
                     Reanalyze buffer size: {}",
                    exploitation_buffer.len(),
                    reanalyze_buffer.len()
                );
            net.save(args.directory.join("model_latest.ot")).unwrap();
        }

        // Save checkpoint.
        if model_steps % STEPS_PER_CHECKPOINT == 0 {
            net.save(args.directory.join(format!("model_{model_steps:0>7}.ot")))
                .unwrap();
            // I don't know if this helps or hurts or does nothing.
            opt.zero_grad();
        }
    }
}

/// Get the path to the model file (ending with ".ot")
/// which has the highest number of steps (number after '_')
/// in the given directory.
fn get_model_path_with_most_steps(directory: &PathBuf) -> Option<(usize, PathBuf)> {
    read_dir(directory)
        .unwrap()
        .filter_map(|res| res.ok().map(|entry| entry.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "ot"))
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
    buffer: &mut Vec<TargetWithContext>,
    seek: &mut u64,
    file_path: &Path,
    forced_uses: u32,
    model_steps: usize,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(OpenOptions::new().read(true).open(file_path)?);
    reader
        .seek(std::io::SeekFrom::Start(*seek))
        .expect("Target file should not get shorter.");
    buffer.extend(
        reader
            .by_ref()
            .lines()
            .filter_map(|line| line.unwrap().parse().ok())
            .map(|target| TargetWithContext {
                target,
                forced_uses,
                model_steps,
            }),
    );
    *seek = reader
        .stream_position()
        .expect("Target file should not get shorter.");
    Ok(())
}

struct Tensors {
    input: Tensor,
    mask: Tensor,
    target_value: Tensor,
    target_policy: Tensor,
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
    let target_ube = Tensor::from_slice(&ube_targets)
        .unsqueeze(1)
        .to(DEVICE)
        .log()
        .clamp(MINIMUM_UBE_TARGET, MAXIMUM_VARIANCE.ln());

    Tensors {
        input,
        mask,
        target_value,
        target_policy,
        target_ube,
    }
}

fn compute_loss_and_take_step(
    net: &mut Net,
    opt: &mut Optimizer,
    tensors: Tensors,
    // early_reference: &Tensor,
    // late_reference: &Tensor,
    train_ube: bool,
) {
    // Get network output.
    let (policy, network_value, network_ube) = net.forward_t(&tensors.input, true);
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
    let loss_ube = if train_ube {
        (tensors.target_ube - network_ube)
            .square()
            .mean(Kind::Float)
    } else {
        // We don't want to train UBE in pre-training.
        Tensor::zeros_like(&loss_value)
    };
    // let loss_rnd = net.forward_rnd(&tensors.input, true).mean(Kind::Float);
    let loss = &loss_policy + &loss_value + &loss_ube; // + &loss_rnd;
    #[rustfmt::skip]
    log::info!(
        "loss = {loss:?}\n\
         loss_policy = {loss_policy:?}\n\
         loss_value = {loss_value:?}\n\
         loss_ube = {loss_ube:?}"
    );
    // loss_rnd = {loss_rnd:?}"

    // Update network RND min and max for normalization.
    // update_rnd(net, early_reference, late_reference);

    // Update hash counts
    net.update_counts(&tensors.input);

    // Take step.
    opt.backward_step(&loss);
}

fn pre_training(
    net: &mut Net,
    opt: &mut Optimizer,
    rng: &mut impl Rng,
    directory: &Path,
    // early_reference: &Tensor,
    // late_reference: &Tensor,
) {
    log::info!("Pre-training");
    let mut actions = Vec::new();
    let mut states = Vec::new();
    let mut buffer = Vec::with_capacity(INITIAL_RANDOM_TARGETS);
    while buffer.len() < INITIAL_RANDOM_TARGETS {
        let mut game = Env::new_opening(rng, &mut actions);
        // Play game until the end.
        while game.terminal().is_none() {
            states.push(game.clone());
            game.populate_actions(&mut actions);
            let action = actions.drain(..).choose(rng).unwrap();
            game.step(action);
        }
        // Create targets from the random game.
        let mut value = Eval::from(game.terminal().unwrap());
        for env in states.drain(..).rev() {
            env.populate_actions(&mut actions);
            // Uniform policy.
            let p = NotNan::new(1.0 / actions.len() as f32)
                .expect("there should always be at least one action");
            let policy = actions.drain(..).map(|a| (a, p)).collect();
            // Value is the discounted end of the game.
            value = value.negate();
            buffer.push(Target {
                env,
                policy,
                value: f32::from(value),
                ube: MAXIMUM_VARIANCE as f32 - f32::EPSILON,
            });
        }
    }
    buffer.shuffle(rng);
    // Save initial targets for inspection.
    let content: String = buffer.iter().map(ToString::to_string).collect();
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(directory.join("targets-initial.txt"))
        .unwrap()
        .write_all(content.as_bytes())
        .unwrap();

    for batch in buffer.chunks_exact(BATCH_SIZE).take(PRE_TRAINING_STEPS) {
        let tensors = create_input_and_target_tensors(batch.iter(), rng);
        compute_loss_and_take_step(
            net, opt, tensors, // early_reference, late_reference,
            false,
        );
    }
}

fn create_batch(
    using_reanalyze: bool,
    exploitation_buffer: &mut Vec<TargetWithContext>,
    reanalyze_buffer: &mut Vec<TargetWithContext>,
    rng: &mut impl Rng,
) -> Tensors {
    // TODO: Can we avoid doing an O(n) operation here?
    // Ideally we would like to sample without replacement,
    // Then swap_remove those targets which have forced_uses == 0.
    exploitation_buffer.shuffle(rng);
    reanalyze_buffer.shuffle(rng);

    if using_reanalyze {
        let batch: Vec<_> = exploitation_buffer
            .drain(exploitation_buffer.len() - BATCH_SIZE / 2..)
            .chain(reanalyze_buffer.drain(reanalyze_buffer.len() - BATCH_SIZE / 2..))
            .collect();
        let tensors = create_input_and_target_tensors(batch.iter().map(|t| &t.target), rng);
        let mut iter = batch.into_iter();
        exploitation_buffer.extend(
            iter.by_ref()
                .take(BATCH_SIZE / 2)
                .filter_map(TargetWithContext::reuse),
        );
        reanalyze_buffer.extend(iter.filter_map(TargetWithContext::reuse));
        return tensors;
    }

    let batch: Vec<_> = exploitation_buffer
        .drain(exploitation_buffer.len() - BATCH_SIZE..)
        .collect();
    let tensors = create_input_and_target_tensors(batch.iter().map(|t| &t.target), rng);
    exploitation_buffer.extend(batch.into_iter().filter_map(TargetWithContext::reuse));
    tensors
}

#[allow(unused)]
fn truncate_buffer_if_needed(buffer: &mut Vec<TargetWithContext>, max_length: usize, name: &str) {
    if buffer.len() > max_length {
        log::info!(
            "Truncating {name} buffer because it is too big. {}",
            buffer.len()
        );
        buffer.sort_unstable_by_key(|t| Reverse((t.model_steps, t.forced_uses)));
        buffer.truncate(max_length);
    }
}

fn fill_buffers(
    exploitation_buffer: &mut Vec<TargetWithContext>,
    exploitation_targets_seek: &mut u64,
    reanalyze_buffer: &mut Vec<TargetWithContext>,
    reanalyze_targets_seek: &mut u64,
    directory: &Path,
    model_steps: usize,
    using_reanalyze: bool,
) {
    let start = Instant::now();

    if let Err(error) = fill_buffer_with_targets(
        exploitation_buffer,
        exploitation_targets_seek,
        &directory.join("targets-selfplay.txt"),
        SELFPLAY_TARGET_FORCED_USES,
        model_steps,
    ) {
        log::error!("Cannot read selfplay targets: {error}");
    }

    if using_reanalyze {
        if let Err(error) = fill_buffer_with_targets(
            reanalyze_buffer,
            reanalyze_targets_seek,
            &directory.join("targets-reanalyze.txt"),
            REANALYZE_TARGET_FORCED_USES,
            model_steps,
        ) {
            log::error!("Cannot read reanalyze targets: {error}");
        }
    }

    log::debug!("It took {:?} to add targets to buffer.", start.elapsed());
}
