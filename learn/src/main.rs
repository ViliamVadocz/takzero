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
const STEPS_PER_EPOCH: usize = 200;
const EPOCHS_PER_EVALUATION: u32 = 5;
const LEARNING_RATE: f64 = 1e-4;
const STEPS_BEFORE_REANALYZE: usize = 1000;
const STEPS_PER_INTERACTION: usize = 1;
const EXPLOITATION_BUFFER_MAXIMUM_SIZE: usize = 20_000;

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

    let mut reanalyze_buffer = VecDeque::new();
    let mut exploitation_buffer = VecDeque::with_capacity(EXPLOITATION_BUFFER_MAXIMUM_SIZE);
    let mut total_interactions = steps / STEPS_PER_INTERACTION;
    // Track of how many interactions we have seen in selfplay.
    let mut interactions_since_last = 0;
    // Track how many targets we used from the reanalyze file.
    let mut targets_already_read = 0;

    // Initialize exploitation buffer with random games.
    {
        let mut actions = Vec::new();
        let mut states = Vec::new();
        while exploitation_buffer.len() < EXPLOITATION_BUFFER_MAXIMUM_SIZE / 2 {
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
                exploitation_buffer.push_front(Target {
                    env,
                    policy,
                    value: f32::from(value),
                    ube: 1.0,
                });
            }
        }
    }

    loop {
        // let mut before = net.clone(DEVICE);
        // before.vs_mut().freeze();
        // let before_steps = steps;

        for _ in 0..EPOCHS_PER_EVALUATION {
            for epoch_steps in 0..STEPS_PER_EPOCH {
                let using_reanalyze = steps >= STEPS_BEFORE_REANALYZE;

                // Make sure there are enough targets.
                loop {
                    if let Err(err) = fill_exploitation_buffer_with_targets(
                        &mut exploitation_buffer,
                        &args.directory,
                        steps,
                        &mut interactions_since_last,
                    ) {
                        log::error!("{err}");
                    }
                    if using_reanalyze {
                        if let Err(err) = get_reanalyze_targets(
                            &mut reanalyze_buffer,
                            &args.directory,
                            steps,
                            &mut targets_already_read,
                        ) {
                            log::error!("{err}");
                        }
                    }
                    let enough_interactions = (total_interactions + interactions_since_last)
                        * STEPS_PER_INTERACTION
                        > (steps + epoch_steps);
                    let enough_reanalyze =
                        !using_reanalyze || reanalyze_buffer.len() >= BATCH_SIZE / 2;
                    if enough_interactions && enough_reanalyze {
                        break;
                    }
                    let time = std::time::Duration::from_secs(10);
                    #[rustfmt::skip]
                    log::info!(
                        "Not enough interactions or targets yet.\n\
                        Training steps: {}\n\
                        Interactions: {}\n\
                        Exploitation buffer size: {}\n\
                        Reanalyze buffer size: {}\n\
                        Sleeping for {time:?}.",
                        steps + epoch_steps,
                        total_interactions + interactions_since_last,
                        exploitation_buffer.len(),
                        reanalyze_buffer.len()
                    );
                    std::thread::sleep(time);
                }

                // Create input and target tensors.
                let tensors = if using_reanalyze {
                    let mut targets = exploitation_buffer
                        .iter()
                        .choose_multiple(&mut rng, BATCH_SIZE / 2);
                    targets.extend(reanalyze_buffer.iter().take(BATCH_SIZE / 2));
                    let tensors = create_input_and_target_tensors(targets.into_iter(), &mut rng);
                    reanalyze_buffer.drain(..BATCH_SIZE / 2);
                    tensors
                } else {
                    create_input_and_target_tensors(
                        exploitation_buffer
                            .iter()
                            .choose_multiple(&mut rng, BATCH_SIZE)
                            .into_iter(),
                        &mut rng,
                    )
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
            total_interactions += interactions_since_last;
            interactions_since_last = 0;
            targets_already_read = 0;
        }

        // // Play some evaluation games.
        // let mut actions = Vec::new();
        // let games: [_; GAME_COUNT] =
        //     std::array::from_fn(|_| Env::new_opening(&mut rng, &mut
        // actions)); log::info!(
        //     "{before_steps} vs {steps}: {:?}",
        //     compete(&before, &net, &games)
        // );
        // log::info!(
        //     "{steps} vs {before_steps}: {:?}",
        //     compete(&net, &before, &games),
        // );
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

fn fill_exploitation_buffer_with_targets(
    buffer: &mut VecDeque<Target<Env>>,
    directory: &Path,
    model_steps: usize,
    interactions_since_last: &mut usize,
) -> std::io::Result<()> {
    let file_name = format!("targets-selfplay_{model_steps:0>6}.txt");
    let path = directory.join(file_name);

    let before = *interactions_since_last;
    for target in BufReader::new(OpenOptions::new().read(true).open(path)?)
        .lines()
        .skip(*interactions_since_last)
        .filter_map(|line| line.unwrap().parse().ok())
    {
        *interactions_since_last += 1;
        if buffer.len() >= EXPLOITATION_BUFFER_MAXIMUM_SIZE {
            buffer.truncate(EXPLOITATION_BUFFER_MAXIMUM_SIZE - 1);
        }
        buffer.push_front(target);
    }
    let added = *interactions_since_last - before;
    if added > 0 {
        log::debug!("Added {} targets to exploitation buffer.", added);
    }
    Ok(())
}

fn get_reanalyze_targets(
    buffer: &mut VecDeque<Target<Env>>,
    directory: &Path,
    model_steps: usize,
    targets_already_read: &mut usize,
) -> std::io::Result<()> {
    let file_name = format!("targets-reanalyze_{model_steps:0>6}.txt");
    let path = directory.join(file_name);

    buffer.extend(
        BufReader::new(OpenOptions::new().read(true).open(path)?)
            .lines()
            .skip(*targets_already_read)
            .filter_map(|line| line.unwrap().parse().ok())
            .map(|x| {
                *targets_already_read += 1;
                x
            }),
    );
    log::debug!("Reanalyze buffer now has {} targets.", buffer.len());
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
