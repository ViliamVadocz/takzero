use std::{
    fmt,
    fs::{read_dir, OpenOptions},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use clap::Parser;
use fast_tak::Game;
use ordered_float::NotNan;
use rand::prelude::*;
use rayon::prelude::*;
use takzero::{
    network::{
        net4::Net4 as Net,
        repr::{game_to_tensor, move_mask, output_channels, output_size, policy_tensor},
        Network,
    },
    search::{
        agent::Agent,
        env::{Environment, Terminal},
        node::{gumbel::batched_simulate, Node},
    },
    target::{Augment, Replay, Target},
};
use tch::{
    nn::{Adam, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

// The environment to learn.
const N: usize = 4;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;
#[rustfmt::skip] #[allow(dead_code)]
const fn assert_env<E: Environment>() where Target<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const DEVICE: Device = Device::Cuda(0);
const MINIMUM_TARGETS_PER_EPOCH: usize = 10_000;
const BATCH_SIZE: usize = 128;
const STEPS_PER_EPOCH: u32 = 100;
const EPOCHS_PER_EVALUATION: u32 = 10;
const LEARNING_RATE: f64 = 1e-4;
const STEPS_BEFORE_REANALYZE: u32 = 500;

const GAME_COUNT: usize = 64;
const VISITS: usize = 400;
const BETA: [f32; GAME_COUNT] = [0.0; GAME_COUNT];
const MAX_MOVES: usize = 40;

#[derive(Parser, Debug)]
struct Args {
    /// Directory where to find targets
    /// and also where to save models.
    #[arg(long)]
    directory: PathBuf,
}

/// Get the path to the model file (ending with ".ot")
/// which has the highest number of steps (number after '_')
/// in the given directory.
fn get_model_path_with_most_steps(directory: &PathBuf) -> Option<(u32, PathBuf)> {
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
                    .parse::<u32>()
                    .ok()?,
                p,
            ))
        })
        .max_by_key(|(s, _)| *s)
}

/// Get a Vec of targets from `directory` for the current model step count.
/// It will return Some only if the amount it larger than the `minimum_amount`
fn get_targets(
    directory: &Path,
    model_steps: u32,
    minimum_amount: usize,
    self_play: bool,
) -> Option<Vec<Target<Env>>> {
    let file_name = if self_play {
        format!("targets-selfplay_{model_steps:0>6}.txt")
    } else {
        format!("targets-reanalyze_{model_steps:0>6}.txt")
    };
    let path = directory.join(file_name);
    let count = BufReader::new(OpenOptions::new().read(true).open(&path).ok()?)
        .lines()
        .count();
    if count < minimum_amount {
        log::debug!("Found {count} targets in the buffer.");
        return None;
    }
    Some(
        BufReader::new(OpenOptions::new().read(true).open(&path).ok()?)
            .lines()
            .filter_map(|line| line.unwrap().parse().ok())
            .collect(),
    )
}

fn wait_until_targets(directory: &Path, steps: u32) -> Vec<Target<Env>> {
    loop {
        let time = std::time::Duration::from_secs(30);
        log::info!("Sleeping for {time:?} before checking targets.");
        std::thread::sleep(time);

        // Sample only from fresh selfplay data.
        if steps < STEPS_BEFORE_REANALYZE {
            match get_targets(directory, steps, MINIMUM_TARGETS_PER_EPOCH, true) {
                None => continue,
                Some(targets) => break targets,
            }
        }
        // Half of the targets are from selfplay, half from reanalyze.
        if let Some(mut self_play_targets) =
            get_targets(directory, steps, MINIMUM_TARGETS_PER_EPOCH / 2, true)
        {
            if let Some(mut reanalyze_targets) =
                get_targets(directory, steps, MINIMUM_TARGETS_PER_EPOCH / 2, false)
            {
                self_play_targets.append(&mut reanalyze_targets);
                break self_play_targets;
            }
        }
    }
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
    let input = Tensor::cat(&inputs, 0);
    let mask = Tensor::cat(&masks, 0);
    // Get the target.
    let target_policy = Tensor::stack(&policy_targets, 0).view([
        BATCH_SIZE as i64,
        output_channels::<N>() as i64,
        N as i64,
        N as i64,
    ]);
    let target_value = Tensor::from_slice(&value_targets).unsqueeze(1);
    let target_ube = Tensor::from_slice(&ube_targets).unsqueeze(1);

    Tensors {
        input,
        mask,
        target_value,
        target_policy,
        target_ube,
    }
}

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

    loop {
        let mut before = net.clone(DEVICE);
        before.vs_mut().freeze();
        let before_steps = steps;

        for _ in 0..EPOCHS_PER_EVALUATION {
            let targets = wait_until_targets(&args.directory, steps);
            log::info!("Target buffer contains {} targets", targets.len());

            for _ in 0..STEPS_PER_EPOCH {
                let batch = targets.choose_multiple(&mut rng, BATCH_SIZE);
                let tensors = create_input_and_target_tensors(batch, &mut rng);

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
        }

        // Play some evaluation games.
        let mut actions = Vec::new();
        let games: [_; GAME_COUNT] =
            std::array::from_fn(|_| Env::new_opening(&mut rng, &mut actions));
        log::info!(
            "{before_steps} vs {steps}: {:?}",
            compete(&before, &net, &games)
        );
        log::info!(
            "{steps} vs {before_steps}: {:?}",
            compete(&net, &before, &games),
        );
    }
}

/// Pit two networks against each other in the given games. Evaluation is from
/// the perspective of white.
fn compete(white: &Net, black: &Net, games: &[Env]) -> Evaluation {
    let mut evaluation = Evaluation::default();

    let mut games = games.to_owned();
    let mut white_nodes: Vec<_> = (0..BATCH_SIZE).map(|_| Node::default()).collect();
    let mut black_nodes: Vec<_> = (0..BATCH_SIZE).map(|_| Node::default()).collect();
    let mut context = <Net as Agent<Env>>::Context::new(0.0);

    let mut actions: Vec<_> = (0..BATCH_SIZE).map(|_| Vec::new()).collect();
    let mut trajectories: Vec<_> = (0..BATCH_SIZE).map(|_| Vec::new()).collect();

    let mut game_replays: Vec<_> = games.iter().cloned().map(Replay::new).collect();

    let mut done = [false; BATCH_SIZE];

    'outer: for _ in 0..MAX_MOVES {
        for (agent, is_white) in [(white, true), (black, false)] {
            if done.iter().all(|x| *x) {
                break 'outer;
            }

            for _ in 0..VISITS {
                batched_simulate(
                    if is_white {
                        &mut white_nodes
                    } else {
                        &mut black_nodes
                    },
                    &games,
                    agent,
                    &BETA,
                    &mut context,
                    &mut actions,
                    &mut trajectories,
                );
            }
            let top_actions = if is_white { &white_nodes } else { &black_nodes }
                .par_iter()
                .map(|node| {
                    node.children
                        .iter()
                        .max_by_key(|(_, child)| {
                            child
                                .evaluation
                                .negate()
                                .map(|_| NotNan::new(child.visit_count as f32).unwrap())
                        })
                        .map(|(a, _)| *a)
                })
                .collect::<Vec<_>>();

            let terminals: Vec<_> = top_actions
                .into_par_iter()
                .zip(games.par_iter_mut())
                .zip(white_nodes.par_iter_mut())
                .zip(black_nodes.par_iter_mut())
                .zip(game_replays.par_iter_mut())
                .zip(done.par_iter_mut())
                .filter(|(_, done)| !**done)
                .filter_map(
                    |(((((action, game), white_node), black_node), replay), done)| {
                        let action = action.unwrap();
                        game.step(action);
                        replay.push(action);

                        if let Some(terminal) = game.terminal() {
                            *game = Env::default();
                            *white_node = Node::default();
                            *black_node = Node::default();
                            *done = true;
                            log::debug!("{replay}");
                            Some(terminal)
                        } else {
                            white_node.descend(&action);
                            black_node.descend(&action);
                            None
                        }
                    },
                )
                .collect();

            for terminal in terminals {
                // This may seem opposite of what is should be.
                // That is because we are looking at the terminal after a move was made, so a
                // loss for the "current player" is actually a win for the one who just played
                match (terminal, is_white) {
                    (Terminal::Loss, true) | (Terminal::Win, false) => evaluation.wins += 1,
                    (Terminal::Win, true) | (Terminal::Loss, false) => evaluation.losses += 1,
                    (Terminal::Draw, _) => evaluation.draws += 1,
                }
            }
        }
    }
    evaluation
}

use std::{iter::Sum, ops::AddAssign};

#[derive(Debug, Default)]
pub struct Evaluation {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl AddAssign for Evaluation {
    fn add_assign(&mut self, rhs: Self) {
        self.wins += rhs.wins;
        self.losses += rhs.losses;
        self.draws += rhs.draws;
    }
}

impl Sum for Evaluation {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut a, b| {
            a += b;
            a
        })
    }
}
