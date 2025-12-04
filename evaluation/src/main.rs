#![warn(clippy::pedantic, clippy::style)]

use std::{
    array,
    fs::{read_dir, OpenOptions},
    io::{BufRead, BufReader},
    path::PathBuf,
};

use clap::Parser;
use fast_tak::takparse::Tps;
use rand::{prelude::*, rngs::StdRng, Rng, SeedableRng};
use takzero::{
    network::{
        net4_simhash::{Env, Net},
        Network,
    },
    search::{
        agent::Agent,
        env::{Environment, Terminal},
        node::{batched::BatchedMCTS, Node},
    },
};
use tch::Device;

const DEVICE: Device = Device::Cuda(0);

const BATCH_SIZE: usize = 64;
const MAX_MOVES: usize = 200;
const SAMPLED_ACTIONS: usize = 64;
const SEARCH_BUDGET: u32 = 768;

#[derive(Parser, Debug)]
struct Args {
    /// Path to models
    #[arg(long)]
    model_path: PathBuf,
    /// How many models to skip when creating match-ups
    #[arg(long, default_value_t = 1)]
    step: usize,
    /// Path to starting positions
    #[arg(long)]
    opening_book: Option<PathBuf>,
}

// #[allow(unused)]
// fn compare_mid_big(
//     path_1: impl AsRef<std::path::Path>,
//     path_2: impl AsRef<std::path::Path>,
//     games: &[Env],
//     rng: &mut impl Rng,
// ) -> Evaluation {
//     let big_1 = path_1
//         .as_ref()
//         .file_name()
//         .unwrap()
//         .to_string_lossy()
//         .split_once('_')
//         .is_some_and(|(f, _)| f == "big");
//     let big_2 = path_2
//         .as_ref()
//         .file_name()
//         .unwrap()
//         .to_string_lossy()
//         .split_once('_')
//         .is_some_and(|(f, _)| f == "big");

//     match (big_1, big_2) {
//         (true, true) => {
//             let a = net4_big::Net::load(path_1, DEVICE).unwrap();
//             let b = net4_big::Net::load(path_2, DEVICE).unwrap();
//             compete(&a, &b, games, rng)
//         }
//         (true, false) => {
//             let a = net4_big::Net::load(path_1, DEVICE).unwrap();
//             let b = net4_rnd::Net::load(path_2, DEVICE).unwrap();
//             compete(&a, &b, games, rng)
//         }
//         (false, true) => {
//             let a = net4_rnd::Net::load(path_1, DEVICE).unwrap();
//             let b = net4_big::Net::load(path_2, DEVICE).unwrap();
//             compete(&a, &b, games, rng)
//         }
//         (false, false) => {
//             let a = net4_rnd::Net::load(path_1, DEVICE).unwrap();
//             let b = net4_rnd::Net::load(path_2, DEVICE).unwrap();
//             compete(&a, &b, games, rng)
//         }
//     }
// }

// fn negative_beta_range_test() {
//     const BETA: [f32; 7] = [0.25, 0.0, -0.01, -0.1, -0.25, -0.5, -1.0];
//     log::info!("negative beta experiments with BETA = {BETA:?}");

//     let args = Args::parse();
//     let seed: u64 = thread_rng().gen();
//     log::info!("seed: {seed}");
//     let mut rng = StdRng::seed_from_u64(seed);

//     let net = Net::load(args.model_path, DEVICE).expect("model path should be
// valid");

//     loop {
//         let mut betas = BETA.choose_multiple(&mut rng, 2);
//         let beta_1 = *betas.next().expect("Exactly two betas should be
// chosen");         let beta_2 = *betas.next().expect("Exactly two betas should
// be chosen");

//         let mut actions = Vec::new();
//         let games: [Env; BATCH_SIZE] = array::from_fn(|_| {
//             let steps = rng.gen_range(2..=3);
//             Env::new_opening_with_random_steps(&mut rng, &mut actions, steps)
//         });

//         let a_as_white = compete(&net, &net, beta_1, beta_2, &games, &mut
// rng);         log::info!(
//             "{beta_1} vs. {beta_2}: {a_as_white:?} {:.1}%",
//             a_as_white.win_rate() * 100.0
//         );
//         let b_as_white = compete(&net, &net, beta_2, beta_1, &games, &mut
// rng);         log::info!(
//             "{beta_2} vs. {beta_1}: {b_as_white:?} {:.1}%",
//             b_as_white.win_rate() * 100.0
//         );
//     }
// }

fn main() {
    env_logger::init();
    log::info!("Begin.");
    tch::no_grad(real_main);
}

#[allow(unused)]
fn real_main() {
    let args = Args::parse();
    let seed: u64 = rand::rng().random();
    log::info!("seed: {seed}");
    let mut rng = StdRng::seed_from_u64(seed);

    let opening_book: Option<Vec<Env>> = args.opening_book.map(|path| {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path)
            .expect("Path to opening book should be valid");
        BufReader::new(file)
            .lines()
            .map(|line| {
                line.expect("Line should be fine to read")
                    .parse()
                    .map(|tps: Tps| tps.into())
            })
            .collect::<Result<_, _>>()
            .expect("Opening book should be valid TPS, one per line")
    });

    loop {
        let mut paths: Vec<_> = read_dir(&args.model_path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension().is_some_and(|ext| ext == "ot")
                    && path.file_stem().is_some_and(|stem| stem != "model_latest")
            })
            .collect();
        paths.sort();
        let paths: Vec<_> = paths.into_iter().step_by(args.step).collect();
        if paths.len() < 2 {
            let time = std::time::Duration::from_secs(600);
            log::info!("Too few models. Sleeping for {time:?}.");
            std::thread::sleep(time);
            continue;
        }

        let mut match_up = paths.sample(&mut rng, 2);
        let path_a = match_up.next().unwrap();
        let path_b = match_up.next().unwrap();

        let Ok(a) = Net::load_partial(path_a, DEVICE) else {
            log::warn!("Cannot load {}", path_a.display());
            continue;
        };
        let Ok(b) = Net::load_partial(path_b, DEVICE) else {
            log::warn!("Cannot load {}", path_b.display());
            continue;
        };
        let name_a = path_a.file_name().unwrap().to_string_lossy().to_string();
        let name_b = path_b.file_name().unwrap().to_string_lossy().to_string();

        let games: [Env; BATCH_SIZE] = if let Some(openings) = &opening_book {
            openings
                .sample(&mut rng, BATCH_SIZE)
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .expect("There should be enough games in the opening book to form a unique batch")
        } else {
            let mut actions = Vec::new();

            array::from_fn(|_| {
                let steps = rng.random_range(2..=3);
                Env::new_opening_with_random_steps(&mut rng, &mut actions, steps)
            })
        };

        let a_as_white = compete(&a, &b, 0.0, 0.0, &games, &mut rng);
        // let a_as_white = compare_mid_big(path_a, path_b, &games, &mut rng);
        log::info!(
            "{name_a} vs. {name_b}: {a_as_white:?} {:.1}%",
            a_as_white.win_rate() * 100.0
        );
        let b_as_white = compete(&b, &a, 0.0, 0.0, &games, &mut rng);
        // let b_as_white = compare_mid_big(path_b, path_a, &games, &mut rng);
        log::info!(
            "{name_b} vs. {name_a}: {b_as_white:?} {:.1}%",
            b_as_white.win_rate() * 100.0
        );
    }
}
/// Pit two networks against each other in the given games. Evaluation is from
/// the perspective of white.
#[allow(dead_code)]
fn compete<W, B>(
    white: &W,
    black: &B,
    white_beta: f32,
    black_beta: f32,
    games: &[Env],
    rng: &mut impl Rng,
) -> Evaluation
where
    W: Network + Agent<Env>,
    B: Network + Agent<Env>,
{
    let mut evaluation = Evaluation::default();

    let mut white_mcts = BatchedMCTS::from_envs(games.to_owned().try_into().unwrap());
    let mut black_mcts = BatchedMCTS::from_envs(games.to_owned().try_into().unwrap());
    let white_beta = [white_beta; BATCH_SIZE];
    let black_beta = [black_beta; BATCH_SIZE];

    let mut done = [false; BATCH_SIZE];

    'outer: for _ in 0..MAX_MOVES {
        for is_white in [true, false] {
            // Check if all games are done.
            if done.iter().all(|x| *x) {
                break 'outer;
            }

            // Perform search as the current agent.
            let (current, other) = if is_white {
                (&mut white_mcts, &mut black_mcts)
            } else {
                (&mut black_mcts, &mut white_mcts)
            };
            let top_actions: [_; BATCH_SIZE] = if is_white {
                current.gumbel_sequential_halving(
                    white,
                    &white_beta,
                    SAMPLED_ACTIONS,
                    SEARCH_BUDGET,
                    rng,
                )
            } else {
                current.gumbel_sequential_halving(
                    black,
                    &black_beta,
                    SAMPLED_ACTIONS,
                    SEARCH_BUDGET,
                    rng,
                )
            };

            // Pick the top actions and take a step.
            current.step(&top_actions);
            other.step(&top_actions);

            // Collect terminals and replays.
            let (terminals, replays): (Vec<_>, Vec<_>) = current
                .restart_terminal_envs(&mut rand::rng())
                .zip(&mut done)
                .filter_map(|(x, done)| if *done { None } else { Some((x?, done)) })
                .map(|(t, done)| {
                    *done = true;
                    t
                })
                .unzip();
            // Also reset other's nodes and envs.
            other
                .nodes_and_envs_mut()
                .zip(current.nodes_and_envs())
                .zip(&done)
                .filter(|(_, done)| **done)
                .for_each(|(((node, other_env), (_, current_env)), _)| {
                    *node = Node::default();
                    *other_env = current_env.clone();
                });

            for replay in replays {
                log::debug!("{}", replay.to_string().trim_end());
            }

            // Update evaluation results.
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

impl Evaluation {
    fn win_rate(&self) -> f64 {
        f64::from(self.wins) / f64::from(self.wins + self.losses + self.draws)
    }
}
