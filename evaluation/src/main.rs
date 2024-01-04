#![warn(clippy::pedantic, clippy::style)]

use std::{array, fs::read_dir, path::PathBuf};

use clap::Parser;
use rand::{prelude::*, rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use takzero::{
    network::{
        net5::{Env, Net},
        Network,
    },
    search::{
        agent::Agent,
        env::{Environment, Terminal},
        node::{gumbel::batched_simulate, Node},
    },
    target::Replay,
};
use tch::Device;

const DEVICE: Device = Device::Cuda(0);

const BATCH_SIZE: usize = 32;
const BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const VISITS: usize = 100;
const MAX_MOVES: usize = 60;

#[derive(Parser, Debug)]
struct Args {
    /// Path to models
    #[arg(long)]
    model_path: PathBuf,
}

fn main() {
    env_logger::init();
    log::info!("Begin.");
    tch::no_grad(real_main);
}

fn real_main() {
    let args = Args::parse();
    let seed: u64 = thread_rng().gen();
    log::info!("seed: {seed}");
    let mut rng = StdRng::seed_from_u64(seed);

    loop {
        let paths: Vec<_> = read_dir(&args.model_path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().map(|ext| ext == "ot").unwrap_or_default())
            .collect();
        if paths.len() < 2 {
            let time = std::time::Duration::from_secs(600);
            log::info!("Too few models. Sleeping for {time:?}.");
            std::thread::sleep(time);
            continue;
        }

        let mut match_up = paths.choose_multiple(&mut rng, 2);
        let path_a = match_up.next().unwrap();
        let path_b = match_up.next().unwrap();

        let Ok(a) = Net::load(path_a, DEVICE) else {
            log::warn!("Cannot load {}", path_a.display());
            continue;
        };
        let Ok(b) = Net::load(path_b, DEVICE) else {
            log::warn!("Cannot load {}", path_b.display());
            continue;
        };
        let name_a = path_a.file_name().unwrap().to_string_lossy().to_string();
        let name_b = path_b.file_name().unwrap().to_string_lossy().to_string();

        let mut actions = Vec::new();
        let games: [Env; BATCH_SIZE] = array::from_fn(|_| Env::new_opening(&mut rng, &mut actions));

        let a_as_white = compete(&a, &b, &games);
        log::info!(
            "{name_a} vs. {name_b}: {a_as_white:?} {:.1}%",
            a_as_white.win_rate() * 100.0
        );
        let b_as_white = compete(&b, &a, &games);
        log::info!(
            "{name_b} vs. {name_a}: {b_as_white:?} {:.1}%",
            b_as_white.win_rate() * 100.0
        );
    }
}
/// Pit two networks against each other in the given games. Evaluation is from
/// the perspective of white.
#[allow(dead_code)]
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
                    if node.evaluation.is_known() {
                        node.children
                            .iter()
                            .min_by_key(|(_, child)| child.evaluation)
                    } else {
                        node.children
                            .iter()
                            .max_by_key(|(_, child)| child.visit_count)
                    }
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
                            log::debug!("{}", replay.to_string().trim_end());
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

impl Evaluation {
    fn win_rate(&self) -> f64 {
        f64::from(self.wins) / f64::from(self.wins + self.losses + self.draws)
    }
}
