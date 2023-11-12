#![warn(clippy::pedantic, clippy::style)]

use std::{
    array,
    collections::HashMap,
    fmt::Write,
    fs::{read_dir, read_to_string},
    path::PathBuf,
};

use clap::Parser;
use evaluation::Evaluation;
use fast_tak::{
    takparse::{Move, Tps},
    Game,
};
use rand::{prelude::*, rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use takzero::{
    network::{net4::Net4, Network},
    search::{
        agent::Agent,
        env::{Environment, Terminal},
        node::{gumbel::gumbel_sequential_halving, Node},
    },
};
use tch::Device;

mod evaluation;

const N: usize = 4;
const HALF_KOMI: i8 = 4;
type Env = Game<N, HALF_KOMI>;
type Net = Net4;

const DEVICE: Device = Device::Cuda(0);

const BATCH_SIZE: usize = 32;
const SAMPLED: usize = 8;
const SIMULATIONS: u32 = 256;
const BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];

const OPENINGS: usize = N * N * (N * N - 1);

#[derive(Parser, Debug)]
struct Args {
    /// Path to models
    #[arg(long)]
    model_path: PathBuf,
    /// Path to RND
    #[cfg(not(feature = "baseline"))]
    #[arg(long)]
    rnd_path: PathBuf,
    /// Number of games to play
    #[arg(long)]
    games: u32,
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

    #[cfg(not(feature = "baseline"))]
    let rnd_losses: Vec<f64> = read_to_string(args.rnd_path)
        .unwrap()
        .lines()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .unwrap();
    let mut paths: Vec<_> = read_dir(args.model_path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().map(|ext| ext == "ot").unwrap_or_default())
        .collect();
    paths.sort();

    let match_ups: Vec<_> = (0..args.games)
        .map(|_| {
            let mut iter = paths.choose_multiple(&mut rng, 2);
            (iter.next().unwrap(), iter.next().unwrap())
        })
        .collect();

    let mut results: HashMap<(String, String), Evaluation> = HashMap::new();
    for (path_a, path_b) in match_ups {
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
        #[cfg(not(feature = "baseline"))]
        let rnd_loss_a = extract_rnd(&rnd_losses, &name_a);
        #[cfg(not(feature = "baseline"))]
        let rnd_loss_b = extract_rnd(&rnd_losses, &name_b);

        let games: [Env; BATCH_SIZE] =
            array::from_fn(|_| get_opening(rng.gen::<usize>() % OPENINGS));

        log::info!("{name_a} vs. {name_b}");
        let a_as_white = compete(
            &a,
            #[cfg(not(feature = "baseline"))]
            rnd_loss_a,
            &b,
            #[cfg(not(feature = "baseline"))]
            rnd_loss_b,
            &games,
        );
        log::info!("{a_as_white:?}");
        *results.entry((name_a.clone(), name_b.clone())).or_default() += a_as_white;

        log::info!("{name_b} vs. {name_a}");
        let b_as_white = compete(
            &b,
            #[cfg(not(feature = "baseline"))]
            rnd_loss_b,
            &a,
            #[cfg(not(feature = "baseline"))]
            rnd_loss_a,
            &games,
        );
        log::info!("{b_as_white:?}");
        *results.entry((name_b, name_a)).or_default() += b_as_white;
    }

    log::info!("Done!");
    for ((a, b), result) in results {
        log::info!("{a} {b} {result:?} {}", result.win_rate());
    }
}

/// Pit two networks against each other in the given games. Evaluation is from
/// the perspective of white.
fn compete(
    white: &Net,
    #[cfg(not(feature = "baseline"))] white_rnd_loss: f64,
    black: &Net,
    #[cfg(not(feature = "baseline"))] black_rnd_loss: f64,
    games: &[Env],
) -> Evaluation {
    let mut evaluation = Evaluation::default();

    let mut games = games.to_owned();
    let mut white_nodes: Vec<_> = (0..BATCH_SIZE).map(|_| Node::default()).collect();
    let mut black_nodes: Vec<_> = (0..BATCH_SIZE).map(|_| Node::default()).collect();
    #[cfg(not(feature = "baseline"))]
    let mut white_context = <Net as Agent<Env>>::Context::new(white_rnd_loss);
    #[cfg(not(feature = "baseline"))]
    let mut black_context = <Net as Agent<Env>>::Context::new(black_rnd_loss);
    #[cfg(feature = "baseline")]
    let mut context = <Net as Agent<Env>>::Context::new(0.0);

    let mut actions: Vec<_> = (0..BATCH_SIZE).map(|_| Vec::new()).collect();
    let mut trajectories: Vec<_> = (0..BATCH_SIZE).map(|_| Vec::new()).collect();

    let mut game_replays: Vec<(Tps, Vec<Move>)> = games
        .iter()
        .cloned()
        .map(|game| (game.into(), Vec::new()))
        .collect();

    let mut done = [false; BATCH_SIZE];

    'outer: loop {
        for (agent, is_white) in [(white, true), (black, false)] {
            if done.iter().all(|x| *x) {
                break 'outer;
            }

            let top_actions = gumbel_sequential_halving(
                if is_white {
                    &mut white_nodes
                } else {
                    &mut black_nodes
                },
                &games,
                agent,
                SAMPLED,
                SIMULATIONS,
                &BETA,
                #[cfg(not(feature = "baseline"))]
                if is_white {
                    &mut white_context
                } else {
                    &mut black_context
                },
                #[cfg(feature = "baseline")]
                &mut context,
                &mut actions,
                &mut trajectories,
                None::<&mut ThreadRng>,
            );

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
                        game.step(action);
                        replay.1.push(action);

                        if let Some(terminal) = game.terminal() {
                            *game = Env::default();
                            *white_node = Node::default();
                            *black_node = Node::default();
                            *done = true;
                            let (tps, moves) =
                                std::mem::replace(replay, (Tps::starting_position(N), Vec::new()));
                            log::debug!(
                                "{tps} {}",
                                moves.into_iter().fold(String::new(), |mut s, m| {
                                    let _ = write!(s, "{m} ");
                                    s
                                })
                            );
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

fn get_opening(i: usize) -> Env {
    assert!(i < OPENINGS);
    let mut actions = Vec::new();

    let mut env = Env::default();
    env.populate_actions(&mut actions);
    let len = actions.len();
    let action = actions.drain(..).nth(i % len).unwrap();

    env.step(action);
    env.populate_actions(&mut actions);
    let action = actions.drain(..).nth(i / len).unwrap();

    env.step(action);
    env
}

const RND_SMOOTHING: usize = 100;

#[allow(clippy::cast_precision_loss)]
fn extract_rnd(rnd_losses: &[f64], name: &str) -> f64 {
    let (steps, _) = name.split_once('_').expect("unexpected file name format");
    let steps: usize = steps.parse().unwrap();
    rnd_losses[steps.saturating_sub(RND_SMOOTHING)..steps]
        .iter()
        .sum::<f64>()
        / RND_SMOOTHING as f64
}
