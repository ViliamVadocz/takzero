#![warn(clippy::pedantic, clippy::style)]

use std::{array, fs::read_dir, path::PathBuf};

use clap::Parser;
use evaluation::Evaluation;
use rand::rngs::ThreadRng;
use rayon::prelude::*;
use takzero::{
    fast_tak::Game,
    network::{net4::Net4, Network},
    search::{
        env::{Environment, Terminal},
        node::{gumbel::gumbel_sequential_halving, Node},
    },
};
use tch::Device;

mod evaluation;

const N: usize = 4;
const HALF_KOMI: i8 = 0;
type Env = Game<N, HALF_KOMI>;
type Net = Net4;

const DEVICE: Device = Device::Cuda(0);

const SAMPLED: usize = usize::MAX;
const SIMULATIONS: u32 = 1024;
const OPENINGS: usize = N * N * (N * N - 1);

#[derive(Parser, Debug)]
struct Args {
    /// Path to models
    #[arg(long)]
    model_path: PathBuf,
    /// Path to reference model
    #[arg(long)]
    reference_model: PathBuf,
}

fn main() {
    env_logger::init();
    log::info!("Begin.");

    let args = Args::parse();
    let reference = Net::load(&args.reference_model, DEVICE).unwrap();

    for entry in read_dir(args.model_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        assert!(path.is_file());
        if path == args.reference_model {
            log::info!("Skipping {path:?} because it is the same as the reference");
            continue;
        }

        log::info!("Competing against {path:?}");
        let subject = Net::load(path, DEVICE).unwrap();

        let result = compete(&reference, &subject);
        log::info!("{result:?} win rate: {:.2}", 100. * result.win_rate());
    }
}

fn compete(reference: &Net, subject: &Net) -> Evaluation {
    let mut evaluation = Evaluation::default();

    let mut actions: [_; OPENINGS] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; OPENINGS] = array::from_fn(|_| Vec::new());

    for order in [[subject, reference], [reference, subject]] {
        let mut games: [_; OPENINGS] = array::from_fn(get_opening);
        let mut subject_nodes: [_; OPENINGS] = array::from_fn(|_| Node::default());
        let mut reference_nodes: [_; OPENINGS] = array::from_fn(|_| Node::default());

        // Play until all games are finished.
        let mut done = [false; OPENINGS];
        while !done.iter().all(|x| *x) {
            for agent in order {
                let subject_playing = std::ptr::eq(agent, subject);

                let top_actions = gumbel_sequential_halving(
                    if subject_playing {
                        &mut subject_nodes
                    } else {
                        &mut reference_nodes
                    },
                    &games,
                    agent,
                    SAMPLED,
                    SIMULATIONS,
                    &mut actions,
                    &mut trajectories,
                    None::<&mut ThreadRng>,
                );

                evaluation += top_actions
                    .into_par_iter()
                    .zip(games.par_iter_mut())
                    .zip(subject_nodes.par_iter_mut())
                    .zip(reference_nodes.par_iter_mut())
                    .zip(done.par_iter_mut())
                    .filter(|(_, done)| !**done)
                    .filter_map(|((((action, game), subject_node), reference_node), done)| {
                        game.step(action);

                        if let Some(terminal) = game.terminal() {
                            *done = true;
                            *game = Env::default();
                            *subject_node = Node::default();
                            *reference_node = Node::default();
                            Some(terminal)
                        } else {
                            subject_node.descend(&action);
                            reference_node.descend(&action);
                            None
                        }
                    })
                    // Mapping is flipped because we look at the terminal AFTER a move was made.
                    .map(|terminal| match (terminal, subject_playing) {
                        // If the position is a loss for the current player and beta just made a
                        // move, it's win.
                        (Terminal::Loss, true) | (Terminal::Win, false) => Evaluation::win(),
                        // If the position is a win for the current player and beta just made a move
                        // it's a loss.
                        (Terminal::Win, true) | (Terminal::Loss, false) => Evaluation::loss(),
                        (Terminal::Draw, _) => Evaluation::draw(),
                    })
                    .sum::<Evaluation>();
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
