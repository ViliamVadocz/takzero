use std::{
    array,
    fmt,
    fs::OpenOptions,
    io::Write,
    iter::{once, Sum},
    ops::AddAssign,
    path::PathBuf,
    sync::atomic::Ordering,
};

use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use takzero::{
    network::Network,
    search::{
        agent::Agent,
        env::{Environment, Terminal},
        node::{gumbel::gumbel_sequential_halving, Node},
    },
};
use tch::Device;

use crate::BetaNet;

const BATCH_SIZE: usize = 128;
const SAMPLED: usize = 32;
const SIMULATIONS: u32 = 1024;

const MIN_WIN_RATE: f32 = 0.60;

// TODO: Clean up

/// Evaluate checkpoints of the network to make sure that it is improving.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    statistics_path: PathBuf,
) where
    E::Action: fmt::Display,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Omega stores the currently best performing network.
    let mut omega_net = NET::new(device, None);
    let mut omega_net_index = beta_net.0.load(Ordering::Relaxed);
    omega_net
        .vs_mut()
        .copy(&beta_net.1.read().unwrap())
        .unwrap();

    let mut net = NET::new(device, None);
    let mut net_index = omega_net_index;
    net.vs_mut().copy(omega_net.vs()).unwrap();

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| E::default());
    let mut omega_nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut beta_nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut action_record: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut gumbel_noise: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut omega_full_games: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut beta_full_games: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());

    let mut results = Evaluation::default();
    loop {
        // Update the beta network.
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index >= net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
            // Reset statistics.
            results = Evaluation::default();
        }
        // If the network are the same there is no need to run evaluation.
        if omega_net_index == net_index {
            std::thread::yield_now();
            continue;
        }

        let pit_results = pit(
            &omega_net,
            &net,
            &mut rng,
            &mut omega_nodes,
            &mut beta_nodes,
            &mut envs,
            &mut action_record,
            &mut actions,
            &mut trajectories,
            &mut gumbel_noise,
            &mut omega_full_games,
            &mut beta_full_games,
        );
        results += pit_results;

        // Save full games to file.
        let path_1 = statistics_path.join(format!(
            "games_omega{omega_net_index}_vs_beta{net_index}.txt"
        ));
        let path_2 = statistics_path.join(format!(
            "games_beta{net_index}_vs_omega{omega_net_index}.txt"
        ));
        let content_1 = full_games_to_string::<E>(&mut omega_full_games);
        let content_2 = full_games_to_string::<E>(&mut beta_full_games);
        rayon::spawn(move || {
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(path_1)
                .expect("statistics file path should be valid and writable");
            file.write_all(content_1.as_bytes()).unwrap();

            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(path_2)
                .expect("statistics file path should be valid and writable");
            file.write_all(content_2.as_bytes()).unwrap();
        });

        // Update omega if beta is significantly better.
        if results.win_rate() >= MIN_WIN_RATE {
            // Save statistics.
            let path = statistics_path.join(format!(
                "evaluation_omega{omega_net_index}_vs_beta{net_index}.txt"
            ));
            let content = format!("{results:?}");
            rayon::spawn(move || {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)
                    .expect("statistics file path should be valid and writable");
                file.write_all(content.as_bytes()).unwrap();
            });
            // Update network parameters.
            omega_net.vs_mut().copy(net.vs()).unwrap();
            omega_net_index = net_index;
        }

        if cfg!(test) {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pit<E: Environment, A: Agent<E>, R: Rng>(
    omega: &A,
    beta: &A,
    rng: &mut R,

    omega_nodes: &mut [Node<E>], // not specific to omega
    beta_nodes: &mut [Node<E>],  // not specific to beta

    envs: &mut [E],
    action_record: &mut [Vec<E::Action>],
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],
    gumbel_noise: &mut [Vec<f32>],

    omega_full_games: &mut [Vec<E::Action>],
    beta_full_games: &mut [Vec<E::Action>],
) -> Evaluation {
    // Evaluation is from the perspective of the omega.
    let mut evaluation = Evaluation::default();

    for (full_games, order) in [
        (omega_full_games, [omega, beta]),
        (beta_full_games, [beta, omega]),
    ] {
        // Reset.
        // TODO: Think about openings???
        envs.par_iter_mut().for_each(|env| *env = E::default());
        omega_nodes
            .par_iter_mut()
            .for_each(|node| *node = Node::default());
        beta_nodes
            .par_iter_mut()
            .for_each(|node| *node = Node::default());

        // Play until all games are finished.
        let mut done = [false; BATCH_SIZE];
        debug_assert!(action_record.iter().all(Vec::is_empty));
        while !done.iter().all(|x| *x) {
            for agent in order {
                let top_actions = gumbel_sequential_halving::<_, _, R>(
                    omega_nodes,
                    envs,
                    agent,
                    SAMPLED,
                    SIMULATIONS,
                    actions,
                    trajectories,
                    Some((rng, gumbel_noise)),
                );

                let is_omega = std::ptr::eq(agent, omega);
                evaluation += top_actions
                    .into_par_iter()
                    .zip(envs.par_iter_mut())
                    .zip(action_record.par_iter_mut())
                    .zip(omega_nodes.par_iter_mut())
                    .zip(beta_nodes.par_iter_mut())
                    .zip(full_games.par_iter_mut())
                    .zip(done.par_iter_mut())
                    .filter(|(_, done)| !**done)
                    .filter_map(
                        |((((((action, env), record), node_1), node_2), full_game), done)| {
                            env.step(action.clone());
                            record.push(action.clone());

                            if let Some(terminal) = env.terminal() {
                                full_game.append(record);
                                *done = true;
                                *env = E::default();
                                *node_1 = Node::default();
                                *node_2 = Node::default();
                                Some(terminal)
                            } else {
                                node_1.descend(&action);
                                node_2.descend(&action);
                                None
                            }
                        },
                    )
                    .map(|terminal| match (terminal, is_omega) {
                        (Terminal::Win, true) | (Terminal::Loss, false) => Evaluation::win(),
                        (Terminal::Loss, true) | (Terminal::Win, false) => Evaluation::loss(),
                        (Terminal::Draw, _) => Evaluation::draw(),
                    })
                    .sum::<Evaluation>();
            }
        }
    }

    evaluation
}

#[derive(Debug, Default)]
struct Evaluation {
    wins: u32,
    losses: u32,
    draws: u32,
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
        iter.fold(Evaluation::default(), |mut a, b| {
            a += b;
            a
        })
    }
}

impl Evaluation {
    fn win_rate(&self) -> f32 {
        self.wins as f32 / (self.wins + self.losses + self.draws) as f32
    }

    fn win() -> Self {
        Self {
            wins: 1,
            ..Default::default()
        }
    }

    fn loss() -> Self {
        Self {
            losses: 1,
            ..Default::default()
        }
    }

    fn draw() -> Self {
        Self {
            draws: 1,
            ..Default::default()
        }
    }
}

fn full_games_to_string<E: Environment>(full_games: &mut [Vec<E::Action>]) -> String
where
    E::Action: fmt::Display,
{
    full_games
        .iter_mut()
        .flat_map(|full_game| {
            full_game
                .drain(..)
                .map(|a| format!("{a} "))
                .chain(once("\n".to_string()))
        })
        .collect()
}
