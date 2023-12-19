use std::{
    fmt::{self, Write},
    fs::OpenOptions,
    io::{BufRead, BufReader},
};

use fast_tak::{
    takparse::{Move, Tps},
    Game,
};
use ordered_float::NotNan;
use rand::prelude::*;
use rayon::prelude::*;
use takzero::{
    network::{
        net4::Net4 as Net,
        repr::{game_to_tensor, move_mask, output_size, policy_tensor},
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
#[rustfmt::skip] #[allow(dead_code)] const fn assert_env<E: Environment>() where Replay<E>: Augment + fmt::Display {}
const _: () = assert_env::<Env>();

// The network architecture.
#[rustfmt::skip] #[allow(dead_code)] const fn assert_net<NET: Network + Agent<Env>>() {}
const _: () = assert_net::<Net>();

const DEVICE: Device = Device::Cuda(0);
const LEARNING_RATE: f64 = 1e-4;
const BATCH_SIZE: usize = 512;
const STEPS: u32 = 200;

const GAME_COUNT: usize = 128;
const VISITS: usize = 200;
const BETA: [f32; GAME_COUNT] = [0.0; GAME_COUNT];
const MAX_PLIES: usize = 40;

fn main() {
    let seed: u64 = rand::thread_rng().gen();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(DEVICE, Some(rng.gen()));
    net.save("before.ot").unwrap();

    let mut opt = Adam::default().build(net.vs_mut(), LEARNING_RATE).unwrap();

    let file = OpenOptions::new().read(true).open("./targets.txt").unwrap();
    let targets = BufReader::new(file)
        .lines()
        .map(|line| line.unwrap().parse().unwrap())
        .collect::<Vec<Target<Env>>>();
    println!("loaded {} targets", targets.len());

    for step in 0..STEPS {
        let batch = targets.choose_multiple(&mut rng, BATCH_SIZE);

        // Create input tensors.
        let mut inputs = Vec::with_capacity(BATCH_SIZE);
        let mut value_targets = Vec::with_capacity(BATCH_SIZE);
        let mut policy_targets = Vec::with_capacity(BATCH_SIZE);
        let mut masks = Vec::with_capacity(BATCH_SIZE);
        for target in batch {
            let target = target.augment(&mut rng);
            inputs.push(game_to_tensor(&target.env, DEVICE));
            value_targets.push(target.value);
            policy_targets.push(policy_tensor::<N>(&target.policy, DEVICE));
            masks.push(move_mask::<N>(
                &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
                DEVICE,
            ));
        }

        // Get network output.
        let input = Tensor::cat(&inputs, 0);
        let mask = Tensor::cat(&masks, 0);
        let (policy, network_value) = net.forward_t(&input, true);
        let log_softmax_network_policy = policy
            .masked_fill(&mask, f64::from(f32::MIN))
            .view([-1, output_size::<N>() as i64])
            .log_softmax(1, Kind::Float);

        // Get the target.
        let target_policy =
            Tensor::stack(&policy_targets, 0).view(log_softmax_network_policy.size().as_slice());
        let target_value = Tensor::from_slice(&value_targets).unsqueeze(1).to(DEVICE);

        // Calculate loss.
        let loss_policy = -(log_softmax_network_policy * &target_policy).sum(Kind::Float)
            / i64::try_from(BATCH_SIZE).unwrap();
        let loss_value = (target_value - network_value).square().mean(Kind::Float);
        let loss = &loss_value + &loss_policy;

        // Take step.
        opt.backward_step(&loss);

        println!(
            "value={loss_value:?}, policy={loss_policy:?}, total={loss:?}, {} / {STEPS}",
            step + 1
        );
    }

    net.save("after.ot").unwrap();

    let before = Net::load("before.ot", DEVICE).unwrap();
    let after = net;

    let mut actions = Vec::new();
    let games: [_; GAME_COUNT] = std::array::from_fn(|_| Env::new_opening(&mut rng, &mut actions));
    println!("before vs after {:?}", compete(&before, &after, &games));
    println!("after vs before {:?}", compete(&after, &before, &games));
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

    let mut game_replays: Vec<(Tps, Vec<Move>)> = games
        .iter()
        .cloned()
        .map(|game| (game.into(), Vec::new()))
        .collect();

    let mut done = [false; BATCH_SIZE];

    'outer: for _ in 0..MAX_PLIES {
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
                )
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
                        replay.1.push(action);

                        if let Some(terminal) = game.terminal() {
                            *game = Env::default();
                            *white_node = Node::default();
                            *black_node = Node::default();
                            *done = true;
                            let (tps, moves) =
                                std::mem::replace(replay, (Tps::starting_position(N), Vec::new()));
                            println!(
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
