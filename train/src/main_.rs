use fast_tak::{
    takparse::{Color, Move},
    Game,
    GameResult,
    Symmetry,
};
use mimalloc::MiMalloc;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
    SeedableRng,
};
use rayon::prelude::*;
use takzero::{
    network::{
        net3::Net3,
        repr::{game_to_tensor, move_mask, policy_tensor},
        Network,
    },
    search::{env::Environment, node::Node},
};
use tch::{
    nn::{Adam, OptimizerConfig},
    Device,
    Kind,
    Reduction,
    Tensor,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const DEVICE: Device = Device::Cuda(0);
const DISCOUNT: f32 = 0.99;
const NETWORK_PATH: &str = "test_network.ot";
const SEED: u64 = 785;
const SELF_PLAY_GAMES: usize = 64;
const EVALUATION_GAMES: usize = 256;
const BATCH_SIZE: usize = 32;
const SIMULATIONS: u32 = 32;

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let net = Net3::new(DEVICE, Some(rng.gen()));
    net.save(NETWORK_PATH).unwrap();

    let mut seeds = [0; SELF_PLAY_GAMES];
    rng.fill(&mut seeds);

    let targets: Vec<_> = seeds
        .into_par_iter()
        .flat_map_iter(|seed| generate_targets(seed).into_iter())
        .collect();

    // TODO: Put into function
    // Optimize
    let mut opt = Adam {
        wd: 1e-4,
        ..Default::default()
    }
    .build(net.vs(), 1e-3)
    .unwrap();

    let mut refs: Vec<_> = targets.iter().collect();
    refs.shuffle(&mut rng);

    for chunk in refs.chunks_exact(BATCH_SIZE) {
        let symmetries = chunk
            .iter()
            .flat_map(|target| target.symmetries().into_iter());

        let mut inputs = Vec::new();
        let mut policies = Vec::new();
        let mut values = Vec::new();
        let mut masks = Vec::new();
        for target in symmetries {
            inputs.push(game_to_tensor(&target.state, DEVICE));
            policies.push(policy_tensor::<3>(&target.policy));
            values.push(target.value);
            masks.push(move_mask::<3>(
                &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
            ))
        }

        // Get network output.
        let input = Tensor::cat(&inputs, 0).to(DEVICE);
        let mask = Tensor::cat(&masks, 0).to(DEVICE);
        let (policy, eval) = net.forward_t(&input, true);
        let policy = policy.masked_fill(&mask, 0.0).log_softmax(1, Kind::Float);

        // Get the target.
        let p = Tensor::stack(&policies, 0)
            .to(DEVICE)
            .view(policy.size().as_slice());
        let z = Tensor::from_slice(&values).unsqueeze(1).to(DEVICE);

        // println!("policy: {policy}");
        // println!("p: {p}");

        // println!("eval: {eval}");
        // println!("z: {z}");

        // Calculate loss.
        // let loss_p = -(p * policy).mean(Kind::Float);
        let loss_p = policy.kl_div(&p, Reduction::Sum, true) / BATCH_SIZE as i64;
        let loss_z = (z - eval).square().sum(Kind::Float) / BATCH_SIZE as i64;
        println!("p={loss_p:?}\t z={loss_z:?}");
        let total_loss = loss_z + loss_p;

        opt.backward_step(&total_loss);
    }

    // TODO: Put into function
    // Evaluate

    let new = net;
    let old = Net3::load(NETWORK_PATH, DEVICE).unwrap();

    tch::no_grad(|| evaluate(&new, &old, &mut rng));

    todo!("compare the two networks")
}

fn generate_targets(seed: u64) -> Vec<Target> {
    let net = Net3::load(NETWORK_PATH, DEVICE).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    tch::no_grad(|| self_play(net, &mut rng))
}

#[derive(Debug)]
struct Target {
    state: Game<3, 0>,          // s_t
    action: Move,               // a_t
    value: f32,                 // V(s_{t+1})
    policy: Box<[(Move, f32)]>, // \pi'(s_t)
}

impl Symmetry<3> for Target {
    fn symmetries(&self) -> [Self; 8] {
        let states = self.state.symmetries();
        let actions = Symmetry::<3>::symmetries(&self.action);

        let mut policies = vec![Vec::with_capacity(self.policy.len()); 8];
        self.policy.iter().for_each(|(mov, p)| {
            Symmetry::<3>::symmetries(mov)
                .into_iter()
                .zip(policies.iter_mut())
                .for_each(|(sym, policy)| policy.push((sym, *p)))
        });

        let mut iter =
            states
                .into_iter()
                .zip(actions)
                .zip(policies)
                .map(|((state, action), policy)| Target {
                    state,
                    action,
                    value: self.value,
                    policy: policy.into_boxed_slice(),
                });
        array::from_fn(|_| iter.next().unwrap())
    }
}

fn self_play(net: Net3, rng: &mut impl Rng) -> Vec<Target> {
    let mut game: Game<3, 0> = Game::default();

    let mut moves = Vec::new();
    // Two random moves because opening is weird.
    for _ in 0..2 {
        game.possible_moves(&mut moves);
        let mov = moves.drain(..).choose(rng).unwrap();
        game.play(mov).unwrap();
    }

    let mut root = Node::default();
    let mut targets: Vec<Target> = Vec::new();
    while game.result() == GameResult::Ongoing {
        let action = if root.evaluation.is_known() {
            *root
                .children
                .iter()
                .min_by_key(|(_, node)| node.evaluation)
                .map(|(a, _)| a)
                .unwrap()
        } else {
            root.sequential_halving_with_gumbel(&game, &mut moves, rng, &net, SIMULATIONS)
        };
        // Update value from tree re-use
        if let Some(target) = targets.last_mut() {
            target.value = DISCOUNT * Into::<f32>::into(root.evaluation.negate());
        }

        let state = game.clone();
        targets.push(Target {
            state,
            action,
            value: DISCOUNT
                // this will be updated on next iteration
                * Into::<f32>::into(
                    root.children
                        .iter()
                        .find(|(a, _)| a == &action)
                        .unwrap()
                        .1
                        .evaluation
                        .negate(),
                ),
            policy: root
                .children
                .iter()
                .zip(root.improved_policy())
                .map(|((a, _), p)| (*a, p))
                .collect(),
        });

        game.step(action);
        root = root.play(&action);
    }

    println!("generated {} targets", targets.len());
    targets
}

pub fn evaluate(new: &Net3, old: &Net3, rng: &mut impl Rng) -> PitResult {
    const N: usize = 3;
    const HALF_KOMI: i8 = 0;

    let mut result = PitResult::default();
    let mut moves = Vec::new();

    for i in 0..EVALUATION_GAMES {
        if result.wins > (EVALUATION_GAMES + EVALUATION_GAMES / 10) as u32
            || result.losses > (EVALUATION_GAMES - EVALUATION_GAMES / 10) as u32
        {
            println!("breaking early because result is already known");
            break;
        }

        println!("evaluation game {i}/{EVALUATION_GAMES}");
        let opening = {
            let mut game: Game<N, HALF_KOMI> = Game::default();
            game.possible_moves(&mut moves);
            let first = moves.drain(..).choose(rng).unwrap();
            game.play(first).unwrap();
            game.possible_moves(&mut moves);
            let second = moves.drain(..).choose(rng).unwrap();
            [first, second]
        };

        for color in [Color::White, Color::Black] {
            let mut game: Game<N, HALF_KOMI> = Game::from_moves(&opening).unwrap();

            let mut new_root = Node::default();
            let mut old_root = Node::default();

            while game.result() == GameResult::Ongoing {
                let (agent, root) = if game.to_move == color {
                    (new, &mut new_root)
                } else {
                    (old, &mut old_root)
                };
                let mov =
                    root.sequential_halving_with_gumbel(&game, &mut moves, rng, agent, SIMULATIONS);

                new_root = new_root.play(&mov);
                old_root = old_root.play(&mov);
                game.play(mov).unwrap();
            }
            println!("{:?} in {} plies as {color}", game.result(), game.ply);

            result.update(game.result(), color);
        }
    }

    result
}

#[derive(Debug, Default)]
pub struct PitResult {
    wins: u32,
    losses: u32,
    draws: u32,
}

impl PitResult {
    pub fn win_rate(&self) -> f64 {
        // another option:
        // (self.wins as f64 + self.draws as f64 / 2.) /
        // (self.wins + self.draws + self.losses) as f64
        self.wins as f64 / (self.wins + self.losses) as f64
    }

    fn update(&mut self, result: GameResult, color: Color) {
        match result {
            GameResult::Winner { color: winner, .. } => {
                if winner == color {
                    self.wins += 1
                } else {
                    self.losses += 1
                }
            }
            GameResult::Draw { .. } => self.draws += 1,
            GameResult::Ongoing => {}
        }
    }
}
