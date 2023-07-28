use fast_tak::{takparse::Move, Game, GameResult};
use mimalloc::MiMalloc;
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use takzero::{
    network::{net3::Net3, Network},
    search::{env::Environment, mcts::Node},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const DISCOUNT: f32 = 0.99;

fn main() {
    let net = Net3::load("test_network.ot").unwrap_or_default();
    // net.save("test_network.ot").unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let targets = self_play(net, &mut rng);
    for target in targets {
        println!(
            "ply: {}, action: {}, value: {}",
            target.state.ply, target.action, target.value
        );
    }
}

#[derive(Debug)]
struct Target {
    state: Game<3, 0>,          // s_t
    action: Move,               // a_t
    value: f32,                 // V(s_{t+1})
    policy: Box<[(Move, f32)]>, // \pi'(s_t)
}

fn self_play(net: Net3, rng: &mut impl Rng) -> Vec<Target> {
    let mut game: Game<3, 0> = Game::default();

    let mut moves = Vec::new();
    // Two random moves because opening is weird.
    for _ in 0..2 {
        game.possible_moves(&mut moves);
        let mov = moves.drain(..).choose(rng).unwrap();
        game.play(mov).unwrap();
        println!("{mov}");
    }

    let mut root = Node::default();
    let mut targets = Vec::new();
    while game.result() == GameResult::Ongoing {
        let action = if root.evaluation.is_known() {
            *root
                .children
                .iter()
                .min_by_key(|(_, node)| node.evaluation)
                .map(|(a, _)| a)
                .unwrap()
        } else {
            root.sequential_halving_with_gumbel(&game, &mut moves, rng, &net, 1024)
        };
        println!("{action}");
        println!("{root}");

        let state = game.clone();
        game.step(action);
        let improved_policy: Vec<_> = root
            .children
            .iter()
            .zip(root.improved_policy())
            .map(|((a, _), p)| (*a, p))
            .collect();
        root = root.play(&action);
        targets.push(Target {
            state,
            action,
            value: root.evaluation.negate().into(),
            policy: improved_policy.into_boxed_slice(),
        });
    }
    targets
}
