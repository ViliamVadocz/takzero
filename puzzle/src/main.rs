use std::path::PathBuf;

use clap::Parser;
use fast_tak::takparse::{Move, Tps};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sqlite::{Connection, Statement, Value};
use takzero::{
    network::{
        net6_simhash::{Env, Net},
        Network,
    },
    search::node::{batched::BatchedMCTS, Node},
};
use tch::Device;

#[derive(Parser, Debug)]
struct Args {
    /// Path where the model is stored
    #[arg(long)]
    model: PathBuf,
    /// Path to puzzle database
    #[arg(long)]
    puzzle_db: PathBuf,
    /// Path to save graph
    #[arg(long)]
    graph: PathBuf,

    /// Sampled actions
    #[arg(long, default_value_t = 64)]
    sampled_actions: usize,
    /// Search budget
    #[arg(long, default_value_t = 768)]
    search_budget: u32,
}

const BATCH_SIZE: usize = 64;
const SEED: u64 = 12345;
const ZERO_BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const DEVICE: Device = Device::Cuda(0);

fn main() {
    env_logger::init();
    log::info!("Begin.");
    tch::no_grad(real_main);
}

fn real_main() {
    let args = Args::parse();
    let connection = sqlite::open(&args.puzzle_db)
        .expect("Puzzle database should be located at path specified by `puzzle-db`.");

    log::info!("Benchmarking {}", args.model.display());
    let net = Net::load_partial(&args.model, DEVICE)
        .expect("Network should be located at path specified by `model`.");

    let mut rng = StdRng::seed_from_u64(SEED);

    let depth_3 = benchmark(
        &net,
        tinue(&connection, 3),
        true,
        args.sampled_actions,
        args.search_budget,
        &mut rng,
    );
    let depth_5 = benchmark(
        &net,
        tinue(&connection, 5),
        true,
        args.sampled_actions,
        args.search_budget,
        &mut rng,
    );
    let depth_7 = benchmark(
        &net,
        tinue(&connection, 7),
        true,
        args.sampled_actions,
        args.search_budget,
        &mut rng,
    );
    let depth_9 = benchmark(
        &net,
        tinue(&connection, 9),
        true,
        args.sampled_actions,
        args.search_budget,
        &mut rng,
    );

    let depth_2 = benchmark(
        &net,
        avoidance(&connection, 2),
        false,
        args.sampled_actions,
        args.search_budget,
        &mut rng,
    );
    let depth_4 = benchmark(
        &net,
        avoidance(&connection, 4),
        false,
        args.sampled_actions,
        args.search_budget,
        &mut rng,
    );
    let depth_6 = benchmark(
        &net,
        avoidance(&connection, 6),
        false,
        args.sampled_actions,
        args.search_budget,
        &mut rng,
    );

    log::info!("depth 3: {}", depth_3.solve_rate());
    log::info!("depth 5: {}", depth_5.solve_rate());
    log::info!("depth 7: {}", depth_7.solve_rate());
    log::info!("depth 9: {}", depth_9.solve_rate());
    log::info!("depth 2: {}", depth_2.solve_rate());
    log::info!("depth 4: {}", depth_4.solve_rate());
    log::info!("depth 6: {}", depth_6.solve_rate());
}

#[derive(Debug)]
struct PuzzleResult {
    attempted: usize,
    solved: usize,
    proven: usize,
}

impl PuzzleResult {
    fn solve_rate(&self) -> f64 {
        self.solved as f64 / self.attempted as f64
    }

    #[allow(unused)]
    fn prove_rate(&self) -> f64 {
        self.proven as f64 / self.attempted as f64
    }
}

fn tinue(connection: &Connection, depth: i64) -> Statement {
    let query = r#"SELECT * FROM puzzles
    JOIN games ON puzzles.game_id = games.id
    WHERE games.size = 6
        -- AND instr(tps, "1C") > 0
        -- AND instr(tps, "2C") > 0
        AND puzzles.tinue_length = :depth
        AND puzzles.tinue_avoidance_length IS NULL
        -- AND puzzles.tiltak_2komi_second_move_eval < 0.7
    ORDER BY puzzles.game_id ASC"#;
    let mut statement = connection.prepare(query).unwrap();
    statement
        .bind::<&[(_, Value)]>(&[(":depth", depth.into())])
        .unwrap();
    statement
}

fn avoidance(connection: &Connection, depth: i64) -> Statement {
    let query = r#"SELECT * FROM puzzles
    JOIN games ON puzzles.game_id = games.id
    WHERE games.size = 6
        -- AND instr(tps, "1C") > 0
        -- AND instr(tps, "2C") > 0
        AND puzzles.tinue_avoidance_length = :depth
        AND puzzles.tinue_length IS NULL
        -- AND puzzles.tiltak_2komi_eval > 0.7
    ORDER BY game_id ASC"#;
    let mut statement = connection.prepare(query).unwrap();
    statement
        .bind::<&[(_, Value)]>(&[(":depth", depth.into())])
        .unwrap();
    statement
}

fn benchmark(
    agent: &Net,
    statement: Statement,
    win: bool,
    sampled_actions: usize,
    search_budget: u32,
    rng: &mut impl Rng,
) -> PuzzleResult {
    let (puzzles, solutions): (Vec<_>, Vec<_>) = statement
        .into_iter()
        .map(|row| {
            let row = row.unwrap();
            let tps: Tps = row.read::<&str, _>("tps").parse().unwrap();
            let solution: Move = row.read::<&str, _>("solution").parse().unwrap();
            let game: Env = tps.into();
            (game, solution)
        })
        .unzip();

    let mut batched_mcts = BatchedMCTS::from_envs(std::array::from_fn(|_| Env::default()));

    // Attempt to solve puzzles.
    let mut attempted = 0;
    let mut solved = 0;
    let mut proven = 0;
    for (puzzle_batch, solution_batch) in puzzles
        .chunks_exact(BATCH_SIZE)
        .zip(solutions.chunks_exact(BATCH_SIZE))
    {
        batched_mcts
            .nodes_and_envs_mut()
            .zip(puzzle_batch)
            .for_each(|((node, env), puzzle)| {
                *node = Node::default();
                *env = puzzle.clone();
            });

        batched_mcts.gumbel_sequential_halving(
            agent,
            &ZERO_BETA,
            sampled_actions,
            search_budget,
            rng,
        );

        attempted += puzzle_batch.len();
        let selected_actions: [_; BATCH_SIZE] = batched_mcts.select_best_actions();
        solved += selected_actions
            .iter()
            .zip(solution_batch)
            .filter(|(a, b)| a == b)
            .count();

        // Print debug information.
        batched_mcts
            .nodes_and_envs()
            .zip(solution_batch)
            .zip(selected_actions)
            .enumerate()
            .for_each(|(index, (((node, env), solution), selected))| {
                log::debug!(
                    "[{index}] tps: {}, selected: {selected}, solution: {solution}, solved: {}",
                    Tps::from(env.clone()),
                    &selected == solution
                );
                log::debug!("{node}");
            });

        // Count how many solutions were proven by the terminal solver.
        proven += if win {
            // Count how many nodes have been solved to a win.
            batched_mcts
                .nodes_and_envs()
                .filter(|(node, _)| node.evaluation.is_win())
                .count()
        } else {
            // Count how many nodes have had all but one child solved as a win
            // (i.e. found the tinue for all other moves).
            batched_mcts
                .nodes_and_envs()
                .filter(|(node, _)| {
                    node.children
                        .iter()
                        .filter(|(_, child)| child.evaluation.is_win())
                        .count()
                        == node.children.len() - 1
                })
                .count()
        };
    }

    let result = PuzzleResult {
        attempted,
        solved,
        proven,
    };
    log::info!("{result:?}");
    result
}
