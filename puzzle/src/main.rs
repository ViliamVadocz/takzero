use std::{
    fs::read_dir,
    path::{Path, PathBuf},
};

use charming::{
    component::{Axis, Legend, Title},
    series::Line,
    theme::Theme,
    Chart,
    HtmlRenderer,
};
use clap::Parser;
use fast_tak::takparse::{Move, Tps};
use sqlite::{Connection, Statement, Value};
use takzero::{
    network::{
        net5::{Env, Net, RndNormalizationContext, N},
        Network,
    },
    search::node::{batched::BatchedMCTS, Node},
};
use tch::Device;

#[derive(Parser, Debug)]
struct Args {
    /// Path where models are stored
    #[arg(long)]
    model: PathBuf,
    /// Path to puzzle database
    #[arg(long)]
    puzzle_db: PathBuf,
    /// Path to save graph
    #[arg(long)]
    graph: PathBuf,
}

#[allow(clippy::assertions_on_constants)]
const _: () = assert!(N == 5, "Tilpaz is only supported for 5x5");

const BATCH_SIZE: usize = 128;
const VISITS: u32 = 1024;
const BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const DEVICE: Device = Device::Cuda(0);

const STEP: usize = 10;

fn main() {
    env_logger::init();
    log::info!("Begin.");
    tch::no_grad(real_main);
}

fn real_main() {
    let args = Args::parse();
    let connection = sqlite::open(&args.puzzle_db_path).unwrap();
    let mut paths: Vec<_> = read_dir(args.model_path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().map(|ext| ext == "ot").unwrap_or_default())
        .step_by(STEP)
        .collect();
    paths.sort();

    let mut points = Vec::new();
    for path in paths {
        let Ok(net) = Net::load(&path, DEVICE) else {
            log::warn!("Cannot load {}", path.display());
            continue;
        };
        let Some(model_steps) = path
            .file_stem()
            .and_then(|p| p.to_str()?.split_once('_')?.1.parse().ok())
        else {
            log::warn!("Cannot parse model steps in {}", path.display());
            continue;
        };
        log::info!("Benchmarking {}", path.display());

        let depth_3 = benchmark(&net, tinue(&connection, 3, 512), true);
        let depth_5 = benchmark(&net, tinue(&connection, 5, 384), true);
        let depth_7 = benchmark(&net, tinue(&connection, 7, 128), true);
        let depth_9 = benchmark(&net, tinue(&connection, 9, 128), true);

        let depth_2 = benchmark(&net, avoidance(&connection, 2, 256), false);
        let depth_4 = benchmark(&net, avoidance(&connection, 4, 128), false);
        let depth_6 = benchmark(&net, avoidance(&connection, 6, 128), false);

        points.push(Point {
            model_steps,
            depth_3_solved: depth_3.solve_rate(),
            depth_5_solved: depth_5.solve_rate(),
            depth_7_solved: depth_7.solve_rate(),
            depth_9_solved: depth_9.solve_rate(),
            depth_2_solved: depth_2.solve_rate(),
            depth_4_solved: depth_4.solve_rate(),
            depth_6_solved: depth_6.solve_rate(),
        });
    }

    graph(points, &args.graph_path);
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

fn tinue(connection: &Connection, depth: i64, limit: i64) -> Statement {
    let query = r#"SELECT * FROM puzzles
    WHERE instr(tps, "1C") > 0
        AND instr(tps, "2C") > 0
        AND tinue_length = :depth
        AND tinue_avoidance_length IS NULL
        AND tiltak_second_move_eval < 0.6
    ORDER BY game_id ASC
    LIMIT :limit"#;
    let mut statement = connection.prepare(query).unwrap();
    statement
        .bind::<&[(_, Value)]>(&[(":depth", depth.into()), (":limit", limit.into())])
        .unwrap();
    assert_eq!(
        statement.iter().count(),
        limit as usize,
        "incorrect amount of puzzles"
    );
    statement
}

fn avoidance(connection: &Connection, depth: i64, limit: i64) -> Statement {
    let query = r#"SELECT * FROM puzzles
    WHERE instr(tps, "1C") > 0
        AND instr(tps, "2C") > 0
        AND tinue_avoidance_length = :depth
        AND tinue_length IS NULL
        AND tiltak_eval > 0.6
    ORDER BY game_id ASC
    LIMIT :limit"#;
    let mut statement = connection.prepare(query).unwrap();
    statement
        .bind::<&[(_, Value)]>(&[(":depth", depth.into()), (":limit", limit.into())])
        .unwrap();
    assert_eq!(
        statement.iter().count(),
        limit as usize,
        "incorrect amount of puzzles"
    );
    statement
}

fn benchmark(agent: &Net, statement: Statement, win: bool) -> PuzzleResult {
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

    let mut batched_mcts = BatchedMCTS::from_envs(
        std::array::from_fn(|_| Env::default()),
        BETA,
        RndNormalizationContext::new(0.0),
    );

    // Attempt to solve puzzles.
    let mut attempted = 0;
    let mut solved = 0;
    let mut proven = 0;
    for (puzzle_batch, solution_batch) in puzzles
        .chunks_exact(BATCH_SIZE)
        .zip(solutions.chunks_exact(BATCH_SIZE))
    {
        println!("batch");
        batched_mcts
            .nodes_and_envs_mut()
            .zip(puzzle_batch)
            .for_each(|((node, env), puzzle)| {
                *node = Node::default();
                *env = puzzle.clone()
            });

        for _ in 0..VISITS {
            batched_mcts.simulate(agent);
        }

        attempted += puzzle_batch.len();
        solved += batched_mcts
            .select_best_actions()
            .into_iter()
            .zip(solution_batch)
            .filter(|(a, b)| a == *b)
            .count();

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

struct Point {
    model_steps: u32,
    // tinue
    depth_3_solved: f64,
    depth_5_solved: f64,
    depth_7_solved: f64,
    depth_9_solved: f64,
    // avoidance
    depth_2_solved: f64,
    depth_4_solved: f64,
    depth_6_solved: f64,
}

fn graph(mut points: Vec<Point>, path: &Path) {
    points.sort_by_key(|p| p.model_steps);
    let chart = Chart::new()
        .title(
            Title::new()
                .text(format!("Puzzle solve rate ({N}x{N}, {VISITS} visits)"))
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("training steps"))
        .y_axis(
            Axis::new()
                .name("ratio of puzzles solved")
                .min(0.0)
                .max(1.0),
        )
        .legend(Legend::new().top("bottom"))
        .series(
            Line::new().name("tinue in 3 solved").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_3_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("tinue in 5 solved").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_5_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("tinue in 7 solved").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_7_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("tinue in 9 solved").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_9_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("avoid in 2 solved").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_2_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("avoid in 4 solved").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_4_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("avoid in 6 proven").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_6_solved])
                    .collect(),
            ),
        );
    let mut renderer = HtmlRenderer::new("graph", 1200, 650).theme(Theme::Default);
    renderer
        .save(&chart, path.join("puzzle-graph.html"))
        .unwrap();
}
