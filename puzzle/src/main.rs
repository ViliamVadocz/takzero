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
        net6_simhash::{Env, Net, N},
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
    /// How many models to skip between benchmarks
    #[arg(long, default_value_t = 1)]
    step: usize,
}

const BATCH_SIZE: usize = 128;
const VISITS: u32 = 1024;
const ZERO_BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const DEVICE: Device = Device::Cuda(0);

fn main() {
    env_logger::init();
    log::info!("Begin.");
    tch::no_grad(real_main);
}

fn real_main() {
    let args = Args::parse();
    let connection = sqlite::open(&args.puzzle_db).unwrap();
    let mut paths: Vec<_> = read_dir(args.model)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "ot"))
        .filter(|p| {
            p.file_stem()
                .and_then(|s| {
                    let lossy = s.to_string_lossy();
                    let (model, steps) = lossy.split_once('_')?;
                    steps.parse::<u32>().ok()?;
                    Some(model == "model")
                })
                .unwrap_or_default()
        })
        .collect();
    paths.sort();
    let paths: Vec<_> = paths.into_iter().step_by(args.step).collect();

    let mut points = Vec::new();
    for path in paths {
        let Ok(net) = Net::load_partial(&path, DEVICE) else {
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

        log::info!("Depth 3");
        let depth_3 = benchmark(&net, tinue(&connection, 3), true);
        log::info!("Depth 5");
        let depth_5 = benchmark(&net, tinue(&connection, 5), true);
        log::info!("Depth 7");
        let depth_7 = benchmark(&net, tinue(&connection, 7), true);
        log::info!("Depth 9");
        let depth_9 = benchmark(&net, tinue(&connection, 9), true);
        
        log::info!("Depth 2");
        let depth_2 = benchmark(&net, avoidance(&connection, 2), false);
        log::info!("Depth 4");
        let depth_4 = benchmark(&net, avoidance(&connection, 4), false);
        log::info!("Depth 6");
        let depth_6 = benchmark(&net, avoidance(&connection, 6), false);

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

    graph(points, &args.graph);
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

        for _ in 0..VISITS {
            batched_mcts.simulate(agent, &ZERO_BETA);
        }

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
            Line::new().name("tinue in 3").data(
                points
                    .iter()
                    .map(|p| vec![f64::from(p.model_steps), p.depth_3_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("tinue in 5").data(
                points
                    .iter()
                    .map(|p| vec![f64::from(p.model_steps), p.depth_5_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("tinue in 7").data(
                points
                    .iter()
                    .map(|p| vec![f64::from(p.model_steps), p.depth_7_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("tinue in 9").data(
                points
                    .iter()
                    .map(|p| vec![f64::from(p.model_steps), p.depth_9_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("avoid in 2").data(
                points
                    .iter()
                    .map(|p| vec![f64::from(p.model_steps), p.depth_2_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("avoid in 4").data(
                points
                    .iter()
                    .map(|p| vec![f64::from(p.model_steps), p.depth_4_solved])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("avoid in 6").data(
                points
                    .iter()
                    .map(|p| vec![f64::from(p.model_steps), p.depth_6_solved])
                    .collect(),
            ),
        );
    let mut renderer = HtmlRenderer::new("graph", 1200, 650).theme(Theme::Default);
    renderer
        .save(&chart, path.join("puzzle-graph.html"))
        .unwrap();
}
