use std::{
    cmp::Ordering,
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
use fast_tak::{
    takparse::{Direction, Move, MoveKind, Piece, Square},
    Game,
    Reserves,
};
use rayon::prelude::*;
use sqlite::{Connection, Value};
use takzero::{
    network::{
        net5::{Env, Net, RndNormalizationContext, N},
        Network,
    },
    search::node::batched::BatchedMCTS,
};
use tch::Device;

#[derive(Parser, Debug)]
struct Args {
    /// Path where models are stored
    #[arg(long)]
    model_path: PathBuf,
    /// Path to puzzle database
    #[arg(long)]
    puzzle_db_path: PathBuf,
    /// Path to save graph
    #[arg(long)]
    graph_path: PathBuf,
}

const BATCH_SIZE: usize = 256;
const VISITS: u32 = 2024;
const BETA: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
const DEVICE: Device = Device::Cuda(0);

const DEPTH_3_LIMIT: Option<i64> = Some(256);
const DEPTH_5_LIMIT: Option<i64> = Some(256);

const STEP: usize = 5;

fn main() {
    env_logger::init();
    log::info!("Begin.");
    tch::no_grad(real_main);
}

fn real_main() {
    let args = Args::parse();
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
        log::info!("Benchmarking {}", path.display());
        let connection = sqlite::open(&args.puzzle_db_path).unwrap();
        let depth_3 = run_benchmark(&connection, 3, DEPTH_3_LIMIT, &net);
        let depth_5 = run_benchmark(&connection, 5, DEPTH_5_LIMIT, &net);
        if let Some(model_steps) = path
            .file_stem()
            .and_then(|p| p.to_str()?.split_once('_')?.1.parse().ok())
        {
            points.push(Point {
                model_steps,
                depth_3,
                depth_5,
            });
        }
    }

    graph(points, &args.graph_path);
}

fn run_benchmark(connection: &Connection, depth: i64, limit: Option<i64>, agent: &Net) -> f64
where
    Reserves<N>: Default,
{
    let query = "SELECT * FROM tinues t
    JOIN games g ON t.gameid = g.id
    WHERE t.size = :size
        AND g.komi = :komi
        AND t.tinue_depth = :depth
        AND g.id NOT IN (149657, 149584, 395154)
    ORDER BY t.id DESC
    LIMIT :limit";
    let mut statement = connection.prepare(query).unwrap();
    statement
        .bind::<&[(_, Value)]>(&[
            (":size", i64::try_from(N).unwrap().into()),
            (":komi", 0.into() /* i64::from(HALF_KOMI / 2).into() */),
            (":depth", depth.into()),
            (":limit", limit.unwrap_or(i64::MAX).into()),
        ])
        .unwrap();

    let rows = statement
        .into_iter()
        .map(|row| {
            row.map(|row| {
                let id: i64 = row.read("id");
                let notation: &str = row.read("notation");
                let plies_to_undo = row.read("plies_to_undo");
                let tinue: &str = row.read("tinue");
                (id, notation.to_owned(), plies_to_undo, tinue.to_owned())
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let row_num = rows.len();

    // Parse puzzles.
    let puzzles: Vec<Env> = rows
        .into_par_iter()
        .map(|(_game_id, notation, plies_to_undo, _tinue)| {
            let mut moves: Vec<_> = notation.split(',').map(parse_playtak_move).collect();
            for _ in 0..plies_to_undo {
                moves.pop();
            }
            Game::from_moves(&moves).unwrap()
        })
        .collect();

    let mut batched_mcts = BatchedMCTS::from_envs(
        std::array::from_fn(|_| Env::default()),
        BETA,
        RndNormalizationContext::new(0.0),
    );

    // Attempt to solve puzzles.
    let mut wins = 0;
    for puzzle_batch in puzzles.chunks_exact(BATCH_SIZE) {
        batched_mcts
            .nodes_and_envs_mut()
            .zip(puzzle_batch)
            .for_each(|((_, env), puzzle)| *env = puzzle.clone());

        for _ in 0..VISITS {
            batched_mcts.simulate(agent);
        }

        let proven_wins = batched_mcts
            .nodes_and_envs()
            .filter(|(node, _)| node.evaluation.is_win())
            .count();
        wins += proven_wins;
    }
    log::info!("depth {depth}: {wins: >5} / {row_num: >5}");
    wins as f64 / row_num as f64
}

fn parse_playtak_move(s: &str) -> Move {
    let mut words = s.split_ascii_whitespace();
    match (words.next(), words.next(), words.next()) {
        (Some("P"), Some(square), piece) => {
            Move::new(parse_square(square), MoveKind::Place(parse_piece(piece)))
        }
        (Some("M"), Some(start), Some(end)) => {
            let start = parse_square(start);
            let end = parse_square(end);
            let direction = match (
                end.column().cmp(&start.column()),
                end.row().cmp(&start.row()),
            ) {
                (Ordering::Less, Ordering::Equal) => Direction::Left,
                (Ordering::Greater, Ordering::Equal) => Direction::Right,
                (Ordering::Equal, Ordering::Less) => Direction::Down,
                (Ordering::Equal, Ordering::Greater) => Direction::Up,
                _ => panic!("start and end squares don't form a straight line"),
            };
            let pattern = words.collect::<String>().parse().unwrap();
            Move::new(start, MoveKind::Spread(direction, pattern))
        }
        _ => panic!("unrecognized move"),
    }
}

fn parse_square(s: &str) -> Square {
    s.to_lowercase().parse().unwrap()
}

fn parse_piece(s: Option<&str>) -> Piece {
    match s {
        None => Piece::Flat,
        Some("W") => Piece::Wall,
        Some("C") => Piece::Cap,
        _ => panic!("unknown piece"),
    }
}

struct Point {
    model_steps: u32,
    depth_3: f64,
    depth_5: f64,
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
        .legend(Legend::new().right(0.0))
        .series(
            Line::new().name("tinue in 3").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_3])
                    .collect(),
            ),
        )
        .series(
            Line::new().name("tinue in 5").data(
                points
                    .iter()
                    .map(|p| vec![p.model_steps as f64, p.depth_5])
                    .collect(),
            ),
        );
    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer
        .save(&chart, path.join("puzzle-graph.html"))
        .unwrap();
}
