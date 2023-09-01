use std::{array, cmp::Ordering, fs::read_dir, path::PathBuf};

use clap::Parser;
use rayon::prelude::*;
use sqlite::{Connection, Value};
use takzero::{
    fast_tak::{
        takparse::{Direction, Move, MoveKind, Piece, Square},
        Game,
        Reserves,
    },
    network::{net4::Net4, Network},
    search::{
        agent::Agent,
        node::{gumbel::gumbel_sequential_halving, Node},
    },
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
}

const N: usize = 4;
const HALF_KOMI: i8 = 0;
type Net = Net4;
const BATCH_SIZE: usize = 256;
const SAMPLED: usize = usize::MAX;
const SIMULATIONS: u32 = 1024;

fn main() {
    env_logger::init();
    log::info!("Begin.");

    let args = Args::parse();
    let mut paths: Vec<_> = read_dir(args.model_path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();

    for path in paths {
        let Ok(net) = Net4::load(&path, Device::Cuda(0)) else {
            log::warn!("Cannot load {}", path.display());
            continue;
        };
        log::info!("Benchmarking {}", path.display());
        let connection = sqlite::open(&args.puzzle_db_path).unwrap();
        run_benchmark::<N, HALF_KOMI, Net>(&connection, 3, None, &net);
        run_benchmark::<N, HALF_KOMI, Net>(&connection, 5, None, &net);
    }
}

fn run_benchmark<const N: usize, const HALF_KOMI: i8, A: Agent<Game<N, HALF_KOMI>>>(
    connection: &Connection,
    depth: i64,
    limit: Option<i64>,
    agent: &A,
) where
    Reserves<N>: Default,
{
    let query = "SELECT * FROM tinues t
    JOIN games g ON t.gameid = g.id
    WHERE t.size = :size
        AND g.komi = :komi
        AND t.tinue_depth = :depth
        AND g.id NOT IN (149657, 149584, 395154)
    ORDER BY t.id
    LIMIT :limit";
    let mut statement = connection.prepare(query).unwrap();
    statement
        .bind::<&[(_, Value)]>(&[
            (":size", i64::try_from(N).unwrap().into()),
            (":komi", i64::from(HALF_KOMI / 2).into()),
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
    let puzzles: Vec<Game<N, HALF_KOMI>> = rows
        .into_par_iter()
        .map(|(_game_id, notation, plies_to_undo, _tinue)| {
            let mut moves: Vec<_> = notation.split(',').map(parse_playtak_move).collect();
            for _ in 0..plies_to_undo {
                moves.pop();
            }
            Game::from_moves(&moves).unwrap()
        })
        .collect();

    // Attempt to solve puzzles.
    let mut wins = 0;
    for envs in puzzles.chunks(BATCH_SIZE) {
        let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
        let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        let _top_actions = gumbel_sequential_halving(
            &mut nodes[..envs.len()],
            envs,
            agent,
            SAMPLED,
            SIMULATIONS,
            &mut actions[..envs.len()],
            &mut trajectories[..envs.len()],
            None::<&mut rand::rngs::ThreadRng>,
        );
        let local_wins = nodes
            .iter()
            .take(envs.len())
            .filter(|env| env.evaluation.is_win())
            .count();
        wins += local_wins;
    }

    log::info!("depth {depth}: {wins: >5} / {row_num: >5}");
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
