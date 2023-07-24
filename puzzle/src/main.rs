use std::cmp::Ordering;

use fast_tak::{
    takparse::{Direction, Move, MoveKind, Piece, Square},
    Game,
    Reserves,
};
use mimalloc::MiMalloc;
use rayon::prelude::*;
use sqlite::{Connection, Value};
use takzero::search::{agent::dummy::Dummy, eval::Eval, mcts::Node};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    const VISITS_PER_PUZZLE: usize = 10_000;

    let mut connection = sqlite::open("puzzle/games.db").unwrap();
    run_benchmark::<4, 0>(&mut connection, 3, VISITS_PER_PUZZLE, None);
    run_benchmark::<4, 0>(&mut connection, 5, VISITS_PER_PUZZLE, None);
    run_benchmark::<5, 0>(&mut connection, 3, VISITS_PER_PUZZLE, Some(10_000));
    run_benchmark::<5, 0>(&mut connection, 5, VISITS_PER_PUZZLE, Some(10_000));
    run_benchmark::<6, 0>(&mut connection, 3, VISITS_PER_PUZZLE, Some(5_000));
    run_benchmark::<6, 0>(&mut connection, 5, VISITS_PER_PUZZLE, Some(5_000));
    run_benchmark::<7, 0>(&mut connection, 3, VISITS_PER_PUZZLE, None);
    run_benchmark::<7, 0>(&mut connection, 5, VISITS_PER_PUZZLE, None);
}

fn run_benchmark<const N: usize, const HALF_KOMI: i8>(
    connection: &mut Connection,
    depth: i64,
    visits: usize,
    limit: Option<i64>,
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
                let id = row.read("id");
                let notation: &str = row.read("notation");
                let plies_to_undo = row.read("plies_to_undo");
                let tinue: &str = row.read("tinue");
                (id, notation.to_owned(), plies_to_undo, tinue.to_owned())
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let row_num = rows.len();

    let wins = rows
        .into_par_iter()
        .filter(|(game_id, notation, plies_to_undo, tinue)| {
            solve_puzzle::<N, HALF_KOMI>(*game_id, notation, *plies_to_undo, tinue, visits)
        })
        .count();

    println!("{N}x{N}, depth: {depth}");
    println!("{wins: >5} / {row_num: >5}");
}

fn solve_puzzle<const N: usize, const HALF_KOMI: i8>(
    game_id: i64,
    notation: &str,
    plies_to_remove: i64,
    _tinue: &str,
    visits: usize,
) -> bool
where
    Reserves<N>: Default,
{
    let mut moves: Vec<_> = notation.split(',').map(parse_playtak_move).collect();
    for _ in 0..plies_to_remove {
        moves.pop();
    }
    let game: Game<N, HALF_KOMI> = match Game::from_moves(&moves) {
        Ok(game) => game,
        Err(e) => {
            eprintln!("error: {e}, id: {game_id}");
            return false;
        }
    };

    let mut root = Node::default();
    let mut actions = Vec::new();
    (0..visits).any(|_| {
        matches!(
            root.simulate(game.clone(), &mut actions, &Dummy),
            Eval::Win(_)
        )
    })
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
