use std::cmp::Ordering;

use fast_tak::{
    takparse::{Direction, Move, MoveKind, Piece, Square},
    Game,
};
use mimalloc::MiMalloc;
use sqlite::Value;
use takzero::search::{agent::dummy::Dummy, eval::Eval, mcts::Node};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    const SIZE: usize = 4;
    const LIMIT: i64 = 1000;
    const HALF_KOMI: i8 = 0;

    const VISITS_PER_PUZZLE: usize = 10_000;

    let connection = sqlite::open("puzzle/games.db").unwrap();
    let query = "SELECT * FROM tinues t
                        JOIN games g ON t.gameid = g.id
                        WHERE t.size = :size AND g.komi = :komi
                        ORDER BY t.id
                        LIMIT :limit";
    let mut statement = connection.prepare(query).unwrap();
    statement
        .bind::<&[(_, Value)]>(&[
            (":size", (SIZE as i64).into()),
            (":komi", (HALF_KOMI as i64 / 2).into()),
            (":limit", LIMIT.into()),
        ])
        .unwrap();

    let mut actions = Vec::new();
    let mut wins = 0;
    let mut fails = 0;
    let mut rows = statement.into_iter();
    while let Some(Ok(row)) = rows.next() {
        let notation: &str = row.read("notation");
        let plies_to_remove: i64 = row.read("plies_to_undo");
        let tinue: &str = row.read("tinue");

        let mut moves: Vec<_> = notation.split(',').map(parse_playtak_move).collect();
        for _ in 0..plies_to_remove {
            moves.pop();
        }
        let game: Game<SIZE, HALF_KOMI> = Game::from_moves(&moves).unwrap();

        let mut root = Node::default();
        if (0..VISITS_PER_PUZZLE).any(|_| {
            matches!(
                root.simulate(game.clone(), &mut actions, &Dummy),
                Eval::Win(_)
            )
        }) {
            wins += 1;
        } else {
            fails += 1;
        }

        println!("{wins}W : {fails}L");
        println!("tinue: {tinue}");
        println!("root: {root}");
    }
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
