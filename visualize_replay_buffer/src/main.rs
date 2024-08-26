use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::Path,
};

use fast_tak::takparse::{Move, MoveKind, Piece, Square};
use rand::prelude::*;
use takzero::{
    network::net6_simhash::{Env, N},
    search::env::Environment,
    target::get_replays,
};

const ACTIONS: usize = 5;

fn load_positions(all_positions: &mut HashSet<Env>, path: impl AsRef<Path>) -> HashMap<Env, u32> {
    let mut positions = HashMap::new();
    for replay in get_replays(path).unwrap().take(250_000) {
        let mut env = replay.env;
        for action in replay.actions.into_iter().take(ACTIONS + 1) {
            let canonical = env.clone().canonical();
            all_positions.insert(canonical.clone());
            *positions.entry(canonical).or_default() += 1;
            env.step(action);
        }
    }
    positions
}

#[allow(unused)]
fn count_state_space_at_depths() {
    let mut args = std::env::args();
    let _process = args.next();

    let mut all_positions = HashSet::new();
    let files = [
        "neurips_directed_00",
        "simhash_directed_00",
        "neurips_undirected_00", // no sampling
        "neurips_undirected_01", // sampling all
        "neurips_undirected_03", // filtered sampling
    ];
    let positions: Vec<_> = files
        .into_iter()
        .map(|path| {
            (
                path,
                load_positions(&mut all_positions, format!("4x4_{path}_replays.txt")),
            )
        })
        .collect();

    let mut queue = VecDeque::new();

    let a1 = Square::new(0, 0);
    let an = Square::new(0, N as u8 - 1);
    let xn = Square::new(N as u8 - 1, N as u8 - 1);
    for opening in [[a1, an], [a1, xn]] {
        let mut env = Env::default();
        for square in opening {
            env.step(Move::new(square, MoveKind::Place(Piece::Flat)));
        }
        queue.push_back(env.canonical());
    }

    let mut layers: Vec<_> = positions.iter().map(|_| vec![vec![]; ACTIONS]).collect();

    let mut actions = Vec::new();
    while let Some(env) = queue.pop_front() {
        env.populate_actions(&mut actions);
        for a in actions.drain(..) {
            let mut clone = env.clone();
            clone.step(a);
            let canonical = clone.canonical();

            for (i, (_, hash)) in positions.iter().enumerate() {
                if let Some(x) = hash.get(&canonical) {
                    layers[i][(env.ply - 2) as usize].push(x);
                }
            }

            if canonical.ply - 2 < ACTIONS as u16 {
                queue.push_back(canonical);
            }
        }
    }

    for (i, (name, _)) in positions.iter().enumerate() {
        println!("{name}");
        for (ii, layer) in layers[i].iter().enumerate() {
            println!("{}: {}", ii + 2, layer.len());
        }
    }
}

// let mut positions: Vec<Env> = all_positions.into_iter().collect();
// positions.sort_by_key(|p| Reverse(p.white_reserves.stones +
// p.black_reserves.stones));

// println!("There are {} unique positions.", positions.len());
// #[allow(clippy::cast_sign_loss)]
// let size = (positions.len() as f64).sqrt().ceil() as usize;
// println!("The image will be {size}x{size}.");

// let mut pixel_data = vec![0u8; size * size * 3];
// for (i, position) in positions.into_iter().enumerate() {
//     let r = if d0.contains_key(&position) { 255 } else { 0 };
//     let g = if d1.contains_key(&position) { 255 } else { 0 };
//     let b = if u0.contains_key(&position) { 255 } else { 0 };

//     let offset = i * 3;
//     pixel_data[offset] = r;
//     pixel_data[offset + 1] = g;
//     pixel_data[offset + 2] = b;
// }

// let img = RgbImage::from_raw(size as u32, size as u32, pixel_data).unwrap();
// img.save("test.png").unwrap();

fn get_positions(path: impl AsRef<Path>) -> Result<impl Iterator<Item = Env>, std::io::Error> {
    get_replays(path).map(|replays| replays.flat_map(|replay| replay.states().collect::<Vec<_>>()))
}

fn sample_positions_into_set(
    path: impl AsRef<Path>,
    amount: usize,
    rng: &mut impl Rng,
) -> HashSet<Env> {
    get_positions(path)
        .expect("Path to replays should be valid")
        .choose_multiple(rng, amount)
        .into_iter()
        .collect()
}

fn main() {
    const AMOUNT: usize = 1_000_000;
    const SEED: u64 = 12345;
    let mut rng = StdRng::seed_from_u64(SEED);

    let positions_a = sample_positions_into_set("undirected_replays.txt", AMOUNT, &mut rng);
    let positions_b = sample_positions_into_set("naive_replays.txt", AMOUNT, &mut rng);

    let positions_both = positions_a.intersection(&positions_b);
    let positions_unique_a = positions_a.difference(&positions_b);
    let positions_unique_b = positions_b.difference(&positions_a);

    println!(
        "{} {} {} {} {}",
        positions_a.len(),          // 949082
        positions_b.len(),          // 967876
        positions_both.count(),     // 7660
        positions_unique_a.count(), // 941422
        positions_unique_b.count()  // 960216
    );
}
