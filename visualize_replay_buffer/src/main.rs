use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet, VecDeque},
    path::Path,
};

use fast_tak::takparse::{Move, MoveKind, Piece, Square};
use image::RgbImage;
use takzero::{
    network::net4_neurips::{Env, N},
    search::env::Environment,
    target::get_replays,
};

const ACTIONS: usize = 5;

fn load_positions(all_positions: &mut HashSet<Env>, path: impl AsRef<Path>) -> HashMap<Env, u32> {
    let mut positions = HashMap::new();
    for replay in get_replays(path).unwrap() {
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

fn main() {
    let mut args = std::env::args();
    let _process = args.next();

    let mut all_positions = HashSet::new();
    let d0 = load_positions(&mut all_positions, "4x4_old_directed_00_replays.txt");
    println!("done loading d0");
    let d1 = load_positions(&mut all_positions, "4x4_old_directed_01_replays.txt");
    println!("done loading d1");
    let u0 = load_positions(&mut all_positions, "4x4_neurips_undirected_00_replays.txt");
    println!("done loading u0");

    let mut positions: Vec<Env> = all_positions.into_iter().collect();
    positions.sort_by_key(|p| Reverse(p.white_reserves.stones + p.black_reserves.stones));

    println!("There are {} unique positions.", positions.len());
    #[allow(clippy::cast_sign_loss)]
    let size = (positions.len() as f64).sqrt().ceil() as usize;
    println!("The image will be {size}x{size}.");

    let mut pixel_data = vec![0u8; size * size * 3];
    for (i, position) in positions.into_iter().enumerate() {
        let r = if d0.contains_key(&position) { 255 } else { 0 };
        let g = if d1.contains_key(&position) { 255 } else { 0 };
        let b = if u0.contains_key(&position) { 255 } else { 0 };

        let offset = i * 3;
        pixel_data[offset] = r;
        pixel_data[offset + 1] = g;
        pixel_data[offset + 2] = b;
    }

    let img = RgbImage::from_raw(size as u32, size as u32, pixel_data).unwrap();
    img.save("test.png").unwrap();

    // ---

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

    let mut layers_d0 = vec![vec![]; ACTIONS];
    let mut layers_d1 = vec![vec![]; ACTIONS];
    let mut layers_u0 = vec![vec![]; ACTIONS];

    let mut actions = Vec::new();
    while let Some(env) = queue.pop_front() {
        env.populate_actions(&mut actions);
        for a in actions.drain(..) {
            let mut clone = env.clone();
            clone.step(a);
            let canonical = clone.canonical();

            if let Some(x) = d0.get(&canonical) {
                layers_d0[(env.ply - 2) as usize].push(x);
            }
            if let Some(x) = d1.get(&canonical) {
                layers_d1[(env.ply - 2) as usize].push(x);
            }
            if let Some(x) = u0.get(&canonical) {
                layers_u0[(env.ply - 2) as usize].push(x);
            }

            if canonical.ply - 2 < ACTIONS as u16 {
                queue.push_back(canonical);
            }
        }
    }

    println!("d0");
    for (i, layer) in layers_d0.into_iter().enumerate() {
        println!("{}: {}", i + 2, layer.len());
    }

    println!("d1");
    for (i, layer) in layers_d1.into_iter().enumerate() {
        println!("{}: {}", i + 2, layer.len());
    }

    println!("u0");
    for (i, layer) in layers_u0.into_iter().enumerate() {
        println!("{}: {}", i + 2, layer.len());
    }
}
