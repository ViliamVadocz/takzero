use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::Path,
};

use charming::{
    component::{Axis, Grid, Legend, Title},
    element::Symbol,
    series::Line,
    theme::Theme,
    Chart,
    HtmlRenderer,
};
use takzero::{network::net5::Env, search::env::Environment, target::Replay};

fn main() {
    let chart = Chart::new()
        .title(
            Title::new()
                .text("Ratio of Unique Positions Seen in 100k Chunks")
                .subtext("Accounting for Symmetries")
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("Positions"))
        .y_axis(Axis::new().name("Ratio"))
        .grid(Grid::new())
        .legend(
            Legend::new()
                .data(vec![
                    "undirected-random",
                    "directed-random",
                    "directed-random-top8",
                    "undirected-seq-hal-16",
                ])
                .bottom(10)
                .left(10),
        )
        .series(
            Line::new()
                .data(get_unique_positions("undirected.txt"))
                .name("undirected-random")
                .symbol(Symbol::None),
        )
        .series(
            Line::new()
                .data(get_unique_positions("directed.txt"))
                .name("directed-random")
                .symbol(Symbol::None),
        )
        .series(
            Line::new()
                .data(get_unique_positions("directed-top.txt"))
                .name("directed-random-top8")
                .symbol(Symbol::None),
        )
        .series(
            Line::new()
                .data(get_unique_positions("undirected-seq-hal-16.txt"))
                .name("undirected-seq-hal-16")
                .symbol(Symbol::None),
        );

    let mut renderer = HtmlRenderer::new("graph", 1200, 800).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}

fn get_unique_positions(path: impl AsRef<Path>) -> Vec<Vec<f64>> {
    let file = OpenOptions::new().read(true).open(path).unwrap();
    let mut positions: HashMap<_, u64> = HashMap::new();
    let mut points = Vec::with_capacity(4096);
    let mut i = 0;
    for line in BufReader::new(file).lines() {
        if i / 100_000 > points.len() {
            points.push(vec![
                i as f64,
                positions.keys().len() as f64 / positions.values().sum::<u64>() as f64,
            ]);
            positions.clear();
        }
        let Ok(replay): Result<Replay<Env>, _> = line.unwrap().parse() else {
            println!("skipping line while at {i} positions");
            continue;
        };
        let mut env = replay.env;
        *positions.entry(env.clone().canonical()).or_default() += 1;
        i += 1;
        for action in replay.actions {
            env.step(action);
            *positions.entry(env.clone().canonical()).or_default() += 1;
            i += 1;
        }
    }
    // println!("unique positions: {}", positions.keys().len());
    // println!("total positions: {}", positions.values().sum::<u64>());
    points
}
