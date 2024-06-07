use std::{collections::HashMap, path::Path};

use charming::{
    component::{Axis, Grid, Legend, Title},
    element::Symbol,
    series::Line,
    theme::Theme,
    Chart,
    HtmlRenderer,
};
use takzero::{
    network::net4_neurips::{HALF_KOMI, N},
    search::env::Environment,
    target::get_replays,
};

fn main() {
    let replays = [
        "undirected_greedy",
        "undirected_sampled",
        "undirected_filtered",
        // "rnd_sampled",
        // "lcghash_filtered_00",
        // "simhash_filtered_00",
        "lcghash_filtered",
        "simhash_filtered",
    ];

    let mut chart = Chart::new()
        .title(
            Title::new()
                .text("Ratio of Unique Positions Seen During Training")
                .subtext("Accounting for Symmetries")
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("Positions"))
        .y_axis(Axis::new().name("Ratio"))
        .grid(Grid::new())
        .legend(Legend::new().data(replays.to_vec()).bottom(10).left(10));
    for r in replays {
        chart = chart.series(
            Line::new()
                .data(get_unique_positions(format!("4x4_{r}.txt")))
                .name(r)
                .symbol(Symbol::None),
        );
    }
    let mut renderer = HtmlRenderer::new("graph", 1200, 800).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}

fn get_unique_positions(path: impl AsRef<Path>) -> Vec<Vec<f64>> {
    let mut positions: HashMap<_, u64> = HashMap::new();
    let mut points = vec![vec![0.0, 1.0]];
    let mut i = 0;

    for replay in get_replays::<N, HALF_KOMI>(path).unwrap() {
        if i / 250_000 > points.len() {
            println!("{i} positions");
            points.push(vec![
                i as f64,
                positions.keys().len() as f64 / positions.values().sum::<u64>() as f64,
            ]);
            // positions.clear();
        }
        if points.len() > 80 {
            break;
        }

        let mut env = replay.env;
        *positions.entry(env.clone() /* .canonical() */).or_default() += 1;
        i += 1;
        for action in replay.actions {
            env.step(action);
            *positions.entry(env.clone() /* .canonical() */).or_default() += 1;
            i += 1;
        }
    }

    // println!("unique positions: {}", positions.keys().len());
    // println!("total positions: {}", positions.values().sum::<u64>());
    points
}
