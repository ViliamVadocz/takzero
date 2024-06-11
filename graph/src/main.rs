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
        // "undirected",
        "lcghash",
        // "lcghash_larger_beta",
        "lcghash_no_window",
        // "lcghash_no_window_smaller_beta",
        // "simhash_no_window_smaller_beta",
    ];

    let mut chart = Chart::new()
        .title(
            Title::new()
                .text("Ratio of Unique Positions in Chunk Seen During Training")
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
                .data(get_unique_positions(format!("4x4_{r}_replays.txt")))
                .name(r)
                .symbol(Symbol::None),
        );
    }
    let mut renderer = HtmlRenderer::new("graph", 1200, 800).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}

fn get_unique_positions(path: impl AsRef<Path>) -> Vec<Vec<f64>> {
    let mut positions: HashMap<_, u64> = HashMap::with_capacity(20_000_000);
    let mut points = vec![vec![0.0, 1.0]];
    let mut positions_count = 0;
    let mut previous_positions_count = 0;
    let mut previous_unique_positions_count = 0;

    for replay in get_replays::<N, HALF_KOMI>(path).unwrap() {
        if positions_count / 250_000 >= points.len() {
            println!("{positions_count} positions");

            let unique_positions = positions.keys().len();
            points.push(vec![
                positions_count as f64,
                (unique_positions - previous_unique_positions_count) as f64
                    / (positions_count - previous_positions_count) as f64,
            ]);
            previous_positions_count = positions_count;
            previous_unique_positions_count = unique_positions;

            // positions.clear();
        }
        if points.len() > 80 {
            break;
        }

        let mut env = replay.env;
        *positions.entry(env.clone() /* .canonical() */).or_default() += 1;
        positions_count += 1;
        for action in replay.actions {
            env.step(action);
            *positions.entry(env.clone() /* .canonical() */).or_default() += 1;
            positions_count += 1;
        }
    }

    // println!("unique positions: {}", positions.keys().len());
    // println!("total positions: {}", positions.values().sum::<u64>());
    points
}
