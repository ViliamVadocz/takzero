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
    network::net6_simhash::{HALF_KOMI, N},
    search::env::Environment,
    target::get_replays,
};

fn main() {
    let replays = ["undirected", "naive", "directed"];

    let mut chart = Chart::new()
        .title(
            Title::new()
                .text("Ratio of Unique Positions in Chunk Seen During Training")
                // .subtext("Accounting for Symmetries")
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("Positions"))
        .y_axis(Axis::new().name("Ratio"))
        .grid(Grid::new())
        .legend(Legend::new().data(replays.to_vec()).bottom(10).left(10));
    for r in replays {
        let data = get_unique_positions(format!("./{r}_replays.txt"));
        println!("{r} = [");
        for p in &data {
            println!("    ({}, {}),", p[0], p[1]);
        }
        println!("]");
        chart = chart.series(Line::new().data(data).name(r).symbol(Symbol::None));
    }
    let mut renderer = HtmlRenderer::new("graph", 1200, 800).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}

fn get_unique_positions(path: impl AsRef<Path>) -> Vec<Vec<f64>> {
    const POSITIONS: usize = 10_000_000;
    const POINT_RATE: usize = 250_000;
    const POINTS: usize = POSITIONS / POINT_RATE;

    let mut positions: HashMap<_, u64> = HashMap::with_capacity(POSITIONS);
    let mut points = vec![vec![0.0, 1.0]];
    let mut positions_count = 0;
    let mut previous_positions_count = 0;
    let mut previous_unique_positions_count = 0;

    for replay in get_replays::<N, HALF_KOMI>(path).unwrap() {
        if positions_count / POINT_RATE >= points.len() {
            let position_count_diff = positions_count - previous_positions_count;

            let unique_positions = positions.keys().len();
            // Yaniv Metric
            points.push(vec![
                positions_count as f64,
                (unique_positions - previous_unique_positions_count) as f64
                    / position_count_diff as f64,
            ]);
            // Will Metric
            // points.push(vec![
            //     positions_count as f64,
            //     unique_positions as f64 / position_count_diff as f64,
            // ]);

            previous_positions_count = positions_count;
            previous_unique_positions_count = unique_positions;

            // Uncomment if using Will Metric.
            // positions.clear();
        }
        if points.len() > POINTS {
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
