use std::collections::HashMap;

use charming::{
    component::{Axis, Title},
    series::Scatter,
    theme::Theme,
    Chart,
    HtmlRenderer,
};
use rand::seq::SliceRandom;

const RESULTS_PATH: &str = "match_results.csv";
const DEFAULT_ELO: f64 = 1000.0;
const K: f64 = 40.0;
const DIFF_DIVIDER: f64 = 400.0;

#[derive(Debug, Clone, Copy)]
struct Match {
    white: u32,
    black: u32,
    white_wins: f64,
    black_wins: f64,
    draws: f64,
}

fn expected_outcome(elo_a: f64, elo_b: f64) -> f64 {
    let diff = elo_b - elo_a;
    1.0 / (1.0 + 10.0f64.powf(diff / DIFF_DIVIDER))
}

fn update_elo(elo: f64, outcome: f64, expected_outcome: f64) -> f64 {
    elo + K * (outcome - expected_outcome)
}

fn calculate_elo(
    elo_ratings: &mut HashMap<u32, f64>,
    Match {
        white,
        black,
        white_wins,
        black_wins,
        draws,
    }: Match,
) {
    let white_elo = *elo_ratings.get(&white).unwrap_or(&DEFAULT_ELO);
    let black_elo = *elo_ratings.get(&black).unwrap_or(&DEFAULT_ELO);
    let total_games = white_wins + black_wins + draws;
    let white_outcome = (white_wins + 0.5 * draws) / total_games;
    let black_outcome = (black_wins + 0.5 * draws) / total_games;
    let white_expected = expected_outcome(white_elo, black_elo);
    let black_expected = expected_outcome(black_elo, white_elo);
    elo_ratings.insert(white, update_elo(white_elo, white_outcome, white_expected));
    elo_ratings.insert(black, update_elo(black_elo, black_outcome, black_expected));
}

fn main() {
    let contents = std::fs::read_to_string(RESULTS_PATH).unwrap();
    let mut matches: Vec<_> = contents
        .lines()
        .map(|line| {
            let data = line
                .split(',')
                .map(str::trim)
                .map(str::parse)
                .collect::<Result<Vec<u32>, _>>()
                .unwrap();
            let white = data[0];
            let black = data[1];
            let white_wins = data[2] as f64;
            let black_wins = data[3] as f64;
            let draws = data[4] as f64;
            Match {
                white,
                black,
                white_wins,
                black_wins,
                draws,
            }
        })
        .collect();
    matches.shuffle(&mut rand::thread_rng());
    let mut elo_ratings: HashMap<_, _> = matches.iter().map(|m| (m.white, DEFAULT_ELO)).collect();
    for m in matches {
        calculate_elo(&mut elo_ratings, m);
    }

    let chart = Chart::new()
        .title(
            Title::new()
                .text("Elo gain during training (4x4)")
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("training steps"))
        .y_axis(Axis::new().name("estimated Elo"))
        .series(
            Scatter::new().data(
                elo_ratings
                    .into_iter()
                    .map(|(i, elo)| vec![i as f64, elo])
                    .collect(),
            ),
        );

    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}
