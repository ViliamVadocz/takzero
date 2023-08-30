use std::{
    io::Write,
    process::{Command, Stdio},
};

use charming::{
    component::{Axis, Title},
    series::Scatter,
    theme::Theme,
    Chart,
    HtmlRenderer,
};

const RESULTS_PATH: &str = "match_results.csv";

#[derive(Debug, Clone, Copy)]
struct Match {
    white: u32,
    black: u32,
    white_wins: u32,
    black_wins: u32,
    draws: u32,
}

fn main() {
    let contents = std::fs::read_to_string(RESULTS_PATH).unwrap();
    let matches: Vec<_> = contents
        .lines()
        .map(|line| {
            let data = line
                .split(',')
                .map(str::trim)
                .map(str::parse)
                .collect::<Result<Vec<u32>, _>>()
                .unwrap();
            Match {
                white: data[0],
                black: data[1],
                white_wins: data[2],
                black_wins: data[3],
                draws: data[4],
            }
        })
        .collect();
    let mut players: Vec<_> = matches
        .iter()
        .flat_map(|m| [m.white, m.black].into_iter())
        .collect();
    players.sort();
    players.dedup();
    assert!(players.iter().enumerate().all(|(i, p)| p / 100 == i as u32));

    // https://www.remi-coulom.fr/Bayesian-Elo/
    let mut bayeselo = Command::new(".\\graph\\bayeselo.exe")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let stdin = bayeselo.stdin.as_mut().unwrap();
    writeln!(stdin, "prompt off").unwrap();
    for player in players {
        writeln!(stdin, "addplayer {player}_steps").unwrap();
    }
    for m in matches {
        let white = m.white / 100;
        let black = m.black / 100;
        for _win in 0..m.white_wins {
            writeln!(stdin, "addresult {white} {black} 2").unwrap();
        }
        for _loss in 0..m.black_wins {
            writeln!(stdin, "addresult {white} {black} 0").unwrap();
        }
        for _draw in 0..m.draws {
            writeln!(stdin, "addresult {white} {black} 1").unwrap();
        }
    }

    writeln!(stdin, "elo").unwrap(); // enter elo interface
    writeln!(stdin, "mm").unwrap(); // compute maximum-likelyhood elo
    writeln!(stdin, "ratings").unwrap(); // print out ratings
    writeln!(stdin, "x").unwrap(); // leave elo interface
    writeln!(stdin, "x").unwrap(); // close application

    let out = bayeselo.wait_with_output().unwrap();
    let mut player_elo: Vec<_> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .skip_while(|line| !line.starts_with("Rank"))
        .skip(1)
        .map(|line| -> Option<_> {
            let mut split = line.split_ascii_whitespace().skip(1);
            let player: u32 = split.next()?.split_once('_')?.0.parse().ok()?;
            let elo: i32 = split.next()?.parse().ok()?;
            Some((player, elo))
        })
        .collect::<Option<_>>()
        .unwrap();

    let min = player_elo.iter().map(|v| v.1).min().unwrap();
    player_elo.iter_mut().for_each(|v| v.1 -= min);

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
                player_elo
                    .into_iter()
                    .map(|(p, e)| vec![p as f64, e as f64])
                    .collect(),
            ),
        );

    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}
