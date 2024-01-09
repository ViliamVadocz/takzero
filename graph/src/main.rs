use std::{
    env,
    io::{self, Write},
    path::Path,
    process::{Command, Stdio},
};

use charming::{
    component::{Axis, Legend, Title},
    series::Line,
    theme::Theme,
    Chart,
    HtmlRenderer,
};

#[derive(Debug, Clone, Copy)]
struct Match {
    white: u32,
    black: u32,
    white_wins: u32,
    black_wins: u32,
    draws: u32,
}

const STEPS_PER_MODEL: u32 = 1000;

fn main() {
    let mut chart = Chart::new()
        .title(
            Title::new()
                .text("Elo gain during training (5x5)")
                .left("center")
                .top(0),
        )
        .x_axis(Axis::new().name("training steps"))
        .y_axis(Axis::new().name("estimated Elo"))
        .legend(Legend::new().right(0.0));

    assert!(
        env::args().count() > 1,
        "maybe you forgot to supply a path to the match results"
    );

    for path in env::args().skip(1) {
        let matches = get_matches(path);
        let players = get_unique_players(&matches);
        let player_elo = get_bayes_elo(players, matches).expect("Could not calculate Bayes Elo");
        chart = chart.series(
            Line::new().data(
                player_elo
                    .into_iter()
                    .map(|(p, e)| vec![f64::from(p), f64::from(e)])
                    .collect(),
            ),
        );
    }

    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}

fn get_matches(path: impl AsRef<Path>) -> Vec<Match> {
    let contents = std::fs::read_to_string(path).expect("path should be valid and readable");
    contents
        .lines()
        .map(|line| {
            let data = line
                .split(',')
                .map(str::trim)
                .map(str::parse)
                .collect::<Result<Vec<u32>, _>>()
                .expect("the file should contain comma separated integers");
            Match {
                white: data[0],
                black: data[1],
                white_wins: data[2],
                black_wins: data[3],
                draws: data[4],
            }
        })
        .collect()
}

fn get_unique_players(matches: &[Match]) -> Vec<u32> {
    let mut players: Vec<_> = matches
        .iter()
        .flat_map(|m| [m.white, m.black].into_iter())
        .collect();
    players.sort_unstable();
    players.dedup();
    assert!(
        players
            .iter()
            .enumerate()
            .all(|(i, p)| p / STEPS_PER_MODEL == i as u32),
        "the rest of the code assumes there are no gaps in players"
    );
    players
}

fn get_bayes_elo(players: Vec<u32>, matches: Vec<Match>) -> Result<Vec<(u32, i32)>, io::Error> {
    // https://www.remi-coulom.fr/Bayesian-Elo/
    let mut bayeselo = Command::new(".\\graph\\bayeselo.exe")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    let stdin = bayeselo.stdin.as_mut().expect("stdin should be piped");
    writeln!(stdin, "prompt off")?;
    for player in players {
        writeln!(stdin, "addplayer {player}_steps")?;
    }
    for m in matches {
        let white = m.white / STEPS_PER_MODEL;
        let black = m.black / STEPS_PER_MODEL;
        for _win in 0..m.white_wins {
            writeln!(stdin, "addresult {white} {black} 2")?;
        }
        for _loss in 0..m.black_wins {
            writeln!(stdin, "addresult {white} {black} 0")?;
        }
        for _draw in 0..m.draws {
            writeln!(stdin, "addresult {white} {black} 1")?;
        }
    }

    writeln!(stdin, "elo")?; // enter elo interface
    writeln!(stdin, "mm")?; // compute maximum-likelyhood elo
    writeln!(stdin, "ratings")?; // print out ratings
    writeln!(stdin, "x")?; // leave elo interface
    writeln!(stdin, "x")?; // close application

    let out = bayeselo.wait_with_output()?;
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

    // let min = player_elo.iter().map(|v| v.1).min().unwrap();
    // player_elo.iter_mut().for_each(|v| v.1 -= min);
    player_elo.sort_by_key(|(i, _)| *i);

    Ok(player_elo)
}
