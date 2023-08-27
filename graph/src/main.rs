use charming::{
    component::{Axis, Title},
    series::Scatter,
    theme::Theme,
    Chart,
    HtmlRenderer,
};

struct Data {
    a: usize,
    b: usize,
    wins: u32,
    losses: u32,
    draws: u32,
}

impl Data {
    const fn new(a: usize, b: usize, wins: u32, losses: u32, draws: u32) -> Self {
        Self {
            a,
            b,
            wins,
            losses,
            draws,
        }
    }
}

const DATA: [Data; 41] = [
    Data::new(0, 1, 426, 7, 47),
    Data::new(1, 2, 298, 143, 39),
    Data::new(2, 3, 358, 102, 20),
    Data::new(3, 4, 347, 122, 11),
    Data::new(4, 5, 326, 151, 3),
    Data::new(5, 6, 297, 182, 1),
    Data::new(6, 7, 284, 194, 2),
    Data::new(7, 8, 256, 221, 3),
    Data::new(8, 9, 261, 218, 1),
    Data::new(9, 10, 283, 197, 0),
    Data::new(10, 11, 266, 214, 0),
    Data::new(11, 12, 250, 230, 0),
    Data::new(12, 13, 252, 228, 0),
    Data::new(13, 14, 262, 218, 0),
    Data::new(14, 15, 278, 202, 0),
    Data::new(15, 16, 224, 256, 0),
    Data::new(15, 17, 217, 263, 0),
    Data::new(15, 18, 237, 243, 0),
    Data::new(15, 19, 248, 232, 0),
    Data::new(19, 20, 242, 237, 1),
    Data::new(20, 21, 257, 222, 1),
    Data::new(21, 22, 250, 229, 1),
    Data::new(22, 23, 237, 242, 1),
    Data::new(22, 24, 272, 208, 0),
    Data::new(24, 25, 246, 229, 5),
    Data::new(25, 26, 252, 228, 0),
    Data::new(26, 27, 267, 213, 0),
    Data::new(27, 28, 221, 257, 2),
    Data::new(27, 29, 209, 269, 2),
    Data::new(27, 30, 239, 239, 2),
    Data::new(27, 31, 238, 241, 1),
    Data::new(27, 32, 214, 264, 2),
    Data::new(27, 33, 212, 267, 1),
    Data::new(27, 34, 250, 230, 0),
    Data::new(34, 35, 237, 242, 1),
    Data::new(34, 36, 281, 196, 3),
    Data::new(36, 37, 261, 212, 7),
    Data::new(37, 38, 248, 231, 1),
    Data::new(38, 39, 234, 243, 3),
    Data::new(38, 40, 254, 222, 4),
    Data::new(40, 41, 244, 233, 3),
    // TODO: Add more data here when job finishes.
];

fn main() {
    let mut elo = vec![0.0; DATA.len() + 1];
    for data in DATA {
        elo[data.b] = elo[data.a]
            + 400. * (data.wins as f64 - data.losses as f64)
                / (data.wins + data.draws + data.losses) as f64;
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
                elo.into_iter()
                    .enumerate()
                    .map(|(i, elo)| vec![(i * 100) as f64, elo])
                    .collect(),
            ),
        );

    let mut renderer = HtmlRenderer::new("graph", 1000, 600).theme(Theme::Infographic);
    renderer.save(&chart, "graph.html").unwrap();
}
