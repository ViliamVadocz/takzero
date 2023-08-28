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

const ELO_SCALE: f64 = 400.;

const DATA: [Data; 130] = [
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
    Data::new(41, 42, 244, 234, 2),
    Data::new(42, 43, 268, 210, 2),
    Data::new(43, 44, 231, 246, 3),
    Data::new(43, 45, 236, 242, 2),
    Data::new(43, 46, 267, 205, 8),
    Data::new(46, 47, 247, 231, 2),
    Data::new(47, 48, 215, 263, 2),
    Data::new(47, 49, 241, 237, 2),
    Data::new(49, 50, 212, 266, 2),
    Data::new(49, 51, 239, 239, 2),
    Data::new(49, 52, 232, 239, 9),
    Data::new(49, 53, 235, 238, 7),
    Data::new(49, 54, 241, 238, 1),
    Data::new(54, 55, 277, 201, 2),
    Data::new(55, 56, 268, 210, 2),
    Data::new(56, 57, 216, 261, 3),
    Data::new(56, 58, 217, 261, 2),
    Data::new(56, 59, 254, 223, 3),
    Data::new(59, 60, 210, 266, 4),
    Data::new(59, 61, 236, 242, 2),
    Data::new(59, 62, 232, 241, 7),
    Data::new(59, 63, 246, 232, 2),
    Data::new(63, 64, 229, 247, 4),
    Data::new(63, 65, 232, 242, 6),
    Data::new(63, 66, 285, 191, 4),
    Data::new(66, 67, 250, 225, 5),
    Data::new(67, 68, 239, 238, 3),
    Data::new(68, 69, 236, 239, 5),
    Data::new(68, 70, 235, 239, 6),
    Data::new(68, 71, 255, 221, 4),
    Data::new(71, 72, 258, 218, 4),
    Data::new(72, 73, 223, 255, 2),
    Data::new(72, 74, 247, 229, 4),
    Data::new(74, 75, 230, 246, 4),
    Data::new(74, 76, 251, 224, 5),
    Data::new(76, 77, 262, 213, 5),
    Data::new(77, 78, 223, 256, 1),
    Data::new(77, 79, 188, 287, 5),
    Data::new(77, 80, 254, 226, 0),
    Data::new(80, 81, 222, 257, 1),
    Data::new(80, 82, 218, 256, 6),
    Data::new(80, 83, 252, 223, 5),
    Data::new(83, 84, 216, 260, 4),
    Data::new(83, 85, 237, 237, 6),
    Data::new(83, 86, 265, 212, 3),
    Data::new(86, 87, 241, 229, 10),
    Data::new(87, 88, 206, 270, 4),
    Data::new(87, 89, 223, 254, 3),
    Data::new(87, 90, 247, 228, 5),
    Data::new(90, 91, 242, 235, 3),
    Data::new(91, 92, 209, 264, 7),
    Data::new(91, 93, 232, 242, 6),
    Data::new(91, 94, 233, 243, 4),
    Data::new(91, 95, 243, 236, 1),
    Data::new(95, 96, 256, 215, 9),
    Data::new(96, 97, 246, 232, 2),
    Data::new(97, 98, 267, 209, 4),
    Data::new(98, 99, 265, 215, 0),
    Data::new(99, 100, 236, 241, 3),
    Data::new(99, 101, 241, 233, 6),
    Data::new(101, 102, 252, 224, 4),
    Data::new(102, 103, 190, 285, 5),
    Data::new(102, 104, 223, 254, 3),
    Data::new(102, 105, 222, 251, 7),
    Data::new(102, 106, 199, 276, 5),
    Data::new(102, 107, 238, 241, 1),
    Data::new(102, 108, 231, 246, 3),
    Data::new(102, 109, 258, 216, 6),
    Data::new(109, 110, 224, 251, 5),
    Data::new(109, 111, 203, 274, 3),
    Data::new(109, 112, 211, 264, 5),
    Data::new(109, 113, 223, 254, 3),
    Data::new(109, 114, 244, 233, 3),
    Data::new(114, 115, 198, 280, 2),
    Data::new(114, 116, 246, 229, 5),
    Data::new(116, 117, 225, 254, 1),
    Data::new(116, 118, 233, 239, 8),
    Data::new(116, 119, 233, 238, 9),
    Data::new(116, 120, 252, 226, 2),
    Data::new(120, 121, 235, 238, 7),
    Data::new(120, 122, 238, 241, 1),
    Data::new(120, 123, 193, 282, 5),
    Data::new(120, 124, 222, 255, 3),
    Data::new(120, 125, 235, 243, 2),
    Data::new(120, 126, 227, 249, 4),
    Data::new(120, 127, 199, 279, 2),
    Data::new(120, 128, 247, 227, 6),
    Data::new(128, 129, 236, 240, 4),
    Data::new(128, 130, 207, 269, 4),
];

fn main() {
    let mut elo = vec![0.0; DATA.len() + 1];
    for data in DATA {
        elo[data.b] = elo[data.a]
            + ELO_SCALE
                * ((data.wins as f64 + data.draws as f64 * 0.5)
                    / (data.wins + data.draws + data.losses) as f64
                    - 0.5);
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
