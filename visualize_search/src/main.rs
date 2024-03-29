use std::f32::consts::PI;

use fast_tak::takparse::Tps;
use rand::{rngs::StdRng, Rng, SeedableRng};
use svg::{
    node::element::{Circle, Line},
    Document,
};
use takzero::{
    network::{
        net4_big::{Env, Net},
        Network,
    },
    search::node::Node,
};

const BETA: f32 = 0.0;
const VISITS: u32 = 1000;
const ARM_LENGTH: f32 = 30.0;
const CIRCLE_RADIUS: f32 = 2.5;
const COLOR: &str = "#8142f5";

fn main() {
    let mut rng = StdRng::seed_from_u64(123);
    // let mut actions = vec![];
    // let game = Env::new_opening(&mut rng, &mut actions);
    let game: Env = "1,x,2,112S/x,12,1,1/2,x,1,x/2,x3 1 9"
        .parse::<Tps>()
        .unwrap()
        .into();
    let net = Net::new(tch::Device::Cuda(0), Some(rng.gen()));
    // let net = Net::load("directed-random-01.ot", tch::Device::Cuda(0)).unwrap();
    let mut node = Node::default();

    for _ in 0..VISITS {
        node.simulate_simple(&net, game.clone(), BETA);
    }

    let mut document = Document::new()
        .set("viewBox", (-500, -500, 1000, 1000))
        .set("style", "background:black");

    document = draw_tree(document, &node, 0.0, 0.0, 0.0, 2.0 * PI);

    svg::save("tree.svg", &document).unwrap();
}

fn opacity(visits: u32) -> f32 {
    (visits as f32 / 25.0).clamp(0.0, 1.0)
}

#[allow(clippy::suboptimal_flops)]
fn draw_tree(
    mut document: Document,
    node: &Node<Env>,
    x: f32,
    y: f32,
    min_angle: f32,
    max_angle: f32,
) -> Document {
    document = document.add(
        Circle::new()
            .set("cx", x)
            .set("cy", y)
            .set("r", CIRCLE_RADIUS)
            .set("fill", COLOR)
            .set("opacity", opacity(node.visit_count())),
    );

    let angle_step = (max_angle - min_angle) / node.children.len() as f32;
    for (i, (_, child)) in node.children.iter().enumerate() {
        if child.visit_count() < 1 {
            continue;
        }
        let angle = min_angle + angle_step * i as f32;
        let x2 = x + ARM_LENGTH * angle.cos();
        let y2 = y + ARM_LENGTH * angle.sin();

        document = document.add(
            Line::new()
                .set("x1", x)
                .set("y1", y)
                .set("x2", x2)
                .set("y2", y2)
                .set("stroke", COLOR)
                .set("opacity", opacity(child.visit_count())),
        );
        document = draw_tree(document, child, x2, y2, angle - PI / 4.0, angle + PI / 4.0);
    }
    document
}
