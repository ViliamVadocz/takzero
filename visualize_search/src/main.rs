use std::f32::consts::PI;

use fast_tak::takparse::Tps;
// use rand::{rngs::StdRng, Rng, SeedableRng};
use svg::{
    node::element::{Circle, Line, Script},
    Document,
};
use takzero::{
    network::{
        net4_rnd::{Env, Net},
        Network,
    },
    search::{env::Environment, node::Node},
};

const VISITS: u32 = 1000;
const ARM_LENGTH: f32 = 40.0;
const CIRCLE_RADIUS: f32 = 6.0;
const COLOR: &str = "#8142f5";

fn main() {
    // let mut rng = StdRng::seed_from_u64(123);
    // let mut actions = vec![];
    // let game = Env::new_opening(&mut rng, &mut actions);
    // let net = Net::new(tch::Device::Cuda(0), Some(rng.gen()));
    let net = Net::load("directed.ot", tch::Device::Cuda(0)).unwrap();
    let env: Env = "x,1,x,1/x4/x4/2,x3 2 2".parse::<Tps>().unwrap().into();

    for beta in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0] {
        visualize_search(&net, &env, beta);
    }
}

fn visualize_search(net: &Net, env: &Env, beta: f32) {
    let mut node = Node::default();

    for _ in 0..VISITS {
        node.simulate_simple(net, env.clone(), beta);
    }

    let mut document = Document::new().set("viewBox", (-400, -400, 1000, 1000));
    // .set("style", "background:black");

    document = draw_tree(document, &node, env, 0.0, 0.0, 0.0, 2.0 * PI);
    document = document.add(Script::new(include_str!("preview.js")));

    svg::save(format!("tree_with_beta={beta}.svg"), &document).unwrap();
}

fn opacity(visits: u32) -> f32 {
    (visits as f32 / 25.0).clamp(0.0, 1.0)
}

#[allow(clippy::suboptimal_flops)]
fn draw_tree(
    mut document: Document,
    node: &Node<Env>,
    env: &Env,
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
            .set("opacity", opacity(node.visit_count))
            .set("tps", Tps::from(env.clone()).to_string()),
    );

    let angle_step = (max_angle - min_angle) / node.children.len() as f32;
    for (i, (action, child)) in node.children.iter().enumerate() {
        if child.visit_count < 1 {
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
                .set("opacity", opacity(child.visit_count))
                .set("action", action.to_string()),
        );
        let mut clone = env.clone();
        clone.step(*action);
        document = draw_tree(
            document,
            child,
            &clone,
            x2,
            y2,
            angle - PI / 4.0,
            angle + PI / 4.0,
        );
    }
    document
}
