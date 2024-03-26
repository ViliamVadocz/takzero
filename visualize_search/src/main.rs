use std::f32::consts::PI;

// use ::rand::{rngs::StdRng, Rng, SeedableRng};
use fast_tak::takparse::Tps;
use macroquad::prelude::*;
use takzero::{
    network::{
        net4_big::{Env, Net},
        Network,
    },
    search::node::Node,
};

const BETA: f32 = 0.5;
const VISITS: u32 = 800;
const NODE_RADIUS: f32 = 4.0;
const ARM_LENGTH: f32 = 30.0;

#[macroquad::main("Visualize Search")]
async fn main() {
    // let mut rng = StdRng::seed_from_u64(123);
    // let mut actions = vec![];
    // let game = Env::new_opening(&mut rng, &mut actions);
    let game: Env = "1,x,2,112S/x,12,1,1/2,x,1,x/2,x3 1 9"
        .parse::<Tps>()
        .unwrap()
        .into();
    // let net = Net::new(tch::Device::Cuda(0), Some(rng.gen()));
    let net = Net::load("model_0650000.ot", tch::Device::Cuda(0)).unwrap();
    let mut node = Node::default();

    for _ in 0..VISITS {
        node.simulate_simple(&net, game.clone(), BETA);
    }

    loop {
        clear_background(BLACK);

        draw_tree(
            &node,
            screen_width() / 2.0,
            screen_height() / 2.0,
            0.0,
            2.0 * PI,
        );

        next_frame().await;
    }
}

fn color(visits: u32) -> Color {
    Color::from_rgba(150, 0, 200, (visits * 20).clamp(0, 200) as u8)
}

#[allow(clippy::suboptimal_flops)]
fn draw_tree(node: &Node<Env>, x: f32, y: f32, min_angle: f32, max_angle: f32) {
    draw_circle(x, y, NODE_RADIUS, color(node.visit_count()));

    let angle_step = (max_angle - min_angle) / node.children.len() as f32;
    for (i, (_, child)) in node.children.iter().enumerate() {
        let angle = min_angle + angle_step * i as f32;
        let x2 = x + ARM_LENGTH * angle.cos();
        let y2 = y + ARM_LENGTH * angle.sin();

        draw_line(x, y, x2, y2, 1.0, color(child.visit_count()));
        draw_tree(child, x2, y2, angle - PI / 4.0, angle + PI / 4.0);
    }
}
