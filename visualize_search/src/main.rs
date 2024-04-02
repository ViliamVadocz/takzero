use std::f32::consts::PI;

use fast_tak::takparse::Tps;
use rand::{rngs::StdRng, Rng, SeedableRng};
use svg::{
    node::element::{Circle, Line, Script},
    Document,
};
use takzero::{
    network::{
        net4_big::{Env, Net},
        Network,
    },
    search::{env::Environment, node::Node},
};

const BETA: f32 = 0.0;
const VISITS: u32 = 1000;
const ARM_LENGTH: f32 = 40.0;
const CIRCLE_RADIUS: f32 = 6.0;
const COLOR: &str = "#8142f5";

fn main() {
    let mut rng = StdRng::seed_from_u64(123);
    // let mut actions = vec![];
    // let game = Env::new_opening(&mut rng, &mut actions);
    let env: Env = "1,x,2,112S/x,12,1,1/2,x,1,x/2,x3 1 9"
        .parse::<Tps>()
        .unwrap()
        .into();
    let net = Net::new(tch::Device::Cuda(0), Some(rng.gen()));
    // let net = Net::load("directed-random-01.ot", tch::Device::Cuda(0)).unwrap();
    let mut node = Node::default();

    for _ in 0..VISITS {
        node.simulate_simple(&net, env.clone(), BETA);
    }

    let mut document = Document::new()
        .set("viewBox", (-500, -500, 1000, 1000))
        .set("style", "background:black");

    document = draw_tree(document, &node, &env, 0.0, 0.0, 0.0, 2.0 * PI);
    document = document.add(Script::new(r#"// <![CDATA[
(()=>{const t={size:"sm",theme:"discord"},e=document.firstChild,s=+e.attributes.viewBox.value.split(" ")[1];let l;for(let n of e.children)if("circle"===n.nodeName){const r=n.attributes.tps.value,i=new URL("https://tps.ptn.ninja/png");for(let e in t)i.searchParams.append(e,t[e]);i.searchParams.append("tps",r),l&&i.searchParams.append("hl",l);const u=i.href,a=document.createElementNS("http://www.w3.org/2000/svg","image"),o=n.attributes.cx.value-125;let d=+n.attributes.cy.value;d<s+200?d+=+n.attributes.r.value:d-=+n.attributes.r.value+200,a.setAttributeNS(null,"width",250),a.setAttributeNS(null,"height",200),a.setAttributeNS(null,"x",o),a.setAttributeNS(null,"y",d),a.setAttributeNS(null,"style","pointer-events: none");let p=null;n.addEventListener("mouseover",(()=>{p&&clearTimeout(p),a.attributes.href||a.setAttributeNS(null,"href",u),e.appendChild(a)})),n.addEventListener("mouseout",(()=>{p=setTimeout((()=>{e.removeChild(a),p=null}),300)}))}else"line"===n.nodeName&&(l=n.attributes.action.value,n.setAttributeNS(null,"style","pointer-events: none"))})();
// ]]>"#
    ));

    svg::save("tree.svg", &document).unwrap();
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
            .set("opacity", opacity(node.visit_count()))
            .set("tps", Tps::from(env.clone()).to_string()),
    );

    let angle_step = (max_angle - min_angle) / node.children.len() as f32;
    for (i, (action, child)) in node.children.iter().enumerate() {
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
                .set("opacity", opacity(child.visit_count()))
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
