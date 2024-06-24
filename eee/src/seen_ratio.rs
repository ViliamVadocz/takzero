use rand::SeedableRng;
use takzero::network::{
    net4_lcghash::{Net, MAXIMUM_VARIANCE},
    HashNetwork,
    Network,
};
use tch::Device;
use utils::reference_envs;

const RANDOM_GAMES_BATCH_SIZE: usize = 65536;
const DEVICE: Device = Device::Cuda(0);

mod utils;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let net = Net::load("_models/lcghash_mid_beta_00/model_1200000.ot", DEVICE).unwrap();

    let mut actions = vec![];
    println!("random = [");
    for i in 0..100 {
        let (_envs, xs) = reference_envs(i, &mut actions, &mut rng, RANDOM_GAMES_BATCH_SIZE);
        let ratio = net.forward_hash(&xs).mean(None) / MAXIMUM_VARIANCE;
        let ratio: f32 = ratio.try_into().unwrap();
        println!("    ({i}, {ratio}),");
    }
    println!("]");
}
