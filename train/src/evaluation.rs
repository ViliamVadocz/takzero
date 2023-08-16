use rand::SeedableRng;

use crate::BetaNet;

/// Evaluate checkpoints of the network to make sure that it is improving.
pub fn run(seed: u64, beta_net: &BetaNet) -> ! {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    loop {
        todo!()
    }
}
