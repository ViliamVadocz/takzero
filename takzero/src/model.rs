use tch::{nn, Tensor};

use crate::repr::input_size;

const fn possible_moves<const N: usize>() -> usize {
    match N {
        3 => 126,
        4 => 480,
        5 => 1_575,
        6 => 4_572,
        7 => 12_495,
        8 => 32_704,
        9 => 82_863,
        10 => 204_700,
        11 => 495_495,
        12 => 1_179_504,
        13 => 2_768_727,
        14 => 6_422_332,
        15 => 14_745_375,
        16 => 33_554_176,
        _ => panic!("Unknown possible move count"),
    }
}

pub fn net(vs: &nn::Path) -> impl nn::Module {
    #![allow(clippy::cast_possible_wrap)]

    const HIDDEN_LAYER_DIMS: i64 = 128;

    nn::seq()
        .add(nn::linear(
            vs,
            input_size::<3>() as i64,
            HIDDEN_LAYER_DIMS,
            nn::LinearConfig::default(),
        ))
        .add_fn(Tensor::relu)
        .add(nn::linear(
            vs,
            HIDDEN_LAYER_DIMS,
            possible_moves::<3>() as i64,
            nn::LinearConfig::default(),
        ))
        .add_fn(|xs| xs.softmax(0, None))
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use tch::{
        nn::{self, Module},
        Device,
    };

    use super::net;
    use crate::repr::game_to_tensor;

    #[test]
    fn make_net() {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let net = net(&vs.root());

        let game: Game<3, 0> = Game::default();
        let tensor = game_to_tensor(&game, Device::cuda_if_available());
        let _out = net.forward(&tensor);
    }
}
