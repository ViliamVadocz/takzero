use std::ops::Add;

use tch::{nn, Tensor};

// <https://medium.com/@bentou.pub/
// alphazero-from-scratch-in-pytorch-for-the-game-of-chain-reaction-part-3-c3fbf0d6f986>

#[derive(Debug)]
pub struct SmallBlock {
    model: nn::SequentialT,
}

impl SmallBlock {
    #[must_use]
    pub fn new(vs: &nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let model = nn::seq_t()
            .add(nn::conv2d(
                vs / "conv2d",
                in_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    bias: false,
                    ..Default::default()
                },
            ))
            .add(nn::batch_norm2d(
                vs / "batch_norm",
                out_channels,
                nn::BatchNormConfig::default(),
            ));
        Self { model }
    }
}

impl nn::ModuleT for SmallBlock {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        self.model.forward_t(xs, train)
    }
}

#[derive(Debug)]
pub struct ResidualBlock {
    model: nn::SequentialT,
}

impl ResidualBlock {
    pub fn new(vs: &nn::Path, in_channels: i64, mid_channels: i64) -> Self {
        let model = nn::seq_t()
            .add(SmallBlock::new(vs, in_channels, mid_channels))
            .add_fn(Tensor::relu)
            .add(SmallBlock::new(vs, mid_channels, in_channels));
        Self { model }
    }
}

impl nn::ModuleT for ResidualBlock {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        self.model.forward_t(xs, train).add(xs).relu()
    }
}
