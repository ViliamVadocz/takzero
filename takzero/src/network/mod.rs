pub mod net4;
pub mod net4_big;
pub mod net4_ensemble;
pub mod net4_neurips;
pub mod net5;
pub mod repr;
pub mod residual;

pub trait Network: Sized {
    fn new(device: tch::Device, seed: Option<i64>) -> Self;
    fn vs(&self) -> &tch::nn::VarStore;
    fn vs_mut(&mut self) -> &mut tch::nn::VarStore;

    #[allow(clippy::missing_errors_doc)]
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), tch::TchError> {
        self.vs().save(path)
    }

    #[allow(clippy::missing_errors_doc)]
    fn load(path: impl AsRef<std::path::Path>, device: tch::Device) -> Result<Self, tch::TchError> {
        let mut nn = Self::new(device, None);
        nn.vs_mut().load(path)?;
        Ok(nn)
    }

    #[allow(clippy::missing_errors_doc)]
    fn load_partial(
        path: impl AsRef<std::path::Path>,
        device: tch::Device,
    ) -> Result<Self, tch::TchError> {
        let mut nn = Self::new(device, None);
        nn.vs_mut().load_partial(path)?;
        Ok(nn)
    }

    #[must_use]
    fn clone(&self, device: tch::Device) -> Self {
        let mut nn = Self::new(device, None);
        nn.vs_mut()
            .copy(self.vs())
            .expect("variables in both VarStores should have identical names");
        nn
    }
}

pub trait RndNetwork: Network {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> (tch::Tensor, tch::Tensor, tch::Tensor);

    fn forward_rnd(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor;
    fn normalized_rnd(&self, xs: &tch::Tensor) -> tch::Tensor;
    fn update_rnd_normalization(&mut self, min: &tch::Tensor, max: &tch::Tensor);
}

pub trait EnsembleNetwork: Network {
    fn forward_t(
        &self,
        xs: &tch::Tensor,
        train: bool,
    ) -> (tch::Tensor, tch::Tensor, tch::Tensor, tch::Tensor);

    fn forward_ensemble(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor;
}
