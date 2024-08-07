pub mod net4_ensemble;
pub mod net4_lcghash;
pub mod net4_rnd;
pub mod net4_simhash;
pub mod net5;
pub mod net6_simhash;
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

    fn forward_ensemble(&self, core_xs: &tch::Tensor, train: bool) -> tch::Tensor;
    fn forward_core_and_ensemble(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor;
}

pub trait HashNetwork<E: crate::search::env::Environment>: Network {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> (tch::Tensor, tch::Tensor, tch::Tensor);

    fn get_indices(&self, xs: &tch::Tensor) -> Vec<usize>;
    fn update_counts(&mut self, xs: &tch::Tensor);
    fn forward_hash(&self, xs: &tch::Tensor) -> tch::Tensor;
}
