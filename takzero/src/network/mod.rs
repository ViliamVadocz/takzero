pub mod net3;
pub mod repr;
mod residual;

pub trait Network: Sized {
    fn new(device: tch::Device, seed: Option<i64>) -> Self;
    fn vs(&self) -> &tch::nn::VarStore;
    fn vs_mut(&mut self) -> &mut tch::nn::VarStore;

    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> (tch::Tensor, tch::Tensor);

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

    #[must_use]
    fn clone(&self, device: tch::Device) -> Self {
        let mut nn = Self::new(device, None);
        nn.vs_mut()
            .copy(self.vs())
            .expect("variables in both VarStores should have identical names");
        nn
    }
}
