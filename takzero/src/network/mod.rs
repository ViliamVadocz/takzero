pub mod net3;
pub mod repr;
mod residual;

pub trait Network: Default + Sized {
    fn vs(&self) -> &tch::nn::VarStore;
    fn vs_mut(&mut self) -> &mut tch::nn::VarStore;

    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), tch::TchError> {
        self.vs().save(path)
    }

    fn load(path: impl AsRef<std::path::Path>) -> Result<Self, tch::TchError> {
        let mut nn = Self::default();
        nn.vs_mut().load(path)?;
        Ok(nn)
    }
}
