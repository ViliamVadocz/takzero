pub mod net3;
pub mod repr;
mod residual;

pub trait Network: Default + Sized {
    fn vs(&mut self) -> &mut tch::nn::VarStore;

    fn save(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), tch::TchError> {
        self.vs().save(path)
    }

    fn load(path: impl AsRef<std::path::Path>) -> Result<Self, tch::TchError> {
        let mut nn = Self::default();
        nn.vs().load(path)?;
        Ok(nn)
    }
}
