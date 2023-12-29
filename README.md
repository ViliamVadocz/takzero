# takzero

An implementation of AlphaZero for the board game Tak. See also https://github.com/ViliamVadocz/tak

# Building

You will need the C++ Pytorch library (LibTorch).
See [tch-rs](https://github.com/LaurentMazare/tch-rs#getting-started)
for installation instructions.

## LibTorch version

It's possible you may not be able to find these versions anymore.
In that case try downloading the newest and update the `tch-rs`
version in `Cargo.toml`.

### Windows

Worked:
- Stable (2.0.1), CUDA 11.8, Release

Did **not** work:
- Preview (Nightly), CUDA 12.1, Release

### Linux

Worked:
- Stable (2.0.1), CUDA 11.8, Pre-cxx11 ABI
- Stable (2.0.1), CUDA 11.7, Pre-cxx11 ABI

Did **not** work:
- Preview (Nightly), CUDA 12.1, Pre-cxx11 ABI
- Preview (Nightly), CUDA 12.1, cxx11 ABI
- Stable (2.0.1), CUDA 11.8, cxx11 ABI

## Note

The codebase is a bit of a mess because I am in the middle of several-month-long
debugging. There will be crates which do not compile, unused dependencies,
warnings, and duplicated code. I will sort it all out once I finally find these
pesky bugs.
