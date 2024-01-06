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

You may also need to set `LIBTORCH_BYPASS_VERSION_CHECK` to `1`.

If you find some version works, please let me know so I can add it here.

### Windows

Worked:
- Stable (2.1.2), CUDA 11.8, Release
- Stable (2.1.2), CUDA 11.8, Debug

Did **not** work:
- TODO

### Linux

Worked:
- Stable (2.1.2), CUDA 11.8, Pre-cxx11 ABI

Did **not** work:
- TODO
