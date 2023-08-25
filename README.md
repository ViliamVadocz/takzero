# takzero

An implementation of AlphaZero for the board game Tak. See also https://github.com/ViliamVadocz/tak

# Building

You will need the C++ Pytorch library (LibTorch).
See [tch-rs](https://github.com/LaurentMazare/tch-rs#getting-started) for installation instructions.

## LibTorch version

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

# Threading model

```
self-play  reanalyze  training  evaluation
    |          |          |         |
    |replays-> |          |         |
    |          |batch->   |         |
    |          |        train       |
    |          |batch->   |         |
    |replays-> |        train       |
    |          |batch->   |         |
    |replays-> |        train       |
    |          |batch->   |         |
    |          |        train       |
    |replays-> |        publish     |
    |          |          |         |
  update     update       |       update
    |          |          |      pit-games (science!)
    |replays-> |batch->   |      pit-games (content?)
    ..        ...        ...       ...
    |          |          |         |
    |replays-> |       publish      |
    |          |         save       |
```
