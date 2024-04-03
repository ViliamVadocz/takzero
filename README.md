# takzero

An implementation of AlphaZero for the board game Tak. See also https://github.com/ViliamVadocz/tak

# Structure

The repository contains several libraries and binaries:
- `takzero` is the main library which implements MCTS and the neural networks
- `selfplay` is used during training to generate replays and exploitation targets
- `reanalyze` computes fresh targets from old replays
- `learn` takes targets from `selfplay` and `reanalyze` to train new models
- `evaluation` pits models against each other
- `puzzle` runs the puzzle benchmark
- `analysis` includes interactive game analysis
- `graph` computes Bayesian Elo from match results (from `evaluation`) and creates a graph
- `tei` a [TEI](https://github.com/MortenLohne/racetrack#tei) implementation

I also have a `python` directory for miscellaneous Python scripts.

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
- Stable (2.2.2), CUDA 11.8, Pre-cxx11 ABI
