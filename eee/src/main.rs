use std::{
    fmt::Write as FmtWrite,
    fs::OpenOptions,
    io::{BufRead, BufReader, Write as IoWrite},
    path::Path,
};

use fast_tak::takparse::{Move, Tps};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
    SeedableRng,
};
use takzero::{
    network::{
        net4_neurips::{Env, N},
        repr::{game_to_tensor, input_channels},
        residual::{ResidualBlock, SmallBlock},
    },
    search::env::Environment,
    target::{Replay, Target},
};
use tch::{
    nn::{self, Adam, ModuleT, OptimizerConfig},
    Device,
    Tensor,
};

const STEPS: usize = 25_000;
const BATCH_SIZE: usize = 128;
const DEVICE: Device = Device::Cuda(0);
const LEARNING_RATE: f64 = 1e-4;

const BATCH_5K: [&str; 128] = [
    "x,1,1,11221S/1,2,2,1/x,2,12,112S/2,21,x,2 1 17",
    "2,x,212S,2121S/x,1,2S,1/x2,2,1/x3,1 1 11",
    "1,2,12,x/2,2,1,1/1,1,12S,2/x,1,x2 1 9",
    "x,1,x2/1,1,12112,x/121,2S,x,21212S/2S,2,1,221 1 19",
    "x2,2,2/1121,1,1112S,x/x,1221112S,x,2/x,1,21,2221S 1 27",
    "1,x,121,x/x,1,12,1/x,2112S,11222,2/x,121,x2 2 20",
    "1,x3/1,x3/1,2,x2/x3,2 2 3",
    "1,1,x,2/x3,1/x4/2,x3 2 3",
    "2,1,x,2/121,2,1,x/x,2,1,x/1,x3 2 7",
    "1,2,x,2/1,x,1212112,x/12,2S,1S,1/x,2,21,1 1 15",
    "x,1,1,1/x,2,x2/x4/2,x3 2 3",
    "2,2S,1,x/2,1,2,x/2,1,2,121S/x,1,1,112 1 12",
    "1,x2,2/1,x3/1,2,x2/x4 2 3",
    "2,x,12S,x/2,12,212,x/1,1,x,121S/x,1,1,112 1 15",
    "1,1,2,1/x,12,2,x/x,1,1,x/2,x3 2 6",
    "2,x,1,x/x,2,1112S,1121/221S,x2,1/12,1212,x2 1 20",
    "x2,2S,2/x2,1,1S/2,2,1,x/x,2,1,1 1 6",
    "x,2,2,1/21,x,121S,1212S/121,x,1,1/2S,11212S,x2 2 21",
    "2,1,x,2/1,2,x2/1,x3/1,x3 2 4",
    "x3,1/x2,2121,12/x,2,x,1/2,x2,1 2 8",
    "1,1,1,2/x,12,12,1/x3,1S/x3,2 2 8",
    "1,2,x,1/2,1112S,11212,2/1,2,1S,1/1,12,2,1 1 17",
    "21,1,2,2/2,2,112S,1/112,2221S,21,1/2,1,1,1 2 20",
    "x3,2/x2,2S,1/x,1,x,1/2,x2,1 2 4",
    "1212S,1,11,221/22,x3/x,12S,1,1/1,x,12,112S 2 20",
    "2,x3/x,1,x2/1,2,2,x/x,12,1,1 1 6",
    "1,2,x,1/2,2,112112S,x/1,1,x,2/x,1,x2 1 11",
    "x3,1/x,1,2,x/x,1,2,1212/2,x,1,x 1 8",
    "x,1,2,2/2,11112S,1,1S/x,212,1,x/1,2,12,1 1 15",
    "1,1,x2/x2,2S,x/x4/2,x3 1 3",
    "1,x,1,x/x4/x4/2,x3 2 2",
    "1,2,x,121/x,1,2S,1/2,12S,x,1/2,1,x2 2 9",
    "x3,1/x2,2121,x/x,2,x,112/2,x2,1 1 9",
    "22121S,1,12,1/1,2S,1,x/212S,x2,1/1,2,12,2 1 16",
    "2,x3/x,1,1,x/1,2,2,x/x,12,1,1 2 6",
    "x,1,x,2/1,1,12,2/12,2S,1,21/x,121,x,2 1 12",
    "x,12,1,1/1,2S,1,x/x,2,1212,x/x,1,x,2 1 10",
    "x4/x,2S,x2/x,121121121S,2S,x/1,x2,1 1 12",
    "2,x,1,2/x2,2,11211112S/x,121,x,1/1,12122221S,x2 2 24",
    "x,12,1,1/1,2S,1S,x/x2,12,x/x,1,1212,2 1 12",
    "x,2,1,11221S/1,2,21,x/x,1,112S,1/2,1,2,2 2 15",
    "x,2,2,1/x2,121S,1212S/12,x,1,x/x,1121121,x2 2 18",
    "x,1,x,2/1,1,12,2/12,2S,1,21/x2,1,221 2 12",
    "1,1,1112S,x/12,2S,2,211221S/2,1,12,1/1,1,1,2S 2 21",
    "x,1S,12,112/1,2,2,1/2S,1,1,1/2,x3 2 10",
    "x3,2/x4/x2,1,x/x,2,1,1 2 3",
    "x3,1/x2,2121,x/x,2,1,112/2,x2,1 2 9",
    "1,1,2,2/1,2,1,112S/12,2S,x,112/1,1,2,1221S 1 19",
    "2,x3/x,1,1,x/1,2,2,x/x,1212,x,1 1 8",
    "2,12,1,1/1,2S,x2/1,2,x,12112/x2,21,x 1 12",
    "2,x3/2S,1,1,1/1,2,2,1/x2,12,112 1 10",
    "2,x,1,2/x,2S,2,1/x,12121S,2S,x/1,2,1,1 1 11",
    "1,x,121121S,x/x,2S,1221121,x/x,2,x,2/x,121112S,x2 2 25",
    "x,1,11,221/221212S,x3/x,12S,x,1/1,x,12,112S 2 19",
    "2,x3/x4/1,2,x2/2,1,1,1 2 4",
    "x,1,2S,2/1212,2,1,x/x,2,1,x/1,x,1,x 1 9",
    "x,11112S,1122,2/22,2,x2/1,12S,x,1/x,1,121112S,x 1 24",
    "2,x2,1/x,12,2S,112221S/1,x,12,1/21S,21,x2 1 17",
    "1,1,2S,21/x,1,x,12/x,2,x2/2,1,2,x 1 8",
    "1,1,1,2/212,x,12,1/x,12S,x,1/1,x,1212S,x 2 13",
    "2,1,x,1/2,1,2,1/1,12S,x2/x3,2 1 7",
    "2,2,1,x/2S,1,1,1/1,2,2,1/x,1S,12,112 2 11",
    "x3,2/x2,1,1S/2,2,1,x/x,2,1,1 2 5",
    "x,11112S,1122,2/22,x3/x,12S,1,1/1,1,12,112S 1 22",
    "1,2,1,x/2,2,1,1/x,1121,12S,x/1,x,2,x 2 10",
    "1,11212S,2,2/1,1112S,x,1/x,2,1,1/2,x2,12221S 1 21",
    "2,12,1,1/1,2S,1,x/1,2,2112,121S/x2,21,x 2 14",
    "1,1,2,2/1,2,1,x/12,2S,x,11212S/x,1,x,1221S 1 17",
    "2,x2,1/12,1,x,1/1,2,2,1/x3,2 1 7",
    "2,1,2,2/2S,1,121S,1/1,2,21,1/2,x2,112 2 13",
    "x4/x,12,x2/x,121121,x2/1,x2,1 2 9",
    "2,x3/x,1,x2/1,2,2,x/x,12,1,1 1 6",
    "x,2,1,11221S/1,x,21112S,x/1,12,x,1/21,x,2,2 2 17",
    "2,1212,x,1/1,x,2,x/x,1,1,x/2,x3 1 8",
    "2,x3/2S,1,1,1/1,2,2,1/x,1,12,112 2 10",
    "1,112,2S,1/121,x,2S,x/x,1,x2/2,221,x2 2 12",
    "x,1S,1S,1/21221,x2,2/x,2,2,x/1,2,x2 2 9",
    "221S,1112S,2S,1/1,221,1,2/2,2S,x,112/x,12,1,112S 2 27",
    "1,x,2S,1/21,212,x,2S/1,2S,1,112/x,12,1,1 2 14",
    "2,x3/x4/x,12,x2/1,2,1,1 1 5",
    "2,x,1,x/x,2,1112S,1121/x,21S,x,1/12,12,1,x 2 17",
    "2,2,2,1/1,1,2S,112221S/x,112112,x,1/x,221S,221,x 1 23",
    "2,x3/2S,1,1,1/1,2,2,x/x,1212,x,1 1 9",
    "1,x,212112,x/1,12S,1,12/221S,2,1,1112S/2,1,2S,2 1 25",
    "x2,21,2/2,11112S,1,1S/x,21212,x2/1,x,1212S,x 1 18",
    "1,112,2S,1/121,x,2S,x/1,2S,x2/x,22121,x2 2 14",
    "x2,12S,2/2212221S,1,1,1S/x4/21,1121,11,221 2 26",
    "x3,1/x2,12,1/x2,12,1/2,1S,1,2 2 8",
    "1,112,1,12S/121,12S,2S,1/12S,12,1,1/2,22,1,1 2 20",
    "112,x,1,x/12,1,2,1/x4/x,21S,2,2 1 10",
    "112,1,1,2/2,x,121,x/2,12S,x,1/1,x,12,112S 1 16",
    "22121S,1,1,1/112S,2S,112,1/x,22221S,2,1/11,x,2,2 2 23",
    "x2,1,2/x2,2,1/x3,1/2,x2,1 2 4",
    "221S,x,212S,2/221,11112S,11,1S/x,212S,12,x/1,x,1,x 2 22",
    "1,x3/1,x3/1,2,x2/2,1,x,2 2 4",
    "x,1,1,11221S/1,2,2,1/x2,12,112S/2,21,x,2 2 16",
    "2,1,1S,2/1,12,x2/1,12,x2/1,x3 2 8",
    "x,1112S,2,2/112,x,1,112S/x,1221,1,x/x,1,2,2221S 1 24",
    "2,1112S,2S,1/2112,x,1,x/2,2S,x,11212S/1S,12,1,1 1 23",
    "x2,2S,1/21,212,x,2S/x2,1,112/2,x,1,1 1 12",
    "x,1,1,1/x,2,x2/x4/x3,2 2 3",
    "12,1,1,2S/x2,2S,1/22221S,2,12,1112S/x2,21,2 1 19",
    "x,1,1,1/x4/x,2,x,1212/2,x,1,x 2 7",
    "1,12,x,21/x,1,1121,2/221S,2,1,x/2,1,2S,2 2 17",
    "1,2,1,x/x,2,1,x/x,1,1,x/2,x,2S,x 2 5",
    "2,1112S,2S,1/121S,221,1,2/2,2S,x,112/x,12,1,112S 1 27",
    "x,12,1,1/1,2S,x2/x4/x3,2 1 5",
    "x3,1/x3,1/x4/2,x3 2 2",
    "x2,2,2/12S,x,1112S,x/1,12211,1,21/21,1,21,2221S 2 29",
    "1,2,x,1/21,2,112112S,2/x,1,2S,1/x2,21121,x 2 16",
    "2,2,1,x/x,1,2,12/x,12S,2S,1/1,22121S,1,1 1 14",
    "112,1,1,2/121S,2,1,21/x,2,1,2/x,1,2S,2 2 13",
    "2,x,1,x/21,12S,1,112/x,2,1S,11/12,12,1,x 2 15",
    "x,2,2,1/x2,121S,1212S/12,x,1,x/x,112112,1,x 1 18",
    "11212,x3/x,1,21,2S/1,1,1,2S/x2,21122221S,2 2 19",
    "1,x,21211212S,21/1,12S,1,1/221S,2,x,1/2,1,2S,2 2 23",
    "x3,2/2,1,12S,1S/x,212,x2/1,2,1,1 1 9",
    "x,1,1,2/2,2,1,x/x,12S,x,1/1,1,12,x 2 8",
    "1,112221S,1,x/x,2S,1,1/x,2,1,2/2,1,2,21S 2 15",
    "2,x,1,2121/x,12,x,12S/1,12S,2S,x/1,22121S,1,1 2 18",
    "1,1,1,2/12,2S,x,1212S/2S,x,112221S,x/21,x,221S,2 1 23",
    "x,1,2,2/x,2,12S,x/1,2,1,21/11221S,1,1,2 2 14",
    "2,1,x,2/x4/1,x3/1,x3 2 3",
    "1,12,1,x/1,1,1S,221212S/1,2S,12,x/2S,2,x,1 2 14",
    "x3,1/x,1,2,x/x,1,2,1212/2,x,1,x 1 8",
    "x3,1/2,x,2S,112/1,2,1,1221S/2,1S,1,x 1 13",
    "x2,2S,2/x,1,1,1S/2,2,1,x/x,2,1,1 2 6",
    "1,1,x2/12,2S,1,1/1,2,1,2/1,x,2,2 2 8",
];

const BATCH_20K: [&str; 128] = [
    "1,2,1,2/x,1,1,2/x2,112,x/x,2,x,21 1 9",
    "1,x,1,2/2,2S,x,1/1,1,12S,2/x2,2,1 1 8",
    "x,1,21,x/x,12S,1,1/1,1,2S,12/2,2,1,2 2 10",
    "1,1,12,x/2,2S,x,1/1,1,12S,2/x2,2,1 1 9",
    "1,x,1,2/2,12S,x,1/x,1,2S,x/x,1,2,1 2 7",
    "x2,1,1/2,2S,1,2/1,x,21,112/1,x,2,2 2 10",
    "2,1,1,1/2112S,x,1,2/x,1121S,2S,1/2,x,1,1212S 2 16",
    "1,1,1,2/2,2S,x,1/1,1,12S,2/x2,2,1 2 8",
    "2,1,2,1/2S,x,121,2/12121S,2S,2S,1/x,1,x,1 2 13",
    "x2,12,1/1,x,12,x/2,1S,x,12/2,1,1,1 2 10",
    "1,1,1112S,x/x,221,x2/2,x,112S,12/11112S,2,1,x 1 20",
    "x,1,x,2/12,112,x,1/1,x2,2/1,x3 1 8",
    "2,x2,1/2,12S,1,1/1,1,2S,2/2,12,1,1 1 10",
    "2,1,2,x/x4/x2,1,x/x,1,2,1 2 4",
    "x,1,12,x/212S,x,2,1/1,112S,x,2/x2,21,1 1 12",
    "1,x2,2/2,12S,x2/1,x,12S,1/1,1,2,1 2 8",
    "1,1,2,2/12,112121S,2S,1/1,2S,x,2/1,x3 1 13",
    "2,1,112,1/212S,x,12,2/1,x,2,112/x,1,x,2 1 16",
    "x,1,x,1/2,2,x,2/12,1,x,1/x4 1 6",
    "x,1,2,1/1,1,12S,2/2,2S,x,1/1,1,12,x 2 9",
    "2,1,1,2/2211212S,1S,12,2/x2,12S,1/1,121,x,1 2 17",
    "1,21,x,1/2,112S,2,12/1,112S,112S,1/x,1,2,x 2 16",
    "1,x2,2/2,12S,x,1/x,1,2S,x/x,1,2,1 1 7",
    "1,2,1,2/x,1,1,2/x2,1,x/x,2,12,21 2 8",
    "1S,1,x,1/2,2,1,x/12,x,1,12/2,x2,1 2 8",
    "1,2,1,x/x4/x4/x3,2 1 3",
    "x2,1212121S,x/21,1,2S,1/11112S,2,12S,2/2,1,21,2 1 21",
    "1,x2,2/x4/1,x3/x4 2 2",
    "x,21,1,21/1,1112S,2,1/2,1,x2/2,2,112S,x 2 15",
    "1,2,2S,1/1,1,2S,1/x,1,1S,1/12,212,12112S,2 2 16",
    "x,21112S,1,1/1,x,112S,x/121,2,x2/x,112S,212,2 1 19",
    "1,x,1212,1/2,2S,2,1/1,1,12S,2/x,1,2,1 1 12",
    "x2,121,1/212S,1,2,1/1,112S,x,2/x,1,2,112S 2 15",
    "x,1,2,1/x,1,12S,1/x2,1,12/2,2S,1,x 2 8",
    "21,1,1,x/x,12S,2,1/2,1,2S,2/x,1,2,1 2 9",
    "2,1,12,2/2,2121S,2S,1/x,211212S,1,1/21,x,1,2 2 19",
    "x,1,12,1/212S,x,2,1/1,112S,x,2/x2,21,1 2 12",
    "x2,12,12/2S,1,1,1/x3,1/2,2,x,1 2 8",
    "x3,2/x4/x4/1,2,1,x 1 3",
    "2,x3/x3,2S/x4/1,1,1,2 1 4",
    "x,1,2,1/x2,1,x/x4/2,x3 2 3",
    "2,x2,1/x3,1/2,x,1,2/x3,1 2 4",
    "x,2,x2/1,2,21,x/2,1,2,21/1,x,12,x 1 9",
    "2,1221S,x2/2,2,2,1/x,1,1,2/1,12,x,1 1 11",
    "1,12,1,x/1,2,2S,21/2,x,112S,1/1,21,2S,x 2 12",
    "x2,12,1/x2,12,x/x,1S,x,12/2,1,x,1 1 9",
    "1,2,2S,1/1,1,2S,1/x2,1S,1/12,212,12112S,2 1 16",
    "2,x3/2,21,2,1/x,12221S,12,21/1,1,2,x 1 14",
    "1,12,2,1/2,1,1,2/x2,1221S,x/2,121S,x,2 1 13",
    "x,21,12,21/1,1112S,x,1/2,1,x2/2,2,112S,x 1 16",
    "x,2,x,21/1,1,12S,1/1S,2S,2S,1/1212212S,x,1,2S 2 16",
    "2,1,2,1/1,2,1,12/2,x,1,x/x3,1 2 7",
    "x,1,12,1/2,1,1221S,x/21,2,2S,121/2,12,1,x 2 16",
    "x4/2,1,2S,2121/1,x,2,x/1,x2,2 1 8",
    "221,11112S,x,21/2,1,2S,1/2121,x,2,2/x,2,112S,1 1 22",
    "x,1,2,1/x,1,2S,x/2,12S,x,1/1,2,1,2 1 8",
    "2,12,21,x/x,2S,1112S,21/122121S,x3/112S,1,x,1221 2 22",
    "1,2,x2/2S,1,1,1/x2,2S,2/2,x2,1 1 6",
    "1,2S,12,1/1,112,12S,2/21S,1,1,2/2,x,1,221S 1 16",
    "x2,2,1/1,1,12S,x/2,2S,x2/1,x,1,2 1 7",
    "1,2,1,2/x,1,x,2/x2,1121121,x/x,2,x,21 2 12",
    "x,1,12,x/21,2S,2,1/1,112S,x,2/x,1,2,1 1 11",
    "1,x2,2/2,2S,x2/1,1,1,x/x4 2 4",
    "2,1,21,x/x,12S,1,x/1,1,2S,1212/2,2,1,x 1 12",
    "2,1212,21S,x/1,112S,2,112/2,1,1,x/2S,1,1,1 2 17",
    "2,211211212S,1,2/1,2,221S,1/12S,2S,2,21/1,2,1,1 1 22",
    "x,21,12121,x/212S,x2,1/1,112S,x,2/x,1,2,112S 2 17",
    "2,x,2S,1/x2,1,x/x,2S,1,x/1,2,1,x 2 5",
    "2,x3/2,21,x,1/x,12221S,12,21/1,1,2,x 2 13",
    "1,2,x2/x,12S,1,1/x2,2S,2/2,1,x,1 1 7",
    "2,x2,1/2,2,2,1/1,12,1,1S/x4 1 7",
    "x,1,2,2/x,12S,12,1/21112,x,1,21/1112S,2,221,21 1 22",
    "221S,1,x,2/2,x2,21S/2,1,112,x/1,12,1,1 2 13",
    "1,2,1,2/x,1,1,2/21,112S,112,1/x,2,x,21 2 15",
    "x,1,1S,x/x,2S,2S,21/2S,1S,2,2/121212121S,x2,2 1 17",
    "x,212,1,x/x3,1/x4/2,2,x,1 1 6",
    "1,1212S,x,1/1,12S,1,1/2,x,12S,1/21,112,2,2 1 16",
    "221S,1,2S,2/2,x2,21S/2,2S,11221121,x/1,121,x,1 2 19",
    "2,x,122121,x/x,2,21,x/x,1,12,x/1,1,1,x 2 13",
    "2,1,1,12S/2,2,2S,21/1,x,12S,1/1,2,1,x 1 11",
    "221,1,2S,1/2,2,1,1/1221S,2S,12S,12/2,1,2,1 1 17",
    "1,x,1212,1/2,2S,2,1/1,1,12S,2/x,1,2,1 1 12",
    "2,x3/x3,1/x4/x3,1 2 2",
    "x,1,2,x/21,1,2S,x/12,12S,1,2/1,12,1,1 2 11",
    "1,2,1,x/x,1,x,2/x4/x3,2 1 4",
    "x4/1,1,x,2/x,12,212,x/2,x,1,1 1 8",
    "2,x2,1/x2,1,1/x3,2/x4 2 3",
    "x,1,2,1/21,2S,12S,x/12,1,1,2/x,1,2,1 2 10",
    "x2,21,1/1,112S,x,2/212S,x,2,1/x,1,12,x 1 12",
    "2,1,x,1/x2,2S,2/2S,1,1,1/x4 2 5",
    "2,12,21S,x/x,2,2,1/x,1,1,2/1,12,x,1 1 10",
    "x3,1/x3,2/x,1,2,1/2,x3 1 4",
    "x,2,x,21/1,x,12S,1/x,2S,2S,1/121221,x,1,2S 2 14",
    "1,1,1,2/2,x3/1,x2,1212/x3,2 1 8",
    "1,x3/1,x3/2,2,x2/1,x2,2 1 4",
    "x4/x2,1,1/x3,2/2,x2,1 2 3",
    "1,1,1212121S,1/2,2S,2S,x/1,1112S,x,1/2,1,2,1212 2 21",
    "21,x,2,x/x,221121,2S,x/2,x,1,1/1,x,2S,1 2 12",
    "1,2,x2/2,12S,1,1/1,x,2S,2/2,1,x,1 1 8",
    "2,2S,x2/x,1,x2/x,1,2S,x/x,1,2,1 1 5",
    "1,1212S,x,1/2,2S,111221S,1/1,2,1,1/1,2,212S,2 2 18",
    "1,1,2,2/x,2S,x,1/x,1,x2/x3,2 1 5",
    "2,x2,1/2,2,2,x/1,1,x,1S/x4 1 5",
    "2,1,1,1/1,12S,x,12/2,x,2S,1/x,1,2,1 2 9",
    "x,1,2,x/21,1,2S,2/12,12S,1,2/1,12,1,1 1 12",
    "1,2,1,2/x,1,1,2/21,112S,112,1/x,2,2,21 1 16",
    "x,1,2,2/121,12S,1,x/12S,112,1,1/1,2,x,2 1 14",
    "2,1,21,x/2,1,112S,1/1221S,1,x,12/112,2S,121,x 1 21",
    "x3,1/x4/x,1S,1212,12/2,1,x,1 2 8",
    "x,1,12,1/21,2S,2,1/1,112S,x,2/x,1,212S,1 2 13",
    "1,1,1,2/2,x,1,1S/x,12112,x2/x2,2,2 1 11",
    "x2,212S,1/1,112S,x,2/212S,1,2,1/x,1,12,1 1 14",
    "x,1,x,2/12,11212,1S,1/1,2S,x,2/1,x3 1 11",
    "2,x2,1/2,x,2,1/1,12121,x,1S/2,x3 2 9",
    "1,1,12,1/2,1,1221S,x/21,2,x,1212S/2,12,1,x 2 17",
    "x,1,2,2/121,12S,1,x/12S,1,1,1/1,2,2,2 1 12",
    "1,1,1,21S/2,x,1,2S/x,12,1112,x/2,x,21S,2 2 14",
    "x3,1/x3,2/x3,1/2,x3 1 3",
    "2,2,1,x/x,1,2S,1/x,112S,x,2/1,2,1,1 1 9",
    "x2,2,1/1,1,1,2S/2,2S,x2/1,x,1,2 2 6",
    "2,x3/x,1,x,2S/x4/1,1,1,2 2 4",
    "x2,212S,x/1,112S,x,21/2,1,2,1/112S,1,12,1 2 15",
    "12,x,2S,2121/1,1,12S,x/1,2S,2S,1/121,221S,112S,2S 2 21",
    "1,2,1,1/212S,x,2,12/x,112S,1,1/x,1,212S,x 1 14",
    "1,2,1,x/x3,1/x4/2,2,x2 1 4",
    "1,1,2,112S/1,112S,121,x/x3,121/2,212,112S,x 2 19",
    "2,1,x,12112/1,1,112S,x/22112S,x2,1/2,1,x,1 2 17",
    "x,2,1,x/x,1,12S,21/x,1,12S,121/2,2,1,x 2 11",
];

fn get_replays(path: impl AsRef<Path>) -> impl Iterator<Item = Replay<Env>> {
    BufReader::new(OpenOptions::new().read(true).open(path).unwrap())
        .lines()
        .filter_map(|line| line.ok()?.parse::<Replay<Env>>().ok())
}

#[allow(unused)]
fn get_targets(path: impl AsRef<Path>) -> impl Iterator<Item = Target<Env>> {
    BufReader::new(OpenOptions::new().read(true).open(path).unwrap())
        .lines()
        .filter_map(|line| line.ok()?.parse::<Target<Env>>().ok())
}

fn random_env(ply: usize, actions: &mut Vec<Move>, rng: &mut impl Rng) -> Env {
    let mut env = Env::default();
    for _ in 0..ply {
        env.populate_actions(actions);
        let Some(action) = actions.drain(..).choose(rng) else {
            break;
        };
        env.step(action);
    }
    env
}

fn reference_envs(ply: usize, actions: &mut Vec<Move>, rng: &mut impl Rng) -> (Vec<Env>, Tensor) {
    let games: Vec<_> = (0..BATCH_SIZE)
        .map(|_| random_env(ply, actions, rng))
        .collect();
    let tensor = Tensor::cat(
        &games
            .iter()
            .map(|g| game_to_tensor(g, Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);
    (games, tensor)
}

fn rnd(path: &nn::Path) -> nn::SequentialT {
    const RES_BLOCKS: u32 = 8;
    const FILTERS: i64 = 64;
    let mut net = nn::seq_t()
        .add(nn::conv2d(
            path / "input_conv2d",
            input_channels::<N>() as i64,
            FILTERS,
            3,
            nn::ConvConfig {
                stride: 1,
                padding: 1,
                bias: false,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(
            path / "batch_norm",
            FILTERS,
            nn::BatchNormConfig::default(),
        ))
        .add_fn(Tensor::relu);
    for n in 0..RES_BLOCKS {
        net = net.add(ResidualBlock::new(
            &(path / format!("res_block_{n}")),
            FILTERS,
            FILTERS,
        ));
    }
    net.add(SmallBlock::new(&(path / "last_small_block"), FILTERS, 32))
        .add_fn(|x| x.flatten(1, 3))
}

fn main() {
    let seed: u64 = 12345;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let vs = nn::VarStore::new(DEVICE);
    let root = vs.root();
    let target = rnd(&(&root / "target"));
    let predictor = rnd(&(&root / "predictor"));

    let mut opt = Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut replays = get_replays("4x4_old_directed_01_replays.txt");
    let mut buffer = Vec::with_capacity(2048);

    // let mut actions = Vec::new();
    // let (_, early_tensor) = reference_envs(4, &mut actions, &mut rng);
    // let (_, late_tensor) = reference_envs(120, &mut actions, &mut rng);
    let early_tensor = Tensor::concat(
        &BATCH_5K
            .into_iter()
            .map(|s| game_to_tensor::<4, 4>(&s.parse::<Tps>().unwrap().into(), Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);
    let late_tensor = Tensor::concat(
        &BATCH_20K
            .into_iter()
            .map(|s| game_to_tensor::<4, 4>(&s.parse::<Tps>().unwrap().into(), Device::Cpu))
            .collect::<Vec<_>>(),
        0,
    )
    .to(DEVICE);

    let mut losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut early_losses: Vec<f64> = Vec::with_capacity(STEPS);
    let mut late_losses: Vec<f64> = Vec::with_capacity(STEPS);

    let mut running_mean = early_tensor.zeros_like();
    let mut running_sum_squares = running_mean.ones_like();
    for step in 0..STEPS {
        if step % 100 == 0 {
            println!("step: {step: >8}");
        }

        // Add replays to buffer until we have enough.
        while buffer.len() < 1024 {
            let replay = replays.next().unwrap();
            let mut env = replay.env;
            buffer.extend(replay.actions.into_iter().map(|a| {
                env.step(a);
                env.clone()
            }));
        }

        // Sample a batch.
        buffer.shuffle(&mut rng);
        let batch = buffer.split_off(buffer.len() - BATCH_SIZE);
        let tensor = Tensor::concat(
            &batch
                .into_iter()
                .map(|env| game_to_tensor(&env, Device::Cpu))
                .collect::<Vec<_>>(),
            0,
        )
        .to(DEVICE);

        // Update normalization statistics.
        let new_running_mean = &running_mean + (&tensor - &running_mean) / (step + 1) as i64;
        running_sum_squares += (&tensor - &running_mean) * (&tensor - &new_running_mean);
        running_mean = new_running_mean;
        let running_variance = &running_sum_squares / (step + 1) as i64; // Normalize.

        // Compute loss for early batch.
        let input = ((&early_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        early_losses.push(loss.try_into().unwrap());

        // Compute loss for late batch.
        let input = ((&late_tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, false).detach();
        let loss = (target_out - predictor_out).square().mean(None);
        late_losses.push(loss.try_into().unwrap());

        // Do a training step.
        let input = ((tensor - &running_mean) / running_variance.sqrt()).clip(-5, 5);
        let target_out = target.forward_t(&input, false).detach();
        let predictor_out = predictor.forward_t(&input, true);
        let loss = (target_out - predictor_out).square().mean(None);
        opt.backward_step(&loss);

        // Save the normalized loss.
        losses.push(loss.try_into().unwrap());
    }

    println!("{running_mean}");
    println!("{running_mean:?}");
    println!("{running_sum_squares}");
    println!("{running_sum_squares:?}");

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("rnd_data.csv")
        .unwrap();
    let content = losses
        .into_iter()
        .zip(early_losses)
        .zip(late_losses)
        .enumerate()
        .fold(
            "step,loss,early,late\n".to_string(),
            |mut s, (step, ((loss, early), late))| {
                writeln!(&mut s, "{step},{loss},{early},{late}").unwrap();
                s
            },
        );
    file.write_all(content.as_bytes()).unwrap();

    println!("Done.");
}
