use std::{array, sync::atomic::Ordering};

use crossbeam::channel::{Sender, TrySendError};
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use takzero::{
    network::Network,
    search::{
        agent::Agent,
        env::Environment,
        node::{gumbel::gumbel_sequential_halving, Node},
    },
    target::{Augment, Replay, Target},
};
use tch::Device;

use crate::{Env, Net, ReplayBuffer, SharedNet};

pub const BATCH_SIZE: usize = 128;
pub const SAMPLED: usize = 16;
pub const SIMULATIONS: u32 = 512;

// TODO: Less n-step for older replays
// TODO: Prioritized sampling

/// Collect new state-action replays from self-play
/// and generate batches for training.
#[allow(clippy::needless_pass_by_value)]
pub fn run(
    device: &Device,
    seed: u64,
    shared_net: &SharedNet,
    tx: &Sender<Vec<Target<Env>>>,
    replay_buffer: &ReplayBuffer,
) {
    log::debug!("started reanalyze thread");
    let device = *device;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(device, None);
    let mut net_index = shared_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
    let mut context = <Net as Agent<Env>>::Context::new(*shared_net.2.read().unwrap());

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| Env::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());

    loop {
        if replay_buffer.read().unwrap().len() < BATCH_SIZE {
            std::thread::yield_now();
            continue;
        }

        log::debug!("sampling batch");
        // FIXME: choose_multiple is O(n).
        let replays: Vec<_> = replay_buffer
            .read()
            .unwrap()
            .iter()
            .choose_multiple(&mut rng, BATCH_SIZE)
            .into_iter()
            .map(|replay| replay.augment(&mut rng))
            .collect();
        log::debug!("sampled replays");
        let targets = reanalyze(
            &net,
            &replays,
            &mut rng,
            &mut envs,
            &mut nodes,
            &mut actions,
            &mut trajectories,
            &mut context,
        );

        match tx.try_send(targets) {
            Ok(()) => {}
            Err(TrySendError::Full(targets)) => {
                log::warn!("target channel was full");
                tx.send(targets).unwrap();
            }
            Err(TrySendError::Disconnected(_)) => return,
        }
        log::debug!("reanalyzed and sent replays");

        //  Get the latest network
        let maybe_new_net_index = shared_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&shared_net.1.read().unwrap()).unwrap();
            log::info!("updating reanalyze to model shared_net_{net_index}");
            context = <Net as Agent<Env>>::Context::new(*shared_net.2.read().unwrap());
        }

        if cfg!(test) {
            break;
        }
    }
}

// FIXME: Refactor
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn reanalyze(
    net: &Net,
    replays: &[Replay<Env>],
    rng: &mut impl Rng,
    envs: &mut [Env],
    nodes: &mut [Node<Env>],
    actions: &mut [Vec<<Env as Environment>::Action>],
    trajectories: &mut [Vec<usize>],
    context: &mut <Net as Agent<Env>>::Context,
) -> Vec<Target<Env>> {
    debug_assert_eq!(replays.len(), BATCH_SIZE);
    let betas = [0.0; BATCH_SIZE];

    envs.iter_mut()
        .zip(replays)
        .for_each(|(env, replay)| *env = replay.env.clone());
    nodes.iter_mut().for_each(|node| *node = Node::default());

    // Perform search at the root to get an improved policy.
    log::debug!("gumbel sequential halving");
    let _top_actions: Vec<<Env as Environment>::Action> = gumbel_sequential_halving(
        nodes,
        envs,
        net,
        SAMPLED,
        SIMULATIONS,
        &betas,
        context,
        actions,
        trajectories,
        Some(rng),
    );

    // Construct targets from the improved policy.
    nodes
        .iter()
        .zip(envs.iter())
        .zip(betas)
        .map(|((node, env), beta)| Target {
            env: env.clone(),
            policy: node
                .improved_policy(
                    #[cfg(not(feature = "baseline"))]
                    beta,
                )
                .zip(node.children.iter())
                .map(|(p, (a, _))| (*a, p))
                .collect(),
            value: node.evaluation.into(),
            #[cfg(not(feature = "baseline"))]
            ube: node.variance.into(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{atomic::AtomicUsize, Arc, RwLock},
    };

    use fast_tak::{takparse::Tps, Game, Reserves};
    use rand::{Rng, SeedableRng};
    use takzero::{
        network::{net5::Net5, Network},
        target::{Replay, Target},
    };
    use tch::Device;

    use crate::{reanalyze::run, Env, SharedNet};

    fn replay_from<const N: usize, const HALF_KOMI: i8>(
        tps: &str,
        moves: Vec<&str>,
    ) -> Replay<Game<N, HALF_KOMI>>
    where
        Reserves<N>: Default,
    {
        let tps: Tps = tps.parse().unwrap();
        let moves = moves
            .into_iter()
            .map(str::parse)
            .collect::<Result<_, _>>()
            .unwrap();
        Replay {
            env: tps.into(),
            actions: moves,
        }
    }

    #[test]
    fn reanalyze_works() {
        const SEED: u64 = 1234;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net5::new(Device::Cpu, Some(rng.gen()));
        let shared_net: SharedNet = (
            AtomicUsize::new(0),
            RwLock::new(net.vs_mut()),
            RwLock::new(0.0),
        );

        let (batch_tx, batch_rx) = crossbeam::channel::unbounded::<Vec<Target<Env>>>();

        // Replays assume 3x3, change constants in main.rs when testing
        let replay_buffer = Arc::new(RwLock::new(VecDeque::from([
            replay_from("x3/x3/x3 1 1", vec!["a3", "a1", "b1", "b3", "c1"]),
            replay_from("2,1,1/2,x2/x3 1 3", vec!["a1", "b1", "b2", "b1<"]),
            replay_from("2,1,1/2,x2/1,x2 2 3", vec!["b1", "b2", "b1<"]),
            replay_from("x2,1/2,x,1/2,x2 1 3", vec![
                "Sb3", "Sc1", "b3>", "c1+", "b3",
            ]),
            replay_from("2,1,x/2,2,x/1,x,1 1 4", vec!["c2", "b1"]),
            replay_from("x,1,x/112,1,x/1,2S,2 1 6", vec![
                "a3", "b1+", "a1+", "2b2<", "b2",
            ]),
        ])));

        run(
            &Device::cuda_if_available(),
            rng.gen(),
            &shared_net,
            &batch_tx,
            &replay_buffer,
        );

        let batch = batch_rx.recv().unwrap();
        for target in batch {
            let tps: Tps = target.env.into();
            let policy: Vec<_> = target
                .policy
                .iter()
                .map(|(a, v)| format!("({a}: {v})"))
                .collect();
            println!("{tps} value: {} policy: {policy:?}", target.value);
        }
    }
}
