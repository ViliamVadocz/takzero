use std::{
    array,
    collections::VecDeque,
    fmt,
    fs::OpenOptions,
    io::Write,
    path::Path,
    sync::atomic::Ordering,
};

use crossbeam::channel::{Receiver, Sender};
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use rayon::prelude::*;
use takzero::{
    network::Network,
    search::{
        agent::Agent,
        env::Environment,
        node::{
            gumbel::{filter_by_unique_ascending_indices, gumbel_sequential_halving},
            Node,
        },
    },
};
use tch::Device;

use crate::{
    target::{Augment, Replay, Target},
    BetaNet,
    STEP,
};

const BATCH_SIZE: usize = 128;
const MAXIMUM_REPLAY_BUFFER_SIZE: usize = 500_000;

const SAMPLED: usize = 4;
const SIMULATIONS: u32 = 128;

const DISCOUNT_FACTOR: f32 = 0.99;

// TODO: Less n-step for older replays
// TODO: Clean up a little bit

/// Collect new state-action replays from self-play
/// and generate batches for training.
#[allow(clippy::needless_pass_by_value)]
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    rx: Receiver<Replay<E>>,
    tx: Sender<Vec<Target<E>>>,
    replay_path: &Path,
) where
    Replay<E>: Augment + fmt::Display,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = NET::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    let mut replay_queue = VecDeque::with_capacity(MAXIMUM_REPLAY_BUFFER_SIZE);

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| E::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());

    loop {
        // Receive until the replay channel is empty.
        // FIXME: If the self-play thread generates replays too fast
        // this can loop without generating any new batches
        while let Ok(replay) = rx.try_recv() {
            if replay_queue.len() == MAXIMUM_REPLAY_BUFFER_SIZE {
                replay_queue.pop_front();
            }
            replay_queue.push_back(replay);
        }
        if replay_queue.len() < BATCH_SIZE {
            std::thread::yield_now();
            continue;
        }

        // TODO: Prioritized sampling
        let replays = replay_queue.iter().choose_multiple(&mut rng, BATCH_SIZE);
        let replays = replays
            .into_iter()
            .map(|replay| replay.augment(&mut rng))
            .collect();
        let targets = reanalyze(
            &net,
            replays,
            &mut rng,
            &mut envs,
            &mut nodes,
            &mut actions,
            &mut trajectories,
        );

        tx.send(targets).unwrap();

        //  Get the latest network
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
            println!("Updating reanalyze to model beta{net_index}");

            // Save replays
            let s: String = replay_queue.iter().map(ToString::to_string).collect();
            let path = replay_path.join("replays.txt"); // same filename each time?
            rayon::spawn(move || {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)
                    .expect("replay file path should be valid and writable");
                file.write_all(s.as_bytes()).unwrap();
            });
        }

        if cfg!(test) {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reanalyze<E: Environment, NET: Network + Agent<E>>(
    net: &NET,
    replays: Vec<Replay<E>>,
    rng: &mut impl Rng,

    envs: &mut [E],
    nodes: &mut [Node<E>],
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],
) -> Vec<Target<E>> {
    debug_assert_eq!(replays.len(), BATCH_SIZE);

    envs.par_iter_mut()
        .zip(&replays)
        .for_each(|(env, replay)| *env = replay.env.clone());
    nodes
        .par_iter_mut()
        .for_each(|node| *node = Node::default());

    // Perform search at the root to get an improved policy.
    let _top_actions: Vec<<E as Environment>::Action> = gumbel_sequential_halving(
        nodes,
        envs,
        net,
        SAMPLED,
        SIMULATIONS,
        actions,
        trajectories,
        Some(rng),
    );
    // Begin constructing targets from the environment and improved policy.
    let mut targets: Vec<_> = nodes
        .par_iter()
        .zip(envs.par_iter())
        .map(|(node, env)| Target {
            env: env.clone(),
            policy: node
                .improved_policy()
                .zip(node.children.iter())
                .map(|(p, (a, _))| (a.clone(), p))
                .collect(),
            value: f32::NAN, // Value still needs to be filled.
        })
        .collect();

    // Step through the actions in the replay.
    // If we have solved a state or reach a terminal we immediately use that value.
    let (indices, batch): (Vec<_>, Vec<_>) = nodes
        .par_iter_mut()
        .zip(envs)
        .zip(&mut targets)
        .zip(actions.par_iter_mut())
        .zip(replays)
        .enumerate()
        .filter_map(|(index, ((((node, env), target), actions), replay))| {
            let mut flip = false;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            if let Some(value) = replay.actions.iter().enumerate().find_map(|(i, action)| {
                // If the node is solved, we can use that value.
                if let Some(ply) = node.evaluation.ply() {
                    return Some(
                        DISCOUNT_FACTOR.powi(i as i32 + ply as i32) * f32::from(node.evaluation),
                    );
                }
                // Take a step in the search tree and the environment.
                node.descend(action);
                env.step(action.clone());
                // If the state is terminal we can use the terminal reward.
                if let Some(terminal) = env.terminal() {
                    return Some(-DISCOUNT_FACTOR.powi(1 + i as i32) * f32::from(terminal));
                }
                // Keep track of perspective.
                flip = !flip;
                None
            }) {
                target.value = if flip { -1.0 } else { 1.0 } * value;
                None
            } else {
                env.populate_actions(actions);
                Some((index, (std::mem::take(env), std::mem::take(actions))))
            }
        })
        .unzip();
    let (batch_envs, batch_actions): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
    let output = net.policy_value(&batch_envs, &batch_actions);
    // Apply the output values onto the targets to finish them.
    let borrows: Vec<_> =
        filter_by_unique_ascending_indices(targets.iter_mut().zip(actions.iter_mut()), indices)
            .collect();
    borrows
        .into_par_iter()
        .zip(output)
        .zip(batch_actions)
        .for_each(|(((target, old_actions), (_, value)), mut actions)| {
            target.value = DISCOUNT_FACTOR.powi(i32::try_from(STEP).unwrap())
                * value
                * if STEP % 2 == 0 { 1.0 } else { -1.0 };
            // Restore actions.
            actions.clear();
            let _ = std::mem::replace(old_actions, actions);
        });

    debug_assert!(targets.iter().all(|target| target.value.is_finite()));
    targets
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::{atomic::AtomicUsize, RwLock},
    };

    use rand::{Rng, SeedableRng};
    use takzero::{
        fast_tak::{takparse::Tps, Game, Reserves},
        network::{net3::Net3, Network},
    };
    use tch::Device;

    use crate::{
        reanalyze::run,
        target::{Replay, Target},
        BetaNet,
    };

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

        let mut net = Net3::new(Device::Cpu, Some(rng.gen()));
        let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

        let (replay_tx, replay_rx) = crossbeam::channel::unbounded::<Replay<Game<3, 0>>>();
        let (batch_tx, batch_rx) = crossbeam::channel::unbounded::<Vec<Target<Game<3, 0>>>>();

        replay_tx
            .send(replay_from("x3/x3/x3 1 1", vec![
                "a3", "a1", "b1", "b3", "c1",
            ]))
            .unwrap();
        replay_tx
            .send(replay_from("2,1,1/2,x2/x3 1 3", vec![
                "a1", "b1", "b2", "b1<",
            ]))
            .unwrap();
        replay_tx
            .send(replay_from("2,1,1/2,x2/1,x2 2 3", vec!["b1", "b2", "b1<"]))
            .unwrap();
        replay_tx
            .send(replay_from("x2,1/2,x,1/2,x2 1 3", vec![
                "Sb3", "Sc1", "b3>", "c1+", "b3",
            ]))
            .unwrap();
        replay_tx
            .send(replay_from("2,1,x/2,2,x/1,x,1 1 4", vec!["c2", "b1"]))
            .unwrap();
        replay_tx
            .send(replay_from("x,1,x/112,1,x/1,2S,2 1 6", vec![
                "a3", "b1+", "a1+", "2b2<", "b2",
            ]))
            .unwrap();

        run::<_, Net3>(
            Device::cuda_if_available(),
            rng.gen(),
            &beta_net,
            replay_rx,
            batch_tx,
            &PathBuf::default(),
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
