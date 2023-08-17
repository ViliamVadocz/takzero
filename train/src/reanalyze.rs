use std::{array, collections::VecDeque, sync::atomic::Ordering};

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
    target::{Replay, Target},
    BetaNet,
    STEP,
};

const BATCH_SIZE: usize = 256;
const MAXIMUM_REPLAY_BUFFER_SIZE: usize = 1_000_000;

const SAMPLED: usize = 32;
const SIMULATIONS: u32 = 1024;

const DISCOUNT_FACTOR: f32 = 0.99;

// TODO: Less n-step for older replays
// TODO: Save replays

/// Collect new state-action replays from self-play
/// and generate batches for training.
pub fn run<E: Environment, NET: Network + Agent<E>>(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    rx: Receiver<Replay<E>>,
    tx: Sender<Vec<Target<E>>>,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = NET::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    let mut replay_queue = VecDeque::with_capacity(MAXIMUM_REPLAY_BUFFER_SIZE);

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| E::default());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());

    loop {
        // Receive until the replay channel is empty.
        // FIXME: If the self-play thread generates replays too fast
        // this can loop without generating any new batches
        while let Ok(replay) = rx.try_recv() {
            if replay_queue.len() + 1 >= MAXIMUM_REPLAY_BUFFER_SIZE {
                replay_queue.pop_front();
            }
            replay_queue.push_back(replay);
        }
        if replay_queue.len() < BATCH_SIZE {
            continue;
        }

        // TODO: Prioritized sampling
        let replays = replay_queue.iter().choose_multiple(&mut rng, BATCH_SIZE);
        let targets = reanalyze(
            &net,
            replays,
            &mut rng,
            &mut envs,
            &mut actions,
            &mut trajectories,
        );

        tx.send(targets).unwrap();

        //  Get the latest network
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index >= net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
        }

        if cfg!(test) {
            break;
        }
    }
}

fn reanalyze<E: Environment, NET: Network + Agent<E>>(
    net: &NET,
    replays: Vec<&Replay<E>>,
    rng: &mut impl Rng,
    envs: &mut [E],
    actions: &mut [Vec<E::Action>],
    trajectories: &mut [Vec<usize>],
) -> Vec<Target<E>> {
    debug_assert_eq!(replays.len(), BATCH_SIZE);

    envs.par_iter_mut()
        .zip(&replays)
        .for_each(|(env, replay)| *env = replay.env.clone());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());

    // Perform search at the root to get an improved policy.
    let _top_actions: Vec<<E as Environment>::Action> = gumbel_sequential_halving(
        &mut nodes,
        envs,
        net,
        rng,
        SAMPLED,
        SIMULATIONS,
        actions,
        trajectories,
    );
    // Begin constructing targets from the environment and improved policy.
    let mut targets: Vec<_> = nodes
        .par_iter()
        .zip(envs.par_iter_mut())
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
            if let Some(value) = replay.actions.iter().enumerate().find_map(|(i, action)| {
                // If the node is solved, we can use that value.
                if let Some(ply) = node.evaluation.ply() {
                    return Some(
                        DISCOUNT_FACTOR.powi(ply as i32) * Into::<f32>::into(node.evaluation),
                    );
                }
                // Take a step in the search tree and the environment.
                node.descend(action);
                env.step(action.clone());
                // If the state is terminal we can use the terminal reward.
                if let Some(terminal) = env.terminal() {
                    return Some(-DISCOUNT_FACTOR.powi(i as i32) * Into::<f32>::into(terminal));
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
            target.value =
                DISCOUNT_FACTOR.powi(STEP as i32) * value * if STEP % 2 == 0 { 1.0 } else { -1.0 };
            // Restore actions.
            actions.clear();
            let _ = std::mem::replace(old_actions, actions);
        });

    debug_assert!(targets.iter().all(|target| target.value.is_normal()));
    targets
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicUsize, RwLock};

    use arrayvec::ArrayVec;
    use fast_tak::{
        takparse::{Move, Tps},
        Game,
    };
    use rand::{Rng, SeedableRng};
    use takzero::network::{net3::Net3, Network};
    use tch::Device;

    use crate::{
        reanalyze::run,
        target::{Replay, Target},
        BetaNet,
        STEP,
    };

    #[test]
    fn reanalyze_works() {
        const SEED: u64 = 1234;

        fn make_array(actions: [&str; STEP]) -> ArrayVec<Move, STEP> {
            actions.map(|s| s.parse().unwrap()).into()
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net3::new(Device::Cpu, Some(rng.gen()));
        let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

        let (replay_tx, replay_rx) = crossbeam::channel::unbounded::<Replay<Game<3, 0>>>();
        let (batch_tx, batch_rx) = crossbeam::channel::unbounded::<Vec<Target<Game<3, 0>>>>();

        replay_tx
            .send(Replay {
                env: Game::default(),
                actions: make_array(["a1", "a2", "a3", "b1", "b2"]),
            })
            .unwrap();
        replay_tx
            .send(Replay {
                env: Game::from_ptn_moves(&["a3", "c1", "c2", "a2"]),
                actions: make_array(["b2", "b1", "Sa1", "b1+", "b1"]),
            })
            .unwrap();
        replay_tx
            .send(Replay {
                env: Game::default(),
                actions: make_array(["a3", "c1", "a1", "a2", "b1"]),
            })
            .unwrap();

        run::<_, Net3>(
            Device::cuda_if_available(),
            rng.gen(),
            &beta_net,
            replay_rx,
            batch_tx,
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
