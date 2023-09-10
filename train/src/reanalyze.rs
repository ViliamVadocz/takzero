use std::{array, sync::atomic::Ordering};

use crossbeam::channel::{Sender, TrySendError};
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use takzero::{
    network::{repr::game_to_tensor, Network},
    search::{
        agent::Agent,
        env::Environment,
        node::{
            gumbel::{filter_by_unique_ascending_indices, gumbel_sequential_halving},
            Node,
        },
        DISCOUNT_FACTOR,
    },
};
use tch::{Device, Tensor};

use crate::{
    target::{Augment, Replay, Target},
    BetaNet,
    Env,
    Net,
    ReplayBuffer,
    STEP,
};

const BATCH_SIZE: usize = 512;

const SAMPLED: usize = 32;
const SIMULATIONS: u32 = 128;

// TODO: Less n-step for older replays
// TODO: Prioritized sampling

/// Collect new state-action replays from self-play
/// and generate batches for training.
#[allow(clippy::needless_pass_by_value)]
pub fn run(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    tx: &Sender<Vec<Target<Env>>>,
    replay_buffer: &ReplayBuffer,
) {
    log::debug!("started reanalyze thread");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut net = Net::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

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
            device,
            &mut envs,
            &mut nodes,
            &mut actions,
            &mut trajectories,
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
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
            log::info!("updating reanalyze to model beta{net_index}");
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
    device: Device,

    envs: &mut [Env],
    nodes: &mut [Node<Env>],
    actions: &mut [Vec<<Env as Environment>::Action>],
    trajectories: &mut [Vec<usize>],
) -> Vec<Target<Env>> {
    debug_assert_eq!(replays.len(), BATCH_SIZE);
    let beta = 0.0;

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
        beta,
        actions,
        trajectories,
        Some(rng),
    );
    // Begin constructing targets from the environment and improved policy.
    let mut targets: Vec<_> = nodes
        .iter()
        .zip(envs.iter())
        .map(|(node, env)| Target {
            env: env.clone(),
            policy: node
                .improved_policy(beta)
                .zip(node.children.iter())
                .map(|(p, (a, _))| (*a, p))
                .collect(),
            value: f32::NAN, // Value still needs to be filled.
            ube: 0.0,        // Will be updated.
        })
        .collect();

    for step in 0..STEP {
        log::debug!("reanalyze step");
        let sign = if step % 2 == 0 { 1.0 } else { -1.0 };
        let raw_rnd: Vec<f32> = net
            .forward_rnd(
                &Tensor::cat(
                    &envs
                        .iter()
                        .map(|env| game_to_tensor(env, device))
                        .collect::<Vec<_>>(),
                    0,
                ),
                false,
            )
            .view([-1])
            .try_into()
            .unwrap();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        nodes
            .iter_mut()
            .zip(envs.iter_mut())
            .zip(replays)
            .zip(raw_rnd)
            .zip(&mut targets)
            .filter(|(_, target)| target.value.is_nan())
            .for_each(|((((node, env), replay), raw_rnd), target)| {
                // Accumulate discounted RND.
                target.ube += DISCOUNT_FACTOR.powi(2 * step as i32) * raw_rnd;

                // If the node is solved, we can use that value.
                if let Some(ply) = node.evaluation.ply() {
                    target.value = sign
                        * DISCOUNT_FACTOR.powi(step as i32 + ply as i32)
                        * f32::from(node.evaluation);
                    return;
                }
                // Take a step in the search tree and the environment.
                let action = replay.actions[step];
                node.descend(&action);
                env.step(action);

                // If the state is terminal, use the terminal reward.
                if let Some(terminal) = env.terminal() {
                    target.value =
                        -sign * DISCOUNT_FACTOR.powi(1 + step as i32) * f32::from(terminal);
                }
            });
    }

    // Collect environments which still need to be evaluated.
    let (indices, batch): (Vec<_>, Vec<_>) = envs
        .iter()
        .zip(actions.iter_mut())
        .enumerate()
        .zip(&targets)
        .filter(|(_, target)| target.value.is_nan())
        .map(|((index, (env, actions)), _)| (index, (env.clone(), std::mem::take(actions))))
        .unzip();
    let (batch_envs, batch_actions): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
    let output = net.policy_value_uncertainty(&batch_envs, &batch_actions);

    // Apply the output values and uncertainties onto the targets to finish them.
    filter_by_unique_ascending_indices(targets.iter_mut().zip(actions.iter_mut()), indices)
        .zip(output)
        .zip(batch_actions)
        .for_each(
            |(((target, old_actions), (_, value, uncertainty)), mut actions)| {
                target.value = DISCOUNT_FACTOR.powi(i32::try_from(STEP).unwrap())
                    * value
                    * if STEP % 2 == 0 { 1.0 } else { -1.0 };
                target.ube += DISCOUNT_FACTOR.powi(2 * i32::try_from(STEP).unwrap()) * uncertainty;
                // Restore actions.
                actions.clear();
                let _ = std::mem::replace(old_actions, actions);
            },
        );

    // Clip ube target between 0.0 and 1.0
    for target in &mut targets {
        target.ube = target.ube.clamp(0.0, 1.0);
    }

    assert!(targets.iter().all(|target| target.value.is_finite()));
    targets
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{atomic::AtomicUsize, Arc, RwLock},
    };

    use fast_tak::{takparse::Tps, Game, Reserves};
    use rand::{Rng, SeedableRng};
    use takzero::network::{net5::Net5, Network};
    use tch::Device;

    use crate::{
        reanalyze::run,
        target::{Replay, Target},
        BetaNet,
        Env,
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

        let mut net = Net5::new(Device::Cpu, Some(rng.gen()));
        let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

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
            Device::cuda_if_available(),
            rng.gen(),
            &beta_net,
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
