use fast_tak::takparse::Move;
use ordered_float::NotNan;
use rand::{
    distributions::{Distribution, WeightedIndex},
    seq::SliceRandom,
    Rng,
    SeedableRng,
};
use takzero::{
    network::{
        net4_ensemble::{Env, Net, ENSEMBLE_SIZE, HALF_KOMI, MAXIMUM_VARIANCE, N},
        repr::{game_to_tensor, move_mask, output_size, policy_tensor},
        EnsembleNetwork,
        Network,
    },
    search::{env::Environment, eval::Eval, DISCOUNT_FACTOR},
    target::{get_targets, Augment, Target},
};
use tch::{
    nn::{Adam, Optimizer, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

const BATCH_SIZE: usize = 256;
const DEVICE: Device = Device::Cuda(0);
const LEARNING_RATE: f64 = 1e-4;

const MINIMUM_UBE_TARGET: f64 = -10.0;
const FORCED_USES: usize = 4;

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1_234_567);
    let net = Net::new(DEVICE, Some(rng.gen()));

    let mut targets = get_targets::<N, HALF_KOMI>("targets.txt").unwrap();
    let mut self_play: Vec<_> = targets
        .by_ref()
        .take(320_000)
        .map(|t| (t, FORCED_USES))
        .collect();
    let mut reanalyze: Vec<_> = targets
        .by_ref()
        .take(320_000)
        .map(|t| (t, FORCED_USES))
        .collect();
    assert_eq!(targets.count(), 0);

    let mut opt = Adam::default().build(net.vs(), LEARNING_RATE).unwrap();

    while self_play.len() >= BATCH_SIZE / 2 && reanalyze.len() >= BATCH_SIZE / 2 {
        // Sample batch.
        self_play.shuffle(&mut rng);
        reanalyze.shuffle(&mut rng);
        let self_play_batch = self_play.split_off(self_play.len() - BATCH_SIZE / 2);
        let reanalyze_batch = reanalyze.split_off(reanalyze.len() - BATCH_SIZE / 2);
        assert_eq!(self_play_batch.len() + reanalyze_batch.len(), BATCH_SIZE);
        let tensors = create_input_and_target_tensors(
            &net,
            self_play_batch
                .iter()
                .map(|(t, _)| t)
                .chain(reanalyze_batch.iter().map(|(t, _)| t)),
            &mut rng,
        );
        self_play.extend(
            self_play_batch
                .into_iter()
                .filter(|(_, x)| *x <= 1)
                .map(|(t, x)| (t, x - 1)),
        );
        reanalyze.extend(
            reanalyze_batch
                .into_iter()
                .filter(|(_, x)| *x <= 1)
                .map(|(t, x)| (t, x - 1)),
        );

        take_step(&net, &mut opt, tensors);

        // TODO: Collect data (current, after, early, late, etc.)
    }
}

struct Tensors {
    input: Tensor,
    mask: Tensor,
    target_value: Tensor,
    target_policy: Tensor,
    target_ube: Tensor,
    target_ensemble: Tensor,
}

fn create_input_and_target_tensors<'a>(
    net: &Net, // for bootstrapped targets
    batch: impl Iterator<Item = &'a Target<Env>>,
    rng: &mut impl Rng,
) -> Tensors {
    // Create input tensors.
    let mut inputs = Vec::with_capacity(BATCH_SIZE);
    let mut policy_targets = Vec::with_capacity(BATCH_SIZE);
    let mut masks = Vec::with_capacity(BATCH_SIZE);
    let mut value_targets = Vec::with_capacity(BATCH_SIZE);
    let mut ube_targets = Vec::with_capacity(BATCH_SIZE);
    let mut ensemble_targets = Vec::with_capacity(BATCH_SIZE);
    for target in batch {
        let target = target.augment(rng);
        inputs.push(game_to_tensor(&target.env, DEVICE));
        policy_targets.push(policy_tensor::<N>(&target.policy, DEVICE));
        masks.push(move_mask::<N>(
            &target.policy.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
            DEVICE,
        ));
        value_targets.push(target.value);
        ube_targets.push(target.ube);
        ensemble_targets.push(get_ensemble_targets(net, &target.env, &target.policy, rng));
    }

    // Get network output.
    let input = Tensor::cat(&inputs, 0).to(DEVICE);
    let mask = Tensor::cat(&masks, 0).to(DEVICE);
    // Get the target.
    let target_policy = Tensor::stack(&policy_targets, 0)
        .view([BATCH_SIZE as i64, output_size::<N>() as i64])
        .to(DEVICE);
    let target_value = Tensor::from_slice(&value_targets).unsqueeze(1).to(DEVICE);
    let target_ube = Tensor::from_slice(&ube_targets)
        .unsqueeze(1)
        .to(DEVICE)
        .log()
        .clamp(MINIMUM_UBE_TARGET, MAXIMUM_VARIANCE.ln());
    let target_ensemble = Tensor::stack(&ensemble_targets, 0);

    Tensors {
        input,
        mask,
        target_value,
        target_policy,
        target_ube,
        target_ensemble,
    }
}

// TODO: Do this for an entire batch at once.
// TODO: Don't run policy, value, UBE heads since they are not needed.
fn get_ensemble_targets(
    net: &Net,
    env: &Env,
    policy: &[(Move, NotNan<f32>)],
    rng: &mut impl Rng,
) -> Tensor {
    // Select an action proportional to the improved policy.
    let weighted_index = WeightedIndex::new(policy.iter().map(|(_, p)| p.into_inner())).expect(
        "there should be at least one action and the improved policy should not be negative",
    );
    let action = policy[weighted_index.sample(rng)].0;
    // Take a step in the environment.
    let mut clone = env.clone();
    clone.step(action);
    // If the state is terminal, use the terminal value. Otherwise bootstrap from
    // network predictions.
    clone.terminal().map_or_else(
        || {
            let xs = game_to_tensor::<N, HALF_KOMI>(&clone, DEVICE);
            let bootstrap = net.forward_t(&xs, false).3;
            -DISCOUNT_FACTOR * bootstrap
        },
        |t| {
            Tensor::ones([1, ENSEMBLE_SIZE as i64], (Kind::Float, DEVICE))
                * f64::from(f32::from(Eval::from(t).negate()))
        },
    )
}

fn take_step(net: &Net, opt: &mut Optimizer, tensors: Tensors) {
    // Get network output.
    let (policy, network_value, network_ube, ensemble_value) = net.forward_t(&tensors.input, true);
    let log_softmax_network_policy = policy
        .masked_fill(&tensors.mask, f64::from(f32::MIN))
        .view([-1, output_size::<N>() as i64])
        .log_softmax(1, Kind::Float);

    // Calculate loss.
    let loss_policy = -(log_softmax_network_policy * &tensors.target_policy).sum(Kind::Float)
        / i64::try_from(BATCH_SIZE).unwrap();
    let loss_value = (tensors.target_value - network_value)
        .square()
        .mean(Kind::Float);
    let loss_ube = (tensors.target_ube - network_ube)
        .square()
        .mean(Kind::Float);
    let loss_ensemble = (tensors.target_ensemble - ensemble_value)
        .square()
        .mean(Kind::Float);

    let loss = &loss_policy + &loss_value + &loss_ube + &loss_ensemble;
    #[rustfmt::skip]
    println!(
        "loss = {loss:?}\n\
         loss_policy = {loss_policy:?}\n\
         loss_value = {loss_value:?}\n\
         loss_ube = {loss_ube:?}\n\
         loss_ensemble = {loss_ensemble:?}"
    );

    // Take step.
    opt.backward_step(&loss);
}
