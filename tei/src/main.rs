use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::TryRecvError,
        Arc,
    },
    time::{Duration, Instant},
};

use fast_tak::takparse::{Color, Move};
use protocol::{GoOption, Id, Input, Output, ParseInputError, Position, ValueType};
use takzero::{
    network::{
        net6_simhash::{Env, Net, HALF_KOMI, N},
        Network,
    },
    search::node::Node,
};
use thiserror::Error;
use rand::prelude::*;
use fast_tak::Symmetry;
use ordered_float::NotNan;

mod protocol;

const MAX_ERRORS_IN_A_ROW: usize = 5;
const DURATION_BETWEEN_INFO_PRINTS: Duration = Duration::from_millis(300);
const DURATION_BEFORE_CHECKING_INPUT: Duration = Duration::from_secs(1);
const BATCH_SIZE: usize = 8;
const BETA: f32 = 0.0;
const SAMPLED_PLIES: u16 = 30;

#[allow(clippy::too_many_lines)] // FIXME
#[allow(clippy::cognitive_complexity)] // FIXME
fn main() {
    env_logger::init();
    let mut line = String::new();
    let stdin = std::io::stdin();
    let mut rng = rand::rng();

    // Wait for first `tei` message.
    let Ok(Input::Tei) = get_input(&stdin, &mut line) else {
        log::error!("first message received should be `tei`");
        return;
    };

    // Print name and author.
    println!("{}", Output::Id(Id::Name("TakZero")));
    println!("{}", Output::Id(Id::Author("Viliam Vadocz (0x57696c6c)")));

    // Describe engine options.
    println!("{}", Output::Option {
        name: "model",
        value_type: ValueType::String,
        default: Some("./path/to/model.ot"),
        min: None,
        max: None,
        variables: &[]
    });
    println!("{}", Output::Option {
        name: "HalfKomi",
        value_type: ValueType::Combo,
        default: Some("4"),
        min: None,
        max: None,
        variables: &["4"]
    });

    println!("{}", Output::Ok);

    // Configure engine options.
    let mut model_path = None;
    loop {
        match get_input(&stdin, &mut line) {
            Ok(Input::IsReady) => break,
            Ok(Input::Option { name, value }) => match name.as_ref() {
                "model" => model_path = Some(value),
                "HalfKomi" => {
                    let Ok(half_komi) = value.parse::<i8>() else {
                        log::error!("could not parse half komi");
                        return;
                    };
                    if half_komi != HALF_KOMI {
                        log::error!(
                            "half komi of {half_komi} was requested, but only {HALF_KOMI} is \
                             supported"
                        );
                        return;
                    }
                }
                _ => log::warn!("unknown option: {name}"),
            },
            Ok(_) => log::warn!("only expecting `isready` or `option` messages"),
            Err(err) => log::error!("{err}"),
        }
    }

    // Validate configuration.
    let Some(model_path) = model_path else {
        log::error!("model path must be set");
        return;
    };

    // Load engine / model.
    let device = tch::Device::cuda_if_available();
    if !device.is_cuda() {
        log::warn!("CUDA is not available, running on CPU");
    }
    let net = match Net::load_partial(model_path, device) {
        Ok(net) => net,
        Err(err) => {
            log::error!("failed to load model: {err}");
            return;
        }
    };

    // Start thread to parse user input and send it over.
    let (tx, rx) = std::sync::mpsc::channel();
    let should_stop = Arc::new(AtomicBool::new(false));
    let should_stop_2 = should_stop.clone();
    let input_thread = std::thread::spawn(move || {
        let should_stop = should_stop_2;
        let mut errors_in_a_row = 0;
        while !should_stop.load(Ordering::Relaxed) {
            match get_input(&stdin, &mut line) {
                Ok(x) => tx.send(x).expect("Main thread should still be alive."),
                Err(err) => {
                    log::error!("{err}");
                    errors_in_a_row += 1;
                    if errors_in_a_row >= MAX_ERRORS_IN_A_ROW {
                        log::error!("there were {MAX_ERRORS_IN_A_ROW} errors in a row");
                        should_stop.store(true, Ordering::Relaxed);
                    }
                    continue;
                }
            }
            errors_in_a_row = 0;
        }
    });

    // Ready!
    println!("{}", Output::ReadyOk);

    let mut node = Node::default();
    let mut env = Env::default();
    node.simulate_simple(&net, env.clone(), 0.0);
    let mut go_status = GoStatus::Stopped;
    let mut go_options = Vec::new();

    let mut nodes = None;
    let mut move_time = None;
    let mut my_time = None;
    let mut my_inc = None;
    let mut visits_at_start = 0;
    let mut start = Instant::now();
    let mut last_info = Instant::now();
    let mut sent_info = false;

    let mut last_position: Position = Position::StartPos;
    let mut last_moves: Vec<Move> = vec![];

    // Process user input
    'main_loop: while !should_stop.load(Ordering::Relaxed) {
        let last_checked_input = Instant::now();
        match if matches!(go_status, GoStatus::Stopped) {
            rx.recv().map_err(|_| TryRecvError::Disconnected)
        } else {
            rx.try_recv()
        } {
            Ok(Input::IsReady) => println!("{}", Output::ReadyOk),
            Ok(Input::NewGame { size }) => {
                if size != N {
                    log::error!("the engine is compiled only for size {N}");
                    break 'main_loop;
                }
                node = Node::default();
                env = Env::default();
            }
            Ok(Input::Position { position, moves }) => {
                if position == last_position && moves.starts_with(&last_moves) {
                    // Tree re-use!
                    let new_moves = &moves[last_moves.len()..];
                    for my_move in new_moves {
                        node.descend(my_move);
                        if let Err(err) = env.play(*my_move) {
                            log::error!("could not play move {my_move}: {err}");
                            break;
                        }
                    }
                } else {
                    // Restart
                    node = Node::default();
                    env = match &position {
                        Position::StartPos => Env::default(),
                        Position::Tps(tps) => tps.clone().into(),
                    };
                    for &my_move in &moves {
                        if let Err(err) = env.play(my_move) {
                            log::error!("could not play move {my_move}: {err}");
                            break;
                        }
                    }
                }
                last_position = position;
                last_moves = moves;
            }
            Ok(Input::Stop) => {
                go_status = GoStatus::Stopping;
            }
            Ok(Input::Go(options)) => {
                go_options.clear();
                go_options.extend(options);
                go_status = GoStatus::Starting;

                // HACK: Opening pamphlet
                if env.ply == 0 {
                    println!("{}", Output::BestMove("a6".parse().unwrap()));
                    go_status = GoStatus::Stopped;
                } else if env.ply == 1 {
                    const OPENINGS: &[([&str; 2], u32)] = &[
                        (["a6", "f1"], 5), // opposite
                        (["a6", "a1"], 4), // adjacent
                        (["a6", "a5"], 1), // hug
                        //
                        (["b6", "a6"], 1), // reverse hug
                    ];
                    let mut viable = vec![];
                    for &([first, second], weight) in OPENINGS {
                        let mut dummy = Env::default();
                        dummy
                            .play(first.parse().expect("Book moves should be valid moves"))
                            .expect("Book moves should be safe to play");

                        if let Some((i, _)) = dummy
                            .symmetries()
                            .into_iter()
                            .enumerate()
                            .find(|(_, symm)| &env == symm)
                        {
                            let book_move: Move = second.parse().expect("Book moves should be valid moves");
                            let symmetries = Symmetry::<N>::symmetries(&book_move);
                            viable.push((symmetries[i], weight));
                        }
                    }
                    if !viable.is_empty() {
                        let chosen = viable.choose_weighted(&mut rng, |x| x.1).unwrap();
                        println!("{}", Output::BestMove(chosen.0));
                        go_status = GoStatus::Stopped;
                    }
                }
            }
            Ok(Input::Option { .. }) => log::warn!("it's too late to specify options"),
            Ok(Input::Tei) => log::warn!("tei does not make sense here"),
            Ok(Input::Quit) | Err(TryRecvError::Disconnected) => break 'main_loop,
            Err(TryRecvError::Empty) => {}
        }

        if matches!(go_status, GoStatus::Starting) {
            for option in go_options.drain(..) {
                match option {
                    GoOption::Nodes(amount) => nodes = Some(amount),
                    GoOption::MoveTime(duration) => move_time = Some(duration),
                    GoOption::WhiteTime(duration) if env.to_move == Color::White => {
                        my_time = Some(duration);
                    }
                    GoOption::BlackTime(duration) if env.to_move == Color::Black => {
                        my_time = Some(duration);
                    }
                    GoOption::WhiteIncrement(duration) if env.to_move == Color::White => {
                        my_inc = Some(duration);
                    }
                    GoOption::BlackIncrement(duration) if env.to_move == Color::Black => {
                        my_inc = Some(duration);
                    }
                    GoOption::Infinite => nodes = Some(usize::MAX), // HACK
                    _ => log::warn!("ignored `go` option {option:?}"),
                }
            }
            if nodes.is_none() && move_time.is_none() && (my_time.is_none() || my_inc.is_none()) {
                log::warn!("no understood stopping condition given");
            }
            // Very basic time management.
            if let (None, Some(my_time), Some(my_inc)) = (move_time, my_time, my_inc) {
                move_time = Some((my_time / 30 + my_inc).min(Duration::from_secs(3)));
            }
            visits_at_start = node.visit_count;
            sent_info = false;
            last_info = Instant::now();
            start = Instant::now();
            go_status = GoStatus::Going;
        }

        if matches!(go_status, GoStatus::Going) {
            loop {
                // node.simulate_batch(&net, &env, BETA, BATCH_SIZE);
                node.simulate_simple(&net, env.clone(), BETA);
                let visits = (node.visit_count - visits_at_start) as _;
                let elapsed = start.elapsed();

                let done = nodes.is_some_and(|amount| visits >= amount)
                    || move_time.is_some_and(|duration| elapsed >= duration);

                if last_info.elapsed() >= DURATION_BETWEEN_INFO_PRINTS {
                    println!("{}", Output::Info {
                        time: elapsed,
                        nodes: visits,
                        score: node.evaluation,
                        principal_variation: node.principal_variation().collect(),
                    });
                    sent_info = true;
                    last_info = Instant::now();
                }
                if done {
                    go_status = GoStatus::Stopping;
                    break;
                }
                // Go check for `stop`.
                if last_checked_input.elapsed() >= DURATION_BEFORE_CHECKING_INPUT {
                    continue 'main_loop;
                }
            }
        }

        if matches!(go_status, GoStatus::Stopping) {
            if !sent_info {
                println!("{}", Output::Info {
                    time: start.elapsed(),
                    nodes: (node.visit_count - visits_at_start) as _,
                    score: node.evaluation,
                    principal_variation: node.principal_variation().collect(),
                });
            }
            println!("{}", Output::BestMove(node.select_selfplay_action((env.ply < SAMPLED_PLIES).then_some(0), NotNan::new(0.25).unwrap(), &mut rng)));
            nodes = None;
            move_time = None;
            my_time = None;
            my_inc = None;
            go_status = GoStatus::Stopped;
        }
    }

    should_stop.store(true, Ordering::Relaxed);
    input_thread
        .join()
        .expect("Input thread should shut down gracefully.");
}

enum GoStatus {
    Stopped,
    Starting,
    Going,
    Stopping,
}

#[derive(Debug, Error)]
enum GetInputError {
    #[error("reading from stdin failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] ParseInputError),
}

fn get_input(stdin: &std::io::Stdin, line: &mut String) -> Result<Input, GetInputError> {
    line.clear();
    stdin.read_line(line)?;
    Ok(line.trim().parse()?)
}
