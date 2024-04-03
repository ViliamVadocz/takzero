use std::time::Instant;

use fast_tak::takparse::Color;
use protocol::{GoOption, Id, Input, Output, ParseInputError, Position, ValueType};
use takzero::{
    network::{
        net5::{Env, Net, HALF_KOMI, N},
        Network,
    },
    search::node::Node,
};
use thiserror::Error;

mod protocol;

const MAX_ERRORS_IN_A_ROW: usize = 5;
const NODES_PER_INFO: usize = 200;

#[allow(clippy::too_many_lines)]
fn main() {
    env_logger::init();
    let mut line = String::new();
    let stdin = std::io::stdin();

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
                            "half komi of {half_komi} was request, but only {HALF_KOMI} is \
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
    let net = match Net::load(model_path, tch::Device::Cuda(0)) {
        Ok(net) => net,
        Err(err) => {
            log::error!("failed to load model: {}", err);
            return;
        }
    };
    println!("{}", Output::ReadyOk);

    let mut node = Node::default();
    let mut env = Env::default();
    node.simulate_simple(&net, env.clone(), 0.0);

    let mut errors_in_a_row = 0;
    loop {
        match get_input(&stdin, &mut line) {
            Ok(Input::IsReady) => println!("{}", Output::ReadyOk),
            Ok(Input::NewGame { size }) => {
                if size != N {
                    log::error!("the engine is compiled only for size {N}");
                }
                node = Node::default();
                env = Env::default();
            }
            Ok(Input::Position { position, moves }) => {
                node = Node::default();
                env = match position {
                    Position::StartPos => Env::default(),
                    Position::Tps(tps) => tps.into(),
                };
                for my_move in moves {
                    if let Err(err) = env.play(my_move) {
                        log::error!("could not play move {my_move}: {err}");
                        break;
                    }
                }
            }
            Ok(Input::Quit) => break,
            Ok(Input::Go(go_options)) => {
                go(&net, &env, &mut node, go_options);
                println!("{}", Output::BestMove(node.select_best_action()));
            }

            Ok(_) => log::warn!("unhandled message"),
            Err(err) => {
                log::error!("{err}");
                errors_in_a_row += 1;
                if errors_in_a_row >= MAX_ERRORS_IN_A_ROW {
                    log::error!("there were {MAX_ERRORS_IN_A_ROW} errors in a row");
                    return;
                }
                continue;
            }
        };
        errors_in_a_row = 0;
    }
}

fn go(net: &Net, env: &Env, node: &mut Node<Env>, go_options: Vec<GoOption>) {
    const BETA: f32 = 0.0;

    let mut nodes = None;
    let mut move_time = None;

    let mut my_time = None;
    let mut my_inc = None;

    for option in go_options {
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
            _ => log::warn!("ignored `go` option {option:?}"),
        }
    }

    if nodes.is_none() && move_time.is_none() && (my_time.is_none() || my_inc.is_none()) {
        log::error!("no understood stopping condition given");
        return;
    }
    // Very basic time management.
    if let (None, Some(my_time), Some(my_inc)) = (move_time, my_time, my_inc) {
        move_time = Some(my_time / 10 + 3 * my_inc / 4);
    }

    let start = Instant::now();
    for visits in 1.. {
        node.simulate_simple(net, env.clone(), BETA);

        let elapsed = start.elapsed();

        if visits % NODES_PER_INFO == 0 {
            println!("{}", Output::Info {
                time: elapsed,
                nodes: visits,
                score: node.evaluation,
            });
        }

        if nodes.is_some_and(|amount| visits >= amount)
            || move_time.is_some_and(|duration| elapsed >= duration)
        {
            break;
        }
    }
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
