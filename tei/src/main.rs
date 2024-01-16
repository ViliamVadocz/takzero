use std::{io::Read, time::Instant};

use protocol::{GoOption, Id, Input, Output, ParseInputError, Position, ValueType};
use takzero::{
    network::{
        net5::{Env, Net, RndNormalizationContext, N},
        Network,
    },
    search::node::Node,
};
use thiserror::Error;

mod protocol;

fn main() {
    env_logger::init();
    let mut line = String::new();
    let mut stdin = std::io::stdin();

    // Wait for first `tei` message.
    let Ok(Input::Tei) = get_input(&mut stdin, &mut line) else {
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
        default: Some("./path/to/model.ot")
    });

    println!("{}", Output::Ok);

    // Configure engine options.
    let mut model_path = None;
    loop {
        match get_input(&mut stdin, &mut line) {
            Ok(Input::IsReady) => break,
            Ok(Input::Option { name, value }) => {
                if name == "model" {
                    model_path = Some(value);
                } else {
                    log::warn!("unknown option: {name}");
                }
            }
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
    let Ok(net) = Net::load(model_path, tch::Device::Cuda(0)) else {
        log::error!("model path should be valid and contain the correct model weights");
        return;
    };
    println!("{}", Output::ReadyOk);

    let mut node = Node::default();
    let mut env = Env::default();
    let mut context = RndNormalizationContext::new(0.0);
    node.simulate_simple(&net, env.clone(), 0.0, &mut context);

    loop {
        match get_input(&mut stdin, &mut line) {
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
                go(&net, &env, &mut node, &mut context, go_options);
                println!("{}", Output::BestMove(node.select_best_action()));
            }

            Ok(_) => log::warn!("unhandled message"),
            Err(err) => {
                log::error!("{err}");
                continue;
            }
        };
    }
}

fn go(
    net: &Net,
    env: &Env,
    node: &mut Node<Env>,
    context: &mut RndNormalizationContext,
    go_options: Vec<GoOption>,
) {
    const BETA: f32 = 0.0;

    let mut nodes = None;
    let mut move_time = None;

    for option in go_options {
        match option {
            GoOption::Nodes(amount) => nodes = Some(amount),
            GoOption::MoveTime(duration) => move_time = Some(duration),
            _ => log::warn!("ignored `go` option"),
        }
    }

    if nodes.is_none() && move_time.is_none() {
        log::error!("no understood stopping condition given");
        return;
    }

    let start = Instant::now();
    for visits in 0.. {
        node.simulate_simple(net, env.clone(), BETA, context);

        if nodes.is_some_and(|amount| visits >= amount)
            || move_time.is_some_and(|duration| start.elapsed() >= duration)
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

fn get_input(stdin: &mut std::io::Stdin, line: &mut String) -> Result<Input, GetInputError> {
    line.clear();
    stdin.read_to_string(line)?;
    Ok(line.trim().parse()?)
}
