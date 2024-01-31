// This is not a complete implementation.
// https://ucichessengine.wordpress.com/2011/03/16/description-of-uci-protocol/

use std::{fmt, num::ParseIntError, str::FromStr, time::Duration};

use fast_tak::takparse::{Move, ParseMoveError, ParseTpsError, Tps};
use thiserror::Error;

pub enum Input {
    Tei,
    IsReady,
    Option {
        name: String,
        value: String,
    },
    NewGame {
        size: usize,
    },
    Position {
        position: Position,
        moves: Vec<Move>,
    },
    Go(Vec<GoOption>),
    // Stop,
    Quit,
}

pub enum Position {
    StartPos,
    Tps(Tps),
}

pub enum GoOption {
    WhiteTime(Duration),
    BlackTime(Duration),
    WhiteIncrement(Duration),
    BlackIncrement(Duration),
    // Infinite,
    MoveTime(Duration),
    Nodes(usize),
}

#[derive(Debug, Error)]
pub enum ParseInputError {
    #[error("missing first word")]
    MissingFirstWord,
    #[error("missing `name` or the actual option name after `option`")]
    MissingName,
    #[error("missing `value` or the actual value after `option name <id>`")]
    MissingValue,
    #[error("missing size after `newgame`")]
    MissingSize,
    #[error("new game size parse error: {0}")]
    ParseSize(#[from] ParseIntError),
    #[error("missing second word after `position`")]
    MissingPosition,
    #[error("missing TPS after `position tps`")]
    MissingTps,
    #[error("position tps parse error: {0}")]
    ParseTps(#[from] ParseTpsError),
    #[error("move parse error: {0}")]
    ParseMove(#[from] ParseMoveError),
    #[error("missing seconds after time parameter in `go`")]
    MissingSeconds,
    #[error("missing the amount of nodes after `go nodes`")]
    MissingAmount,
    #[error("unrecognized option")]
    Unrecognized,
}

impl FromStr for Input {
    type Err = ParseInputError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut words = s.split_whitespace();
        match words.next().ok_or(ParseInputError::MissingFirstWord)? {
            "tei" => Ok(Self::Tei),
            "isready" => Ok(Self::IsReady),
            "option" => {
                words
                    .next()
                    .filter(|&word| word == "name")
                    .ok_or(ParseInputError::MissingName)?;
                let name = words
                    .next()
                    .ok_or(ParseInputError::MissingName)?
                    .to_string();
                words
                    .next()
                    .filter(|&word| word == "value")
                    .ok_or(ParseInputError::MissingValue)?;
                let value = words
                    .next()
                    .ok_or(ParseInputError::MissingName)?
                    .to_string();
                Ok(Self::Option { name, value })
            }
            "teinewgame" => {
                let size: usize = words.next().ok_or(ParseInputError::MissingSize)?.parse()?;
                Ok(Self::NewGame { size })
            }
            "position" => {
                let position = match words.next().ok_or(ParseInputError::MissingPosition)? {
                    "startpos" => Position::StartPos,
                    "tps" => {
                        let tps: Tps = words.next().ok_or(ParseInputError::MissingTps)?.parse()?;
                        Position::Tps(tps)
                    }
                    _ => return Err(ParseInputError::Unrecognized),
                };
                let mut moves = Vec::new();
                if words.next() == Some("moves") {
                    moves.extend(words.map(Move::from_str).collect::<Result<Vec<_>, _>>()?);
                }
                Ok(Self::Position { position, moves })
            }
            "go" => {
                let mut go_options = Vec::new();
                while let Some(word) = words.next() {
                    go_options.push(match word {
                        "wtime" | "btime" | "winc" | "binc" | "movetime" => {
                            let seconds = words
                                .next()
                                .ok_or(ParseInputError::MissingSeconds)?
                                .parse()?;
                            let duration = Duration::from_secs(seconds);
                            match word {
                                "wtime" => GoOption::WhiteTime(duration),
                                "btime" => GoOption::BlackTime(duration),
                                "winc" => GoOption::WhiteIncrement(duration),
                                "binc" => GoOption::BlackIncrement(duration),
                                "movetime" => GoOption::MoveTime(duration),
                                _ => unreachable!(),
                            }
                        }
                        "nodes" => {
                            let amount = words
                                .next()
                                .ok_or(ParseInputError::MissingAmount)?
                                .parse()?;
                            GoOption::Nodes(amount)
                        }
                        // "infinite" => GoOption::Infinite,
                        _ => return Err(ParseInputError::Unrecognized),
                    });
                }
                Ok(Self::Go(go_options))
            }
            // "stop" => Ok(Self::Stop),
            "quit" => Ok(Self::Quit),
            _ => Err(ParseInputError::Unrecognized),
        }
    }
}

pub enum Output {
    Id(Id),
    Option {
        name: &'static str,
        value_type: ValueType,
        default: Option<&'static str>,
    },
    Ok,
    ReadyOk,
    BestMove(Move),
}

pub enum Id {
    Name(&'static str),
    Author(&'static str),
}

#[allow(unused)]
pub enum ValueType {
    Check,
    Spin,
    Combo,
    Button,
    String,
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Check => write!(f, "check"),
            Self::Spin => write!(f, "spin"),
            Self::Combo => write!(f, "combo"),
            Self::Button => write!(f, "button"),
            Self::String => write!(f, "string"),
        }
    }
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id(Id::Name(name)) => writeln!(f, "id name {name}"),
            Self::Id(Id::Author(name)) => writeln!(f, "id author {name}"),
            Self::Option {
                name,
                value_type,
                default,
            } => {
                write!(f, "option name {name} type {value_type}")?;
                if let Some(default) = default {
                    write!(f, " default {default}")?;
                }
                writeln!(f)
            }
            Self::Ok => writeln!(f, "teiok"),
            Self::ReadyOk => writeln!(f, "readyok"),
            Self::BestMove(the_move) => writeln!(f, "bestmove {the_move}"),
        }
    }
}