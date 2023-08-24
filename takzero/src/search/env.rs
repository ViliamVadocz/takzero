use fast_tak::{takparse::Move, Game, Reserves};

pub trait Environment: Send + Sync + Clone + Default {
    type Action: Send + Sync + Clone + PartialEq;

    fn populate_actions(&self, actions: &mut Vec<Self::Action>);
    fn step(&mut self, action: Self::Action);
    fn terminal(&self) -> Option<Terminal>;
    fn steps(&self) -> u16;
}

pub enum Terminal {
    Win,
    Loss,
    Draw,
}

impl<const N: usize, const HALF_KOMI: i8> Environment for Game<N, HALF_KOMI>
where
    Reserves<N>: Default,
{
    type Action = Move;

    fn populate_actions(&self, actions: &mut Vec<Self::Action>) {
        self.possible_moves(actions);
    }

    fn step(&mut self, action: Self::Action) {
        self.play(action).expect("Action should be valid.");
    }

    fn terminal(&self) -> Option<Terminal> {
        match self.result() {
            fast_tak::GameResult::Winner { color, .. } => {
                if color == self.to_move {
                    Some(Terminal::Win)
                } else {
                    Some(Terminal::Loss)
                }
            }
            fast_tak::GameResult::Draw { .. } => Some(Terminal::Draw),
            fast_tak::GameResult::Ongoing => None,
        }
    }

    fn steps(&self) -> u16 {
        self.ply
    }
}

impl From<Terminal> for f32 {
    fn from(value: Terminal) -> Self {
        match value {
            Terminal::Win => 1.0,
            Terminal::Loss => -1.0,
            Terminal::Draw => 0.0,
        }
    }
}
