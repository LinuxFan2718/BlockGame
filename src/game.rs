use std::fmt;

pub const ACTION_COUNT: usize = 9;

const WINNING_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    X,
    O,
}

impl Player {
    pub fn opponent(self) -> Self {
        match self {
            Self::X => Self::O,
            Self::O => Self::X,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X => write!(f, "X"),
            Self::O => write!(f, "O"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    InProgress,
    Win(Player),
    Draw,
}

impl Outcome {
    pub fn is_terminal(self) -> bool {
        self != Self::InProgress
    }

    /// Sparse, zero-sum rewards: +1 for winning, -1 for losing, otherwise 0.
    pub fn reward_for(self, player: Player) -> f32 {
        match self {
            Self::Win(winner) if winner == player => 1.0,
            Self::Win(_) => -1.0,
            Self::InProgress | Self::Draw => 0.0,
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InProgress => write!(f, "game in progress"),
            Self::Win(player) => write!(f, "{player} wins"),
            Self::Draw => write!(f, "draw"),
        }
    }
}

/// A numeric view for an agent, with no information hidden from either player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    /// Whose marks are encoded as +1. This perspective never changes implicitly.
    pub perspective: Player,
    /// Row-major cells: +1 = own mark, -1 = opponent, 0 = empty.
    pub cells: [i8; ACTION_COUNT],
    /// +1 = perspective's turn, -1 = opponent's turn, 0 = game over.
    pub to_move: i8,
    /// True exactly for legal actions; all false after the game ends.
    pub legal_mask: [bool; ACTION_COUNT],
}

/// The result of one legal move. Query rewards for BOTH players at termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    pub player: Player,
    pub action: usize,
    pub outcome: Outcome,
}

impl Step {
    pub fn is_terminal(self) -> bool {
        self.outcome.is_terminal()
    }

    pub fn reward_for(self, player: Player) -> f32 {
        self.outcome.reward_for(player)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveError {
    OutOfBounds { action: usize },
    Occupied { action: usize },
    GameOver { outcome: Outcome },
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { action } => {
                write!(f, "action {action} is outside 0..{ACTION_COUNT}")
            }
            Self::Occupied { action } => write!(f, "action {action} targets an occupied square"),
            Self::GameOver { outcome } => write!(f, "the game has ended: {outcome}"),
        }
    }
}

impl std::error::Error for MoveError {}

/// Private state ensures a game can only be constructed through legal play.
/// Cloning a game gives an independent snapshot for simulations or search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Game {
    board: [Option<Player>; ACTION_COUNT],
    to_move: Player,
    outcome: Outcome,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        Self {
            board: [None; ACTION_COUNT],
            to_move: Player::X,
            outcome: Outcome::InProgress,
        }
    }

    /// Start a fresh episode. The rules themselves contain no randomness.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn board(&self) -> &[Option<Player>; ACTION_COUNT] {
        &self.board
    }

    /// Terminal positions have no current player and no legal actions.
    pub fn current_player(&self) -> Option<Player> {
        (!self.outcome.is_terminal()).then_some(self.to_move)
    }

    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    pub fn legal_actions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..ACTION_COUNT)
            .filter(|&action| !self.outcome.is_terminal() && self.board[action].is_none())
    }

    pub fn legal_mask(&self) -> [bool; ACTION_COUNT] {
        std::array::from_fn(|action| !self.outcome.is_terminal() && self.board[action].is_none())
    }

    /// Explicit perspective avoids silently swapping reward or feature signs.
    pub fn observe(&self, perspective: Player) -> Observation {
        Observation {
            perspective,
            cells: self.board.map(|cell| match cell {
                Some(player) if player == perspective => 1,
                Some(_) => -1,
                None => 0,
            }),
            to_move: match self.current_player() {
                Some(player) if player == perspective => 1,
                Some(_) => -1,
                None => 0,
            },
            legal_mask: self.legal_mask(),
        }
    }

    /// Apply ONE player's move. Invalid actions leave the entire state unchanged.
    /// They return an error, not a reward or a forfeited turn.
    pub fn step(&mut self, action: usize) -> Result<Step, MoveError> {
        if self.outcome.is_terminal() {
            return Err(MoveError::GameOver {
                outcome: self.outcome,
            });
        }
        if action >= ACTION_COUNT {
            return Err(MoveError::OutOfBounds { action });
        }
        if self.board[action].is_some() {
            return Err(MoveError::Occupied { action });
        }

        let player = self.to_move;
        self.board[action] = Some(player);
        self.outcome = if WINNING_LINES
            .iter()
            .any(|line| line.iter().all(|&cell| self.board[cell] == Some(player)))
        {
            Outcome::Win(player)
        } else if self.board.iter().all(Option::is_some) {
            Outcome::Draw
        } else {
            Outcome::InProgress
        };
        self.to_move = player.opponent();

        Ok(Step {
            player,
            action,
            outcome: self.outcome,
        })
    }
}

/// Empty squares use the terminal UI's human-friendly numbering, 1 through 9.
impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..3 {
            for col in 0..3 {
                let action = row * 3 + col;
                match self.board[action] {
                    Some(player) => write!(f, " {player} ")?,
                    None => write!(f, " {} ", action + 1)?,
                }
                if col < 2 {
                    write!(f, "|")?;
                }
            }
            if row < 2 {
                writeln!(f, "\n---+---+---")?;
            }
        }
        Ok(())
    }
}
