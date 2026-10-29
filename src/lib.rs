//! Deterministic game rules, independent of terminal I/O and agent policies.
//!
//! Actions are row-major indices `0..9`. X always starts. One call to
//! [`Game::step`] places exactly one mark, then hands control to the opponent.
//!
//! ```
//! use tic_tac_toe::{Agent, Game, RandomAgent};
//!
//! let mut game = Game::new();
//! let mut bot = RandomAgent::new(42);
//! while let Some(player) = game.current_player() {
//!     let observation = game.observe(player);
//!     let action = bot.select_action(&observation).unwrap();
//!     let step = game.step(action).unwrap();
//!     assert!((-1.0..=1.0).contains(&step.reward_for(player)));
//! }
//! assert!(game.outcome().is_terminal());
//! ```

mod agent;
pub mod duo;
pub mod duo_engine;
pub mod duo_mcts;
pub mod duo_replay;
pub mod duo_search;
pub mod duo_solver;
pub mod duo_v2;
mod game;

pub use agent::{Agent, RandomAgent};
pub use game::{ACTION_COUNT, Game, MoveError, Observation, Outcome, Player, Step};
