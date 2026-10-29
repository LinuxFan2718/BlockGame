//! Exact endgame solver (implemented in phase 2 of the v2 roadmap).
use crate::duo::Game;
use std::sync::atomic::AtomicU64;

/// Exact margin for the player to move and a best action, or None if the
/// position cannot be proved within `budget` nodes.
pub fn solve_with_budget(_game: &Game, _budget: u64, _nodes: &AtomicU64) -> Option<(i32, usize)> {
    None
}
