//! Exact endgame solver: negamax alpha-beta over the final score margin.
//!
//! Near the end of a Duo game few placements remain, so the whole game tree
//! can be searched. The result is the exact margin (mover's score minus the
//! opponent's) under best play by both sides, plus a move that achieves it.
//! Maximizing the margin also maximizes the result: if a win exists, the best
//! margin is positive.
//!
//! A per-thread transposition table keeps proven bounds. Entries are facts
//! about positions, so they stay valid across solves; searches that run out
//! of budget never store anything.
use crate::duo::{self, Game, PASS};
use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

const TABLE_BITS: usize = 20;

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    value: i16,
    bound: Bound,
    best: u16,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Bound {
    #[default]
    Empty,
    Exact,
    Lower,
    Upper,
}

struct Keys {
    cells: [[u64; duo::CELLS]; 2],
    pieces: [[u64; duo::PIECES]; 2],
    bonus: [u64; 2],
    turn: u64,
    pass: u64,
}

fn keys() -> &'static Keys {
    static KEYS: OnceLock<Keys> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut state = 0x5eed_b10c_u64;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Keys {
            cells: std::array::from_fn(|_| std::array::from_fn(|_| next())),
            pieces: std::array::from_fn(|_| std::array::from_fn(|_| next())),
            bonus: [next(), next()],
            turn: next(),
            pass: next(),
        }
    })
}

/// Everything that affects the rest of the game and its score.
fn hash(game: &Game) -> u64 {
    let k = keys();
    let mut h = 0;
    for player in 0..2 {
        for cell in game.owned_cells(player).cells() {
            h ^= k.cells[player][cell];
        }
        let remaining = game.remaining_mask(player);
        for piece in 0..duo::PIECES {
            if remaining & (1 << piece) != 0 {
                h ^= k.pieces[player][piece];
            }
        }
        if remaining == 0 && game.last_piece(player) == Some(0) {
            h ^= k.bonus[player];
        }
    }
    if game.turn() == 1 {
        h ^= k.turn;
    }
    if game.consecutive_passes() > 0 {
        h ^= k.pass;
    }
    h
}

/// Transposition table of proven bounds. Each self-play game owns one, so
/// consecutive solves in that game reuse earlier proofs.
pub struct SolverTable {
    entries: Vec<Entry>,
    mask: usize,
}

impl SolverTable {
    pub fn new(bits: usize) -> Self {
        Self {
            entries: vec![Entry::default(); 1 << bits],
            mask: (1 << bits) - 1,
        }
    }
}

thread_local! {
    static TABLE: RefCell<SolverTable> = RefCell::new(SolverTable::new(TABLE_BITS));
}

struct Abort;

struct Search<'a> {
    table: &'a mut SolverTable,
    nodes: u64,
    budget: u64,
}

fn margin(game: &Game) -> i32 {
    let scores = game.scores();
    let me = game.turn();
    i32::from(scores[me]) - i32::from(scores[1 - me])
}

fn piece_size(action: usize) -> usize {
    duo::placement(action).map_or(0, |p| p.cells.len())
}

impl Search<'_> {
    fn negamax(&mut self, game: &Game, mut alpha: i32, beta: i32) -> Result<(i32, usize), Abort> {
        self.nodes += 1;
        if self.nodes > self.budget {
            return Err(Abort);
        }
        if game.is_terminal() {
            return Ok((margin(game), PASS));
        }
        let key = hash(game);
        let slot = (key as usize) & self.table.mask;
        let entry = self.table.entries[slot];
        let mut tt_move = None;
        if entry.bound != Bound::Empty && entry.key == key {
            let v = i32::from(entry.value);
            let best = usize::from(entry.best);
            match entry.bound {
                Bound::Exact => return Ok((v, best)),
                Bound::Lower if v >= beta => return Ok((v, best)),
                Bound::Upper if v <= alpha => return Ok((v, best)),
                _ => {}
            }
            tt_move = Some(best);
        }
        let mut moves = game.legal_actions();
        // Larger pieces first: they usually decide endgames, and good early
        // moves make alpha-beta cut far more of the tree.
        moves.sort_by_key(|&a| std::cmp::Reverse((Some(a) == tt_move, piece_size(a))));
        let original_alpha = alpha;
        let mut best = (i32::MIN, moves[0]);
        for &action in &moves {
            let mut child = game.clone();
            child.apply_trusted(action);
            let (v, _) = self.negamax(&child, -beta, -alpha)?;
            let v = -v;
            if v > best.0 {
                best = (v, action);
            }
            alpha = alpha.max(v);
            if alpha >= beta {
                break;
            }
        }
        let bound = if best.0 <= original_alpha {
            Bound::Upper
        } else if best.0 >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.table.entries[slot] = Entry {
            key,
            value: best.0 as i16,
            bound,
            best: best.1 as u16,
        };
        Ok(best)
    }
}

/// Exact margin for the player to move and a best action, or None if the
/// position is not proved within `budget` nodes. Searched nodes are added to `nodes`.
pub fn solve(
    game: &Game,
    budget: u64,
    table: &mut SolverTable,
    nodes: &AtomicU64,
) -> Option<(i32, usize)> {
    if game.is_terminal() || budget == 0 {
        return None;
    }
    let mut search = Search {
        table,
        nodes: 0,
        budget,
    };
    let result = search.negamax(game, -1000, 1000);
    nodes.fetch_add(search.nodes, Ordering::Relaxed);
    result.ok()
}

/// `solve` with this thread's shared table (for one-off calls).
pub fn solve_with_budget(game: &Game, budget: u64, nodes: &AtomicU64) -> Option<(i32, usize)> {
    TABLE.with(|table| solve(game, budget, &mut table.borrow_mut(), nodes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duo_engine::Rng;

    /// Plain minimax without pruning or a table: the reference answer.
    fn minimax(game: &Game) -> i32 {
        if game.is_terminal() {
            return margin(game);
        }
        game.legal_actions()
            .iter()
            .map(|&a| {
                let mut child = game.clone();
                child.apply_trusted(a);
                -minimax(&child)
            })
            .max()
            .unwrap()
    }

    fn tree_size(game: &Game, limit: u64) -> u64 {
        if game.is_terminal() {
            return 1;
        }
        let mut total = 1;
        for a in game.legal_actions() {
            let mut child = game.clone();
            child.apply_trusted(a);
            total += tree_size(&child, limit);
            if total > limit {
                return total;
            }
        }
        total
    }

    /// Random late positions whose full trees are small enough to enumerate.
    fn late_positions(count: usize, seed: u64) -> Vec<Game> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::new();
        while out.len() < count {
            let mut game = Game::new();
            let mut history = Vec::new();
            while !game.is_terminal() {
                history.push(game.clone());
                let actions = game.legal_actions();
                // Prefer big pieces like a real player so endgames look realistic.
                let pick = if rng.uniform() < 0.7 {
                    *actions
                        .iter()
                        .max_by_key(|&&a| (piece_size(a), rng.next_u64() % 8))
                        .unwrap()
                } else {
                    actions[rng.below(actions.len())]
                };
                game.apply_trusted(pick);
            }
            // Walk back from the end while the full tree stays enumerable,
            // then keep the three hardest (earliest) such positions.
            let mut small = Vec::new();
            for g in history.iter().rev() {
                if tree_size(g, 20_000) > 20_000 {
                    break;
                }
                small.push(g.clone());
            }
            let small: Vec<_> = small.into_iter().rev().take(3).collect();
            out.extend(small);
        }
        out
    }

    #[test]
    fn matches_plain_minimax_on_small_endgames() {
        let counter = AtomicU64::new(0);
        let positions = late_positions(60, 21);
        assert!(positions.len() >= 60);
        for game in positions {
            let exact = minimax(&game);
            let (value, action) = solve_with_budget(&game, 10_000_000, &counter).unwrap();
            assert_eq!(value, exact);
            // The returned move achieves the value.
            let mut child = game.clone();
            child.apply_trusted(action);
            let child_value = if child.is_terminal() {
                -margin(&child)
            } else {
                -minimax(&child)
            };
            assert_eq!(child_value, exact);
        }
    }

    #[test]
    fn budget_exhaustion_returns_none() {
        let counter = AtomicU64::new(0);
        assert!(solve_with_budget(&Game::new(), 1000, &counter).is_none());
        assert!(counter.load(Ordering::Relaxed) > 1000);
    }
}
