//! Blokus Duo rules. No policy, rendering toolkit, or training dependency.
//!
//! A move is a stable integer identifying an oriented piece and its top-left
//! board square. The first player chooses either start; the second uses the other.
//! Pass is legal only when no placement is possible. Two passes end the game.
use std::collections::BTreeSet;
use std::fmt;
use std::sync::OnceLock;

pub const SIZE: usize = 14;
pub const CELLS: usize = SIZE * SIZE;
pub const PIECES: usize = 21;
pub const ORIENTATIONS: usize = 91;
pub const PASS: usize = ORIENTATIONS * CELLS;
pub const ACTION_COUNT: usize = PASS + 1;
pub const STARTS: [usize; 2] = [4 * SIZE + 4, 9 * SIZE + 9];
const ALL_PIECES: u32 = (1 << PIECES) - 1;
type Shape = Vec<(i8, i8)>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bits(pub [u64; 4]);

impl Bits {
    pub fn insert(&mut self, cell: usize) {
        self.0[cell / 64] |= 1 << (cell % 64);
    }
    pub fn contains(self, cell: usize) -> bool {
        self.0[cell / 64] & (1 << (cell % 64)) != 0
    }
    pub fn intersects(self, other: Self) -> bool {
        (0..4).any(|i| self.0[i] & other.0[i] != 0)
    }
    pub fn union(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] | other.0[i]))
    }
    pub fn minus(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] & !other.0[i]))
    }
    pub fn count(self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
    pub fn cells(self) -> impl Iterator<Item = usize> {
        (0..4).flat_map(move |word| {
            let mut bits = self.0[word];
            std::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let cell = word * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    cell
                })
            })
        })
    }
}

#[derive(Clone, Debug)]
pub struct Orientation {
    pub piece: usize,
    /// Sorted (row, column) offsets, normalized to a top-left bounding box.
    pub cells: Shape,
}

#[derive(Clone, Debug)]
pub struct Placement {
    pub action: usize,
    pub piece: usize,
    pub orientation: usize,
    pub row: usize,
    pub col: usize,
    pub cells: Vec<usize>,
    pub mask: Bits,
}

struct Catalog {
    pieces: Vec<Shape>,
    orientations: Vec<Orientation>,
    placements: Vec<Option<Placement>>,
    /// Placements covering each cell, stored contiguously for move generation.
    compact: Vec<Vec<Compact>>,
}

#[derive(Clone, Copy, Debug)]
struct Compact {
    mask: Bits,
    action: u16,
    piece: u8,
}

fn normalize(mut cells: Shape) -> Shape {
    let row = cells.iter().map(|c| c.0).min().unwrap();
    let col = cells.iter().map(|c| c.1).min().unwrap();
    for cell in &mut cells {
        cell.0 -= row;
        cell.1 -= col;
    }
    cells.sort_unstable();
    cells
}

fn transforms(cells: &Shape) -> BTreeSet<Shape> {
    let mut result = BTreeSet::new();
    for reflect in [false, true] {
        let mut shape: Shape = cells
            .iter()
            .map(|&(r, c)| (r, if reflect { -c } else { c }))
            .collect();
        for _ in 0..4 {
            result.insert(normalize(shape.clone()));
            shape = shape.iter().map(|&(r, c)| (c, -r)).collect();
        }
    }
    result
}

fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        // Grow every free polyomino up to area five. Canonicalization under the
        // eight square symmetries removes rotations/reflections of the same piece.
        let mut level = BTreeSet::from([vec![(0, 0)]]);
        let mut pieces = Vec::new();
        for area in 1..=5 {
            pieces.extend(level.iter().cloned());
            if area == 5 {
                break;
            }
            let mut next = BTreeSet::new();
            for shape in &level {
                for &(r, c) in shape {
                    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let cell = (r + dr, c + dc);
                        if !shape.contains(&cell) {
                            let mut grown = shape.clone();
                            grown.push(cell);
                            next.insert(transforms(&grown).into_iter().next().unwrap());
                        }
                    }
                }
            }
            level = next;
        }
        assert_eq!(pieces.len(), PIECES);
        let mut orientations = Vec::new();
        for (piece, shape) in pieces.iter().enumerate() {
            for cells in transforms(shape) {
                orientations.push(Orientation { piece, cells });
            }
        }
        assert_eq!(orientations.len(), ORIENTATIONS);
        let mut placements = vec![None; PASS];
        for (orientation, shape) in orientations.iter().enumerate() {
            for row in 0..SIZE {
                for col in 0..SIZE {
                    if shape
                        .cells
                        .iter()
                        .any(|&(r, c)| row + r as usize >= SIZE || col + c as usize >= SIZE)
                    {
                        continue;
                    }
                    let cells: Vec<_> = shape
                        .cells
                        .iter()
                        .map(|&(r, c)| (row + r as usize) * SIZE + col + c as usize)
                        .collect();
                    let mut mask = Bits::default();
                    for &cell in &cells {
                        mask.insert(cell);
                    }
                    let action = orientation * CELLS + row * SIZE + col;
                    placements[action] = Some(Placement {
                        action,
                        piece: shape.piece,
                        orientation,
                        row,
                        col,
                        cells,
                        mask,
                    });
                }
            }
        }
        let mut by_cell = vec![Vec::new(); CELLS];
        for p in placements.iter().flatten() {
            for &cell in &p.cells {
                by_cell[cell].push(p.action);
            }
        }
        let compact = by_cell
            .iter()
            .map(|actions| {
                actions
                    .iter()
                    .map(|&a| {
                        let p = placements[a].as_ref().unwrap();
                        Compact {
                            mask: p.mask,
                            action: a as u16,
                            piece: p.piece as u8,
                        }
                    })
                    .collect()
            })
            .collect();
        Catalog {
            pieces,
            orientations,
            placements,
            compact,
        }
    })
}

pub fn pieces() -> &'static [Shape] {
    &catalog().pieces
}
pub fn orientations() -> &'static [Orientation] {
    &catalog().orientations
}
pub fn placement(action: usize) -> Option<&'static Placement> {
    catalog().placements.get(action)?.as_ref()
}

/// A malformed/illegal move never changes the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveError {
    GameOver,
    IllegalPlacement,
    MustPlace,
}
impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::GameOver => "the game is over",
            Self::IllegalPlacement => "illegal placement",
            Self::MustPlace => "passing is allowed only when no piece fits",
        })
    }
}
impl std::error::Error for MoveError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    board: [u8; CELLS],
    occupied: Bits,
    owned: [Bits; 2],
    edges: [Bits; 2],
    corners: [Bits; 2],
    remaining: [u32; 2],
    last_piece: [Option<usize>; 2],
    first_start: Option<usize>,
    turn: usize,
    passes: u8,
    terminal: bool,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        Self {
            board: [0; CELLS],
            occupied: Bits::default(),
            owned: [Bits::default(); 2],
            edges: [Bits::default(); 2],
            corners: [Bits::default(); 2],
            remaining: [ALL_PIECES; 2],
            last_piece: [None; 2],
            first_start: None,
            turn: 0,
            passes: 0,
            terminal: false,
        }
    }
    pub fn board(&self) -> &[u8; CELLS] {
        &self.board
    }
    pub fn current_player(&self) -> Option<usize> {
        (!self.terminal).then_some(self.turn)
    }
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
    pub fn consecutive_passes(&self) -> u8 {
        self.passes
    }
    /// Empty corner anchors excluding own edge contacts. A remaining piece
    /// need not fit at every anchor; this is geometry rather than a move count.
    pub fn frontier_counts(&self) -> [usize; 2] {
        std::array::from_fn(|player| {
            if self.last_piece[player].is_none() {
                let starts = self
                    .first_start
                    .map_or_else(|| STARTS.to_vec(), |first| vec![STARTS[1 - first]]);
                starts.iter().filter(|&&cell| self.board[cell] == 0).count()
            } else {
                (0..4)
                    .map(|i| {
                        (self.corners[player].0[i]
                            & !(self.edges[player].0[i] | self.occupied.0[i]))
                            .count_ones() as usize
                    })
                    .sum()
            }
        })
    }
    pub fn remaining(&self, player: usize) -> [bool; PIECES] {
        std::array::from_fn(|piece| self.remaining[player] & (1 << piece) != 0)
    }
    pub fn required_starts(&self) -> Vec<usize> {
        if self.last_piece[self.turn].is_some() || self.terminal {
            vec![]
        } else if let Some(first) = self.first_start {
            vec![STARTS[1 - first]]
        } else {
            STARTS.to_vec()
        }
    }
    fn legal_placement(&self, p: &Placement) -> bool {
        self.legal_placement_for(self.turn, p)
    }
    fn legal_placement_for(&self, player: usize, p: &Placement) -> bool {
        if self.remaining[player] & (1 << p.piece) == 0
            || p.mask.intersects(self.occupied)
            || p.mask.intersects(self.edges[player])
        {
            return false;
        }
        if self.last_piece[player].is_some() {
            p.mask.intersects(self.corners[player])
        } else {
            self.start_cells_for(player)
                .iter()
                .any(|&cell| p.mask.contains(cell))
        }
    }
    /// Start squares `player` could still use for a first piece (empty list after it).
    fn start_cells_for(&self, player: usize) -> Vec<usize> {
        if self.last_piece[player].is_some() {
            vec![]
        } else if let Some(first) = self.first_start {
            // The first player claimed `first`; whoever has not placed takes the other.
            vec![STARTS[1 - first]]
        } else {
            STARTS.to_vec()
        }
    }
    /// Empty cells where `player` could attach a new piece by a corner. Before the
    /// first piece these are the usable start squares. Rules-derived, no heuristics.
    pub fn attach_cells(&self, player: usize) -> Bits {
        if self.last_piece[player].is_none() {
            let mut bits = Bits::default();
            for cell in self.start_cells_for(player) {
                if !self.occupied.contains(cell) {
                    bits.insert(cell);
                }
            }
            bits
        } else {
            self.corners[player].minus(self.edges[player].union(self.occupied))
        }
    }
    /// Empty cells `player` may never fill because they touch its own pieces by an edge.
    pub fn forbidden_cells(&self, player: usize) -> Bits {
        self.edges[player].minus(self.occupied)
    }
    pub fn owned_cells(&self, player: usize) -> Bits {
        self.owned[player]
    }
    pub fn remaining_mask(&self, player: usize) -> u32 {
        self.remaining[player]
    }
    pub fn last_piece(&self, player: usize) -> Option<usize> {
        self.last_piece[player]
    }
    /// Index into STARTS of the square the first player used, once chosen.
    pub fn first_start(&self) -> Option<usize> {
        self.first_start
    }
    /// The player whose turn it is, even after termination (for search bookkeeping).
    pub fn turn(&self) -> usize {
        self.turn
    }
    pub fn legal_actions_scan(&self) -> Vec<usize> {
        if self.terminal {
            return vec![];
        }
        let mut actions: Vec<_> = catalog()
            .placements
            .iter()
            .flatten()
            .filter(|p| self.legal_placement(p))
            .map(|p| p.action)
            .collect();
        if actions.is_empty() {
            actions.push(PASS);
        }
        actions
    }

    pub fn legal_actions(&self) -> Vec<usize> {
        if self.terminal {
            return vec![];
        }
        self.legal_actions_for(self.turn)
    }

    /// Placements `player` could make if it were to move now; `[PASS]` when none.
    /// Used for the opponent's mobility features; it does not change whose turn it is.
    pub fn legal_actions_for(&self, player: usize) -> Vec<usize> {
        let mut legal = [0_u64; PASS.div_ceil(64)];
        let mut any = false;
        self.scan_for(player, |c| {
            legal[usize::from(c.action) / 64] |= 1 << (c.action % 64);
            any = true;
        });
        if !any {
            return vec![PASS];
        }
        // Reading the bitset in order yields the canonical sorted action list.
        let mut actions = Vec::with_capacity(256);
        for (word, &bits) in legal.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                actions.push(word * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        actions
    }

    /// Union of cells covered by `player`'s legal placements, and their count.
    pub fn reach_for(&self, player: usize) -> (Bits, usize) {
        let mut cells = Bits::default();
        let mut count = 0;
        self.scan_for(player, |c| {
            cells = cells.union(c.mask);
            count += 1;
        });
        (cells, count)
    }

    /// Visit each legal placement of `player` once. Every candidate covers an
    /// attach cell (or a usable start square), so it is legal exactly when its
    /// piece remains and it avoids occupied cells and the player's own edges.
    fn scan_for(&self, player: usize, mut visit: impl FnMut(&Compact)) {
        let anchors = self.attach_cells(player);
        let blocked = self.occupied.union(self.edges[player]);
        let remaining = self.remaining[player];
        let mut seen = [0_u64; PASS.div_ceil(64)];
        for cell in anchors.cells() {
            for c in &catalog().compact[cell] {
                if remaining & (1 << c.piece) == 0 {
                    continue;
                }
                let (word, bit) = (usize::from(c.action) / 64, 1_u64 << (c.action % 64));
                if seen[word] & bit != 0 {
                    continue;
                }
                seen[word] |= bit;
                if !c.mask.intersects(blocked) {
                    visit(c);
                }
            }
        }
    }

    /// The score is meaningful at any point, but determines the winner only at termination.
    pub fn scores(&self) -> [i16; 2] {
        std::array::from_fn(|player| {
            if self.remaining[player] == 0 {
                if self.last_piece[player] == Some(0) {
                    20
                } else {
                    15
                }
            } else {
                -pieces()
                    .iter()
                    .enumerate()
                    .filter(|(piece, _)| self.remaining[player] & (1 << piece) != 0)
                    .map(|(_, shape)| shape.len() as i16)
                    .sum::<i16>()
            }
        })
    }
    /// One turn, including a forced pass. Rewards are terminal win/draw/loss.
    pub fn step(&mut self, action: usize) -> Result<[f32; 2], MoveError> {
        if self.terminal {
            return Err(MoveError::GameOver);
        }
        if action == PASS {
            if self.legal_actions().first() != Some(&PASS) {
                return Err(MoveError::MustPlace);
            }
        } else if !placement(action).is_some_and(|p| self.legal_placement(p)) {
            return Err(MoveError::IllegalPlacement);
        }
        Ok(self.apply_trusted(action))
    }

    /// Apply an action already taken from this position's own legal list.
    /// Search replays thousands of known-legal paths; re-validating each would
    /// regenerate the move list for every pass.
    pub fn apply_trusted(&mut self, action: usize) -> [f32; 2] {
        debug_assert!(!self.terminal);
        if action == PASS {
            self.passes += 1;
            self.terminal = self.passes == 2;
        } else {
            let p = placement(action).expect("trusted placement");
            if self.first_start.is_none() {
                self.first_start = STARTS.iter().position(|&cell| p.mask.contains(cell));
            }
            let player = self.turn;
            for &cell in &p.cells {
                self.board[cell] = player as u8 + 1;
                self.occupied.insert(cell);
                self.owned[player].insert(cell);
                let (r, c) = (cell / SIZE, cell % SIZE);
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    if let Some(n) = neighbor(r, c, dr, dc) {
                        self.edges[player].insert(n);
                    }
                }
                for (dr, dc) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    if let Some(n) = neighbor(r, c, dr, dc) {
                        self.corners[player].insert(n);
                    }
                }
            }
            self.remaining[player] &= !(1 << p.piece);
            self.last_piece[player] = Some(p.piece);
            self.passes = 0;
        }
        self.turn = 1 - self.turn;
        if self.terminal {
            let scores = self.scores();
            let reward = (scores[0] - scores[1]).signum() as f32;
            [reward, -reward]
        } else {
            [0.0, 0.0]
        }
    }
}

fn neighbor(row: usize, col: usize, dr: isize, dc: isize) -> Option<usize> {
    let r = row.checked_add_signed(dr)?;
    let c = col.checked_add_signed(dc)?;
    (r < SIZE && c < SIZE).then_some(r * SIZE + c)
}

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    A B C D E F G H I J K L M N")?;
        for row in 0..SIZE {
            write!(f, "{:>2}  ", row + 1)?;
            for col in 0..SIZE {
                let cell = row * SIZE + col;
                let ch = match self.board[cell] {
                    1 => 'X',
                    2 => 'O',
                    _ if STARTS.contains(&cell) => '*',
                    _ => '.',
                };
                write!(f, "{ch} ")?;
            }
            writeln!(f)?;
        }
        write!(f, "Scores: X={} O={}", self.scores()[0], self.scores()[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexed_actions_match_scan_through_random_games() {
        let mut seed = 1070_u64;
        let mut passes = 0;
        for _ in 0..64 {
            let mut game = Game::new();
            while !game.is_terminal() {
                let actions = game.legal_actions();
                assert_eq!(actions, game.legal_actions_scan());
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let action = actions[(seed >> 32) as usize % actions.len()];
                if action == PASS {
                    passes += 1;
                }
                game.step(action).unwrap();
            }
            assert_eq!(game.legal_actions(), game.legal_actions_scan());
        }
        assert!(passes >= 128);
    }
    #[test]
    fn completion_bonuses() {
        let mut game = Game::new();
        game.remaining = [0, 0];
        game.last_piece = [Some(0), Some(7)];
        assert_eq!(game.scores(), [20, 15]);
    }
}
