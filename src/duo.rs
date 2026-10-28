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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Bits([u64; 4]);

impl Bits {
    fn insert(&mut self, cell: usize) {
        self.0[cell / 64] |= 1 << (cell % 64);
    }
    fn contains(self, cell: usize) -> bool {
        self.0[cell / 64] & (1 << (cell % 64)) != 0
    }
    fn intersects(self, other: Self) -> bool {
        (0..4).any(|i| self.0[i] & other.0[i] != 0)
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
    mask: Bits,
}

struct Catalog {
    pieces: Vec<Shape>,
    orientations: Vec<Orientation>,
    placements: Vec<Option<Placement>>,
    by_cell: Vec<Vec<usize>>,
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
        Catalog {
            pieces,
            orientations,
            placements,
            by_cell,
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
        if self.remaining[self.turn] & (1 << p.piece) == 0
            || p.mask.intersects(self.occupied)
            || p.mask.intersects(self.edges[self.turn])
        {
            return false;
        }
        if self.last_piece[self.turn].is_some() {
            p.mask.intersects(self.corners[self.turn])
        } else if let Some(first) = self.first_start {
            p.mask.contains(STARTS[1 - first])
        } else {
            STARTS.iter().any(|&cell| p.mask.contains(cell))
        }
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
        let mut anchors = Bits::default();
        if self.last_piece[self.turn].is_some() {
            for i in 0..4 {
                anchors.0[i] = self.corners[self.turn].0[i]
                    & !(self.edges[self.turn].0[i] | self.occupied.0[i]);
            }
        } else {
            for cell in self.required_starts() {
                anchors.insert(cell);
            }
        }
        let mut seen = [0_u64; PASS.div_ceil(64)];
        let mut actions = Vec::new();
        for word in 0..4 {
            let mut bits = anchors.0[word];
            while bits != 0 {
                let cell = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                for &action in &catalog().by_cell[cell] {
                    let bit = 1_u64 << (action % 64);
                    if seen[action / 64] & bit != 0 {
                        continue;
                    }
                    seen[action / 64] |= bit;
                    let placement = catalog().placements[action].as_ref().unwrap();
                    if self.legal_placement(placement) {
                        actions.push(action);
                    }
                }
            }
        }
        actions.sort_unstable(); // Preserve the canonical ordering and tie breaks.
        if actions.is_empty() {
            actions.push(PASS);
        }
        actions
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
            self.passes += 1;
            self.terminal = self.passes == 2;
        } else {
            let p = placement(action)
                .filter(|p| self.legal_placement(p))
                .ok_or(MoveError::IllegalPlacement)?;
            if self.first_start.is_none() {
                self.first_start = STARTS.iter().position(|&cell| p.mask.contains(cell));
            }
            let player = self.turn;
            for &cell in &p.cells {
                self.board[cell] = player as u8 + 1;
                self.occupied.insert(cell);
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
            Ok([reward, -reward])
        } else {
            Ok([0.0, 0.0])
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
