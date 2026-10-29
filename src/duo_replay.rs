//! Compact training replay: games are stored as move lists plus the searched
//! visit counts of recorded positions. Features are rebuilt at sampling time,
//! under a random board symmetry, so a checkpoint holds a few bytes per move
//! instead of encoded tensors.
use crate::duo::{CELLS, Game, PASS};
use crate::duo_engine::{FinishedGame, PositionRecord, Rng};
use crate::duo_v2::{self, PLANES, SCALARS, SYMMETRIES};
use rayon::prelude::*;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct StoredGame {
    pub moves: Vec<u16>,
    pub scores: [i16; 2],
    pub records: Vec<PositionRecord>,
    final_board: [u8; CELLS],
}

impl StoredGame {
    pub fn new(moves: Vec<u16>, records: Vec<PositionRecord>) -> Result<Self, String> {
        let mut game = Game::new();
        for &m in &moves {
            game.step(usize::from(m))
                .map_err(|e| format!("invalid stored move: {e}"))?;
        }
        if !game.is_terminal() {
            return Err("stored game is unfinished".into());
        }
        for r in &records {
            if usize::from(r.ply) >= moves.len() || r.visits.is_empty() {
                return Err("record outside game or without visits".into());
            }
        }
        Ok(Self {
            moves,
            scores: game.scores(),
            records,
            final_board: *game.board(),
        })
    }
}

#[derive(Default, Debug)]
pub struct Sample {
    pub planes: Vec<u8>,
    pub scalars: Vec<f32>,
    pub legal: Vec<u16>,
    pub counts: Vec<u32>,
    pub policy: Vec<f32>,
    /// 0 win, 1 draw, 2 loss for the player to move.
    pub outcome: Vec<u8>,
    pub margin: Vec<f32>,
    /// Per cell: 0 empty, 1 own, 2 opponent at the end of the game.
    pub ownership: Vec<u8>,
}

pub struct Replay {
    games: VecDeque<StoredGame>,
    /// (game id, record index), oldest first.
    positions: VecDeque<(u64, u32)>,
    first_game: u64,
    pub capacity: usize,
    pub added_games: u64,
    pub added_positions: u64,
}

impl Replay {
    pub fn new(capacity: usize) -> Self {
        Self {
            games: VecDeque::new(),
            positions: VecDeque::new(),
            first_game: 0,
            capacity,
            added_games: 0,
            added_positions: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
    pub fn games(&self) -> usize {
        self.games.len()
    }

    pub fn add(&mut self, game: StoredGame) {
        if game.records.is_empty() {
            return;
        }
        let id = self.first_game + self.games.len() as u64;
        for i in 0..game.records.len() {
            self.positions.push_back((id, i as u32));
        }
        self.added_positions += game.records.len() as u64;
        self.added_games += 1;
        self.games.push_back(game);
        while self.positions.len() > self.capacity {
            let (oldest, _) = self.positions[0];
            // Drop whole games so every stored record still has its game.
            while self.positions.front().is_some_and(|p| p.0 == oldest) {
                self.positions.pop_front();
            }
            while self.first_game <= oldest {
                self.games.pop_front();
                self.first_game += 1;
            }
        }
    }

    pub fn add_finished(&mut self, game: FinishedGame) -> Result<(), String> {
        self.add(StoredGame::new(game.moves, game.records)?);
        Ok(())
    }

    /// Imitation data: record every non-forced move from `from_ply` as a
    /// one-hot target for the move actually played.
    pub fn add_moves(&mut self, moves: &[usize], from_ply: usize) -> Result<(), String> {
        let mut game = Game::new();
        let mut records = Vec::new();
        for (ply, &m) in moves.iter().enumerate() {
            if ply >= from_ply && m != PASS {
                records.push(PositionRecord {
                    ply: ply as u16,
                    visits: vec![(m as u16, 1)],
                    root_value: 0.0,
                    solved: false,
                });
            }
            game.step(m).map_err(|e| format!("ply {ply}: {e}"))?;
        }
        let moves = moves.iter().map(|&m| m as u16).collect();
        self.add(StoredGame::new(moves, records)?);
        Ok(())
    }

    /// Sample `n` recorded positions uniformly from the newest `window`.
    pub fn sample(&self, n: usize, window: usize, seed: u64, augment: bool) -> Sample {
        assert!(!self.positions.is_empty(), "empty replay");
        let window = window.clamp(1, self.positions.len());
        let start = self.positions.len() - window;
        let mut rng = Rng::new(seed);
        let picks: Vec<_> = (0..n)
            .map(|_| {
                let position = self.positions[start + rng.below(window)];
                let sym = if augment { rng.below(SYMMETRIES) } else { 0 };
                (position, sym)
            })
            .collect();
        let items: Vec<Sample> = picks
            .par_iter()
            .map(|&((id, record), sym)| {
                let game = &self.games[(id - self.first_game) as usize];
                encode_record(game, &game.records[record as usize], sym)
            })
            .collect();
        let mut out = Sample::default();
        for item in items {
            out.planes.extend(item.planes);
            out.scalars.extend(item.scalars);
            out.legal.extend(item.legal);
            out.counts.extend(item.counts);
            out.policy.extend(item.policy);
            out.outcome.extend(item.outcome);
            out.margin.extend(item.margin);
            out.ownership.extend(item.ownership);
        }
        out
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"BKR2".to_vec();
        out.extend((self.capacity as u64).to_le_bytes());
        out.extend(self.added_games.to_le_bytes());
        out.extend(self.added_positions.to_le_bytes());
        out.extend((self.games.len() as u64).to_le_bytes());
        for g in &self.games {
            out.extend((g.moves.len() as u16).to_le_bytes());
            for m in &g.moves {
                out.extend(m.to_le_bytes());
            }
            out.extend((g.records.len() as u16).to_le_bytes());
            for r in &g.records {
                out.extend(r.ply.to_le_bytes());
                out.push(u8::from(r.solved));
                out.extend(r.root_value.to_le_bytes());
                out.extend((r.visits.len() as u16).to_le_bytes());
                for (a, v) in &r.visits {
                    out.extend(a.to_le_bytes());
                    out.extend(v.to_le_bytes());
                }
            }
        }
        out
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut r = Reader { data, at: 0 };
        if r.take(4)? != b"BKR2" {
            return Err("not a v2 replay".into());
        }
        let capacity = r.u64()? as usize;
        let added_games = r.u64()?;
        let added_positions = r.u64()?;
        let count = r.u64()?;
        let mut replay = Self::new(capacity.max(1));
        for _ in 0..count {
            let n = r.u16()?;
            let moves = (0..n).map(|_| r.u16()).collect::<Result<Vec<_>, _>>()?;
            let n = r.u16()?;
            let mut records = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let ply = r.u16()?;
                let solved = r.take(1)?[0] != 0;
                let root_value = f32::from_le_bytes(r.take(4)?.try_into().unwrap());
                let k = r.u16()?;
                let visits = (0..k)
                    .map(|_| Ok((r.u16()?, r.u16()?)))
                    .collect::<Result<Vec<_>, String>>()?;
                records.push(PositionRecord {
                    ply,
                    visits,
                    root_value,
                    solved,
                });
            }
            replay.add(StoredGame::new(moves, records)?);
        }
        replay.added_games = added_games;
        replay.added_positions = added_positions;
        Ok(replay)
    }
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let slice = self
            .data
            .get(self.at..self.at + n)
            .ok_or("truncated replay")?;
        self.at += n;
        Ok(slice)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn encode_record(stored: &StoredGame, record: &PositionRecord, sym: usize) -> Sample {
    let mut game = Game::new();
    for &m in &stored.moves[..usize::from(record.ply)] {
        game.apply_trusted(usize::from(m));
    }
    let me = game.turn();
    let actions = game.legal_actions();
    let mut planes = vec![0; PLANES * CELLS];
    let mut scalars = vec![0.0; SCALARS];
    let mut legal = Vec::new();
    duo_v2::encode(&game, &actions, sym, &mut planes, &mut scalars, &mut legal);
    let mut policy: Vec<f32> = actions
        .iter()
        .map(|&a| {
            record
                .visits
                .iter()
                .find(|v| usize::from(v.0) == a)
                .map_or(0.0, |v| f32::from(v.1))
        })
        .collect();
    let total: f32 = policy.iter().sum();
    assert!(total > 0.0, "record visits not among legal moves");
    policy.iter_mut().for_each(|p| *p /= total);
    let margin = i32::from(stored.scores[me]) - i32::from(stored.scores[1 - me]);
    let mut ownership = vec![0; CELLS];
    duo_v2::ownership(&stored.final_board, me, sym, &mut ownership);
    Sample {
        planes,
        scalars,
        counts: vec![legal.len() as u32],
        legal,
        policy,
        outcome: vec![match margin.signum() {
            1 => 0,
            0 => 1,
            _ => 2,
        }],
        margin: vec![margin as f32],
        ownership,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_game(seed: u64) -> Vec<usize> {
        let mut game = Game::new();
        let mut rng = Rng::new(seed);
        let mut moves = vec![];
        while !game.is_terminal() {
            let actions = game.legal_actions();
            let m = actions[rng.below(actions.len())];
            game.step(m).unwrap();
            moves.push(m);
        }
        moves
    }

    #[test]
    fn round_trip_eviction_and_sampling() {
        let mut replay = Replay::new(150);
        for seed in 0..10 {
            replay.add_moves(&random_game(seed), 0).unwrap();
        }
        assert!(replay.len() <= 150 && replay.len() > 50);
        let copy = Replay::from_bytes(&replay.to_bytes()).unwrap();
        assert_eq!(copy.len(), replay.len());
        let (a, b) = (
            replay.sample(64, 1000, 3, true),
            copy.sample(64, 1000, 3, true),
        );
        assert_eq!(a.planes, b.planes);
        assert_eq!(a.policy, b.policy);
        assert_eq!(a.counts.len(), 64);
        assert_eq!(a.policy.len(), a.legal.len());
        // One-hot imitation targets.
        let mut offset = 0;
        for &c in &a.counts {
            let row = &a.policy[offset..offset + c as usize];
            assert_eq!(row.iter().filter(|&&p| p == 1.0).count(), 1);
            offset += c as usize;
        }
        // Ownership agrees with the outcome sign: own cells minus opponent cells
        // relates to piece areas, so just check classes are valid.
        assert!(a.ownership.iter().all(|&o| o <= 2));
    }
}
