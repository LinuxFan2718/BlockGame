//! Many concurrent Blokus Duo games whose searches advance independently.
//!
//! Python only evaluates the network. `prepare(group)` advances every game in
//! one group (in parallel) until each is waiting on network evaluations, then
//! returns one batch. `apply(group, ...)` feeds results back. Two groups let the
//! GPU evaluate one group while the CPU searches the other.
//!
//! Each game moves as soon as its own search finishes, so full and fast
//! searches never wait for one another, and a finished game is immediately
//! replaced by a new one.
use crate::duo::{CELLS, Game, PASS};
use crate::duo_mcts::{Leaf, SearchParams, Selection, Tree};
use crate::duo_solver::SolverTable;
use crate::duo_v2::{self, PLANES, SCALARS};
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Small, fast deterministic generator (SplitMix64) with the few draws we need.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n.max(1)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Marsaglia–Tsang gamma sampler; shape < 1 uses the standard boost.
    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform().max(1e-300);
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
    pub fn dirichlet(&mut self, n: usize, alpha: f64) -> Vec<f32> {
        let draws: Vec<f64> = (0..n).map(|_| self.gamma(alpha)).collect();
        let sum: f64 = draws.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return vec![1.0 / n as f32; n];
        }
        draws.iter().map(|x| (x / sum) as f32).collect()
    }
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Which network evaluates this agent's searches (lets two networks play).
    pub net: u8,
    pub full_simulations: u32,
    pub fast_simulations: u32,
    /// Probability that a move gets a full (recorded, noised) search.
    pub full_probability: f32,
    pub noise_weight: f32,
    /// Dirichlet alpha = noise_mass / legal move count.
    pub noise_mass: f32,
    /// Before this ply, moves are sampled in proportion to visits.
    pub temperature_plies: u32,
    pub record: bool,
    pub reuse_tree: bool,
    /// Moves come from Python (for example Pentobi) instead of search.
    pub external: bool,
    pub leaf_batch: u32,
    /// Exact solver node budgets; 0 disables. Root solves replace search entirely.
    pub solver_root_nodes: u64,
    pub solver_leaf_nodes: u64,
    /// Only attempt solves when both sides together have at most this many placements.
    pub solver_max_moves: u32,
    pub params: SearchParams,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            net: 0,
            full_simulations: 1024,
            fast_simulations: 128,
            full_probability: 0.25,
            noise_weight: 0.25,
            noise_mass: 10.0,
            temperature_plies: 12,
            record: true,
            reuse_tree: true,
            external: false,
            leaf_batch: 4,
            solver_root_nodes: 0,
            solver_leaf_nodes: 0,
            solver_max_moves: 0,
            params: SearchParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub agents: [AgentConfig; 2],
    /// Concurrent games (split across `groups`).
    pub slots: usize,
    pub groups: usize,
    /// Total games to start. Game k's agent 0 plays first when k is even.
    pub games: u64,
    pub alternate_seats: bool,
    /// Optional opening move lists; game k uses `openings[(k / 2) % len]` when
    /// alternating seats (each opening is played from both sides).
    pub openings: Vec<Vec<usize>>,
    /// Self-play: both seats share one search tree.
    pub shared_tree: bool,
    pub seed: u64,
}

/// One searched decision kept for training.
#[derive(Clone, Debug, Default)]
pub struct PositionRecord {
    pub ply: u16,
    /// (action, visits) for every visited root move, in action order.
    pub visits: Vec<(u16, u16)>,
    /// Root value for the mover after search.
    pub root_value: f32,
    /// The move was chosen by the exact solver.
    pub solved: bool,
}

#[derive(Clone, Debug)]
pub struct FinishedGame {
    pub index: u64,
    pub moves: Vec<u16>,
    pub scores: [i16; 2],
    /// Agent index playing as player 0 and player 1.
    pub seats: [usize; 2],
    pub opening_plies: u16,
    pub records: Vec<PositionRecord>,
}

#[derive(Default)]
pub struct Counters {
    pub simulations: AtomicU64,
    pub evaluations: AtomicU64,
    pub exact_backups: AtomicU64,
    pub collisions: AtomicU64,
    pub moves: AtomicU64,
    pub solver_moves: AtomicU64,
    pub solver_leaf_hits: AtomicU64,
    pub solver_nodes: AtomicU64,
}

struct Slot {
    active: bool,
    index: u64,
    game: Game,
    ply: u32,
    moves: Vec<u16>,
    opening: Vec<usize>,
    opening_plies: u16,
    seats: [usize; 2],
    trees: [Option<Tree>; 2],
    rng: Rng,
    searching: bool,
    full: bool,
    noise_pending: bool,
    target: u32,
    requested: u32,
    pending: Vec<Leaf>,
    pending_agent: usize,
    records: Vec<PositionRecord>,
    waiting_external: bool,
    table: Option<Box<SolverTable>>,
    // Encoded pending leaves.
    planes: Vec<u8>,
    scalars: Vec<f32>,
    legal: Vec<u16>,
    counts: Vec<u32>,
}

/// One batch of positions for the network, in slot order.
#[derive(Default, Debug)]
pub struct Batch {
    pub planes: Vec<u8>,
    pub scalars: Vec<f32>,
    pub legal: Vec<u16>,
    pub counts: Vec<u32>,
    pub nets: Vec<u8>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.counts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

pub type Solver = fn(&Game, u64, &mut SolverTable, &AtomicU64) -> Option<(i32, usize)>;

/// Per-game solver table size (2^16 entries, 1 MiB), allocated on first use.
const SLOT_TABLE_BITS: usize = 16;

/// Slots are partitioned into groups (slot i belongs to group i % groups).
/// Each group has its own lock, so one thread can search a group while another
/// applies network results to a different group.
pub struct Engine {
    pub config: EngineConfig,
    groups: Vec<Mutex<Vec<(usize, Slot)>>>,
    next_game: AtomicU64,
    finished: Mutex<Vec<FinishedGame>>,
    pub counters: Counters,
    solver: Option<Solver>,
}

impl Engine {
    pub fn new(config: EngineConfig, solver: Option<Solver>) -> Self {
        assert!(config.groups >= 1 && config.slots >= config.groups);
        let mut groups: Vec<Vec<(usize, Slot)>> = (0..config.groups).map(|_| vec![]).collect();
        for i in 0..config.slots {
            groups[i % config.groups].push((
                i,
                Slot {
                    active: false,
                    index: 0,
                    game: Game::new(),
                    ply: 0,
                    moves: vec![],
                    opening: vec![],
                    opening_plies: 0,
                    seats: [0, 1],
                    trees: [None, None],
                    rng: Rng::new(config.seed.wrapping_mul(1_000_003).wrapping_add(i as u64)),
                    searching: false,
                    full: false,
                    noise_pending: false,
                    target: 0,
                    requested: 0,
                    pending: vec![],
                    pending_agent: 0,
                    records: vec![],
                    waiting_external: false,
                    table: None,
                    planes: vec![],
                    scalars: vec![],
                    legal: vec![],
                    counts: vec![],
                },
            ));
        }
        Self {
            config,
            groups: groups.into_iter().map(Mutex::new).collect(),
            next_game: AtomicU64::new(0),
            finished: Mutex::new(vec![]),
            counters: Counters::default(),
            solver,
        }
    }

    pub fn games_started(&self) -> u64 {
        self.next_game
            .load(Ordering::Relaxed)
            .min(self.config.games)
    }
    pub fn active_games(&self) -> usize {
        self.groups
            .iter()
            .map(|g| g.lock().unwrap().iter().filter(|(_, s)| s.active).count())
            .sum()
    }
    /// All games have been started and finished.
    pub fn done(&self) -> bool {
        self.next_game.load(Ordering::Relaxed) >= self.config.games
            && self
                .groups
                .iter()
                .all(|g| g.lock().unwrap().iter().all(|(_, s)| !s.active))
    }
    pub fn drain_finished(&self) -> Vec<FinishedGame> {
        std::mem::take(&mut *self.finished.lock().unwrap())
    }

    /// Advance one group until each game waits on the network, an external
    /// move, or has nothing left to play. Returns that group's batch.
    pub fn prepare(&self, group: usize) -> Batch {
        let ctx = Context {
            config: &self.config,
            next_game: &self.next_game,
            finished: &self.finished,
            counters: &self.counters,
            solver: self.solver,
        };
        let mut slots = self.groups[group].lock().unwrap();
        slots.par_iter_mut().for_each(|(_, slot)| ctx.advance(slot));
        let mut batch = Batch::default();
        for (_, slot) in slots.iter().filter(|(_, s)| !s.pending.is_empty()) {
            batch.planes.extend_from_slice(&slot.planes);
            batch.scalars.extend_from_slice(&slot.scalars);
            batch.legal.extend_from_slice(&slot.legal);
            batch.counts.extend_from_slice(&slot.counts);
            let net = self.config.agents[slot.pending_agent].net;
            batch
                .nets
                .extend(std::iter::repeat_n(net, slot.pending.len()));
        }
        self.counters
            .evaluations
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        batch
    }

    /// Network results for the batch `prepare(group)` returned: probabilities
    /// over each position's legal moves (concatenated) and values for the mover.
    pub fn apply(&self, group: usize, probabilities: &[f32], values: &[f32]) {
        let mut slots = self.groups[group].lock().unwrap();
        let mut work = Vec::new();
        let (mut p_offset, mut v_offset) = (0, 0);
        for (_, slot) in slots.iter_mut().filter(|(_, s)| !s.pending.is_empty()) {
            let legal: usize = slot.counts.iter().map(|&c| c as usize).sum();
            let n = slot.pending.len();
            work.push((slot, p_offset, v_offset));
            p_offset += legal;
            v_offset += n;
        }
        assert_eq!(p_offset, probabilities.len(), "probability count mismatch");
        assert_eq!(v_offset, values.len(), "value count mismatch");
        let config = &self.config;
        work.into_par_iter().for_each(|(slot, p, v)| {
            let agent = &config.agents[slot.pending_agent];
            let tree_index = tree_index(config, slot.pending_agent);
            let tree = slot.trees[tree_index].as_mut().unwrap();
            let mut offset = p;
            let leaves = std::mem::take(&mut slot.pending);
            for (k, leaf) in leaves.iter().enumerate() {
                let n = leaf.actions.len();
                let priors = &probabilities[offset..offset + n];
                offset += n;
                tree.expand(leaf, priors, values[v + k]);
                if leaf.path.len() == 1 && slot.noise_pending {
                    let alpha = f64::from(agent.noise_mass) / n as f64;
                    let noise = slot.rng.dirichlet(n, alpha);
                    tree.mix_root_noise(&noise, agent.noise_weight);
                    slot.noise_pending = false;
                }
            }
        });
    }

    /// Games waiting for a move from Python: (slot, moves so far).
    pub fn waiting_external(&self) -> Vec<(usize, Vec<u16>)> {
        let mut out = Vec::new();
        for group in &self.groups {
            for (id, s) in group.lock().unwrap().iter() {
                if s.active && s.waiting_external {
                    out.push((*id, s.moves.clone()));
                }
            }
        }
        out.sort_unstable_by_key(|x| x.0);
        out
    }

    pub fn external_move(&self, slot: usize, action: usize) -> Result<(), String> {
        let group = self
            .groups
            .get(slot % self.config.groups)
            .ok_or("unknown slot")?;
        let mut slots = group.lock().unwrap();
        let s = slots
            .iter_mut()
            .find(|(id, _)| *id == slot)
            .map(|(_, s)| s)
            .ok_or("unknown slot")?;
        if !s.active || !s.waiting_external {
            return Err("slot is not waiting for an external move".into());
        }
        let legal = s.game.legal_actions();
        if legal.binary_search(&action).is_err() {
            return Err(format!("illegal external move {action}"));
        }
        s.waiting_external = false;
        play(s, action);
        Ok(())
    }
}

fn tree_index(config: &EngineConfig, agent: usize) -> usize {
    if config.shared_tree { 0 } else { agent }
}

struct Context<'a> {
    config: &'a EngineConfig,
    next_game: &'a AtomicU64,
    finished: &'a Mutex<Vec<FinishedGame>>,
    counters: &'a Counters,
    solver: Option<Solver>,
}

fn play(slot: &mut Slot, action: usize) {
    slot.game.apply_trusted(action);
    slot.moves.push(action as u16);
    slot.ply += 1;
    for tree in slot.trees.iter_mut().flatten() {
        tree.advance(action);
    }
}

fn small_enough(game: &Game, limit: u32) -> bool {
    if limit == 0 {
        return false;
    }
    let me = game.turn();
    let count = |player| {
        let actions = game.legal_actions_for(player);
        if actions == [PASS] { 0 } else { actions.len() }
    };
    let mine = count(me);
    mine <= limit as usize && mine + count(1 - me) <= limit as usize
}

impl Context<'_> {
    fn start_game(&self, slot: &mut Slot) -> bool {
        let index = self.next_game.fetch_add(1, Ordering::Relaxed);
        if index >= self.config.games {
            return false;
        }
        let swap = self.config.alternate_seats && index % 2 == 1;
        slot.active = true;
        slot.index = index;
        slot.game = Game::new();
        slot.ply = 0;
        slot.moves.clear();
        slot.records.clear();
        slot.seats = if swap { [1, 0] } else { [0, 1] };
        slot.trees = [None, None];
        slot.searching = false;
        slot.waiting_external = false;
        slot.opening = if self.config.openings.is_empty() {
            vec![]
        } else {
            let pair = if self.config.alternate_seats {
                index / 2
            } else {
                index
            };
            self.config.openings[(pair as usize) % self.config.openings.len()].clone()
        };
        slot.opening_plies = slot.opening.len() as u16;
        slot.rng = Rng::new(self.config.seed ^ index.wrapping_mul(0x2545_f491_4f6c_dd1d));
        true
    }

    fn finish_game(&self, slot: &mut Slot) {
        let game = FinishedGame {
            index: slot.index,
            moves: std::mem::take(&mut slot.moves),
            scores: slot.game.scores(),
            seats: slot.seats,
            opening_plies: slot.opening_plies,
            records: std::mem::take(&mut slot.records),
        };
        self.finished.lock().unwrap().push(game);
        slot.active = false;
        slot.trees = [None, None];
        slot.table = None;
    }

    fn advance(&self, slot: &mut Slot) {
        slot.planes.clear();
        slot.scalars.clear();
        slot.legal.clear();
        slot.counts.clear();
        loop {
            if !slot.active && !self.start_game(slot) {
                return;
            }
            if slot.game.is_terminal() {
                self.finish_game(slot);
                continue;
            }
            if (slot.ply as usize) < slot.opening.len() {
                let action = slot.opening[slot.ply as usize];
                play(slot, action);
                continue;
            }
            let player = slot.game.turn();
            let agent_index = slot.seats[player];
            let agent = &self.config.agents[agent_index];
            if !slot.searching {
                // External players answer every turn themselves, including
                // forced passes, so callers can stop after exactly one move.
                if agent.external {
                    slot.waiting_external = true;
                    return;
                }
                if slot.game.legal_actions() == [PASS] {
                    play(slot, PASS);
                    continue;
                }
                if self.try_root_solve(slot, agent) {
                    continue;
                }
                self.begin_search(slot, agent_index);
            }
            let agent = &self.config.agents[agent_index];
            let tree_index = tree_index(self.config, agent_index);
            if slot.requested >= slot.target {
                self.finish_search(slot, agent_index);
                continue;
            }
            let tree = slot.trees[tree_index].as_mut().unwrap();
            let solver = self.solver.filter(|_| agent.solver_leaf_nodes > 0);
            let counters = self.counters;
            let table = &mut slot.table;
            let mut leaf_solver = |game: &Game| {
                let s = solver?;
                if !small_enough(game, agent.solver_max_moves) {
                    return None;
                }
                let table =
                    table.get_or_insert_with(|| Box::new(SolverTable::new(SLOT_TABLE_BITS)));
                let result = s(game, agent.solver_leaf_nodes, table, &counters.solver_nodes)?;
                counters.solver_leaf_hits.fetch_add(1, Ordering::Relaxed);
                Some(result.0)
            };
            let mut exact = 0;
            while slot.pending.len() < agent.leaf_batch as usize && slot.requested < slot.target {
                match tree.select(&agent.params, &mut leaf_solver) {
                    Selection::Leaf(leaf) => {
                        // Expanding a fresh root is setup, not a simulation.
                        if leaf.path.len() > 1 {
                            slot.requested += 1;
                        }
                        slot.pending.push(leaf);
                    }
                    Selection::Exact => {
                        slot.requested += 1;
                        exact += 1;
                    }
                    Selection::Collision => {
                        self.counters.collisions.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
            }
            self.counters
                .exact_backups
                .fetch_add(exact, Ordering::Relaxed);
            if !slot.pending.is_empty() {
                slot.pending_agent = agent_index;
                self.encode_pending(slot);
                return;
            }
        }
    }

    fn try_root_solve(&self, slot: &mut Slot, agent: &AgentConfig) -> bool {
        let Some(solver) = self.solver else {
            return false;
        };
        if agent.solver_root_nodes == 0 || !small_enough(&slot.game, agent.solver_max_moves) {
            return false;
        }
        let table = slot
            .table
            .get_or_insert_with(|| Box::new(SolverTable::new(SLOT_TABLE_BITS)));
        let Some((margin, action)) = solver(
            &slot.game,
            agent.solver_root_nodes,
            table,
            &self.counters.solver_nodes,
        ) else {
            return false;
        };
        if agent.record {
            slot.records.push(PositionRecord {
                ply: slot.ply as u16,
                visits: vec![(action as u16, 1)],
                root_value: agent.params.utility(margin),
                solved: true,
            });
        }
        self.counters.solver_moves.fetch_add(1, Ordering::Relaxed);
        self.counters.moves.fetch_add(1, Ordering::Relaxed);
        play(slot, action);
        true
    }

    fn begin_search(&self, slot: &mut Slot, agent_index: usize) {
        let agent = &self.config.agents[agent_index];
        let index = tree_index(self.config, agent_index);
        let stale = slot.trees[index]
            .as_ref()
            .is_none_or(|t| !agent.reuse_tree || t.root_game() != &slot.game);
        if stale {
            slot.trees[index] = Some(Tree::new(slot.game.clone()));
        }
        slot.full =
            agent.full_probability >= 1.0 || (slot.rng.uniform() as f32) < agent.full_probability;
        slot.target = if slot.full {
            agent.full_simulations
        } else {
            agent.fast_simulations
        }
        .max(1);
        slot.requested = 0;
        slot.searching = true;
        slot.noise_pending = false;
        if slot.full && agent.noise_weight > 0.0 {
            let tree = slot.trees[index].as_mut().unwrap();
            if tree.root_expanded() {
                let n = tree.root_edge_count();
                let alpha = f64::from(agent.noise_mass) / n as f64;
                let noise = slot.rng.dirichlet(n, alpha);
                tree.mix_root_noise(&noise, agent.noise_weight);
            } else {
                slot.noise_pending = true;
            }
        }
    }

    fn finish_search(&self, slot: &mut Slot, agent_index: usize) {
        let agent = &self.config.agents[agent_index];
        let tree = slot.trees[tree_index(self.config, agent_index)]
            .as_ref()
            .unwrap();
        let children = tree.root_children();
        let total: u64 = children.iter().map(|c| u64::from(c.visits)).sum();
        let action = if total == 0 {
            // Root proven exact (e.g. solved) before any child was visited.
            children
                .iter()
                .max_by(|a, b| a.prior.total_cmp(&b.prior))
                .map_or_else(|| slot.game.legal_actions()[0], |c| c.action)
        } else if slot.ply < agent.temperature_plies {
            let mut pick = slot.rng.uniform() * total as f64;
            let mut chosen = children[0].action;
            for c in &children {
                if c.visits == 0 {
                    continue;
                }
                chosen = c.action;
                pick -= f64::from(c.visits);
                if pick < 0.0 {
                    break;
                }
            }
            chosen
        } else {
            children
                .iter()
                .max_by(|a, b| a.visits.cmp(&b.visits).then(a.prior.total_cmp(&b.prior)))
                .unwrap()
                .action
        };
        if slot.full && agent.record && total > 0 {
            slot.records.push(PositionRecord {
                ply: slot.ply as u16,
                visits: children
                    .iter()
                    .filter(|c| c.visits > 0)
                    .map(|c| (c.action as u16, c.visits.min(u32::from(u16::MAX)) as u16))
                    .collect(),
                root_value: tree.root_value(),
                solved: false,
            });
        }
        self.counters
            .simulations
            .fetch_add(u64::from(slot.requested), Ordering::Relaxed);
        self.counters.moves.fetch_add(1, Ordering::Relaxed);
        slot.searching = false;
        play(slot, action);
    }

    fn encode_pending(&self, slot: &mut Slot) {
        let n = slot.pending.len();
        slot.planes.resize(n * PLANES * CELLS, 0);
        slot.scalars.resize(n * SCALARS, 0.0);
        let mut legal = Vec::new();
        for (k, leaf) in slot.pending.iter().enumerate() {
            duo_v2::encode(
                &leaf.game,
                &leaf.actions,
                0,
                &mut slot.planes[k * PLANES * CELLS..(k + 1) * PLANES * CELLS],
                &mut slot.scalars[k * SCALARS..(k + 1) * SCALARS],
                &mut legal,
            );
            slot.legal.extend_from_slice(&legal);
            slot.counts.push(legal.len() as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_round(engine: &mut Engine, group: usize) -> usize {
        let batch = engine.prepare(group);
        let probabilities: Vec<f32> = batch
            .counts
            .iter()
            .flat_map(|&c| std::iter::repeat_n(1.0 / c as f32, c as usize))
            .collect();
        let values = vec![0.0; batch.len()];
        engine.apply(group, &probabilities, &values);
        batch.len()
    }

    #[test]
    fn self_play_finishes_every_game_with_records() {
        let mut agent = AgentConfig {
            full_simulations: 32,
            fast_simulations: 8,
            ..AgentConfig::default()
        };
        agent.temperature_plies = 4;
        let config = EngineConfig {
            agents: [agent.clone(), agent],
            slots: 8,
            groups: 2,
            games: 12,
            alternate_seats: false,
            openings: vec![],
            shared_tree: true,
            seed: 5,
        };
        let mut engine = Engine::new(config, None);
        let mut finished = Vec::new();
        while !engine.done() {
            uniform_round(&mut engine, 0);
            uniform_round(&mut engine, 1);
            finished.extend(engine.drain_finished());
        }
        assert_eq!(finished.len(), 12);
        for g in &finished {
            let mut game = Game::new();
            for &m in &g.moves {
                game.step(usize::from(m)).unwrap();
            }
            assert!(game.is_terminal());
            assert_eq!(game.scores(), g.scores);
            assert!(!g.records.is_empty());
            for r in &g.records {
                let visits: u32 = r.visits.iter().map(|v| u32::from(v.1)).sum();
                assert!(visits >= 32);
            }
        }
    }

    #[test]
    fn solver_moves_are_recorded_and_games_stay_valid() {
        let agent = AgentConfig {
            full_simulations: 16,
            fast_simulations: 4,
            solver_max_moves: 20,
            solver_root_nodes: 100_000,
            solver_leaf_nodes: 2_000,
            ..AgentConfig::default()
        };
        let config = EngineConfig {
            agents: [agent.clone(), agent],
            slots: 6,
            groups: 2,
            games: 6,
            alternate_seats: false,
            openings: vec![],
            shared_tree: true,
            seed: 13,
        };
        let mut engine = Engine::new(config, Some(crate::duo_solver::solve));
        let mut finished = Vec::new();
        while !engine.done() {
            uniform_round(&mut engine, 0);
            uniform_round(&mut engine, 1);
            finished.extend(engine.drain_finished());
        }
        assert!(engine.counters.solver_moves.load(Ordering::Relaxed) > 0);
        for g in &finished {
            let mut game = Game::new();
            for &m in &g.moves {
                game.step(usize::from(m)).unwrap();
            }
            assert_eq!(game.scores(), g.scores);
            assert!(g.records.iter().any(|r| r.solved));
        }
    }

    #[test]
    fn external_seats_wait_for_moves() {
        let agent = AgentConfig {
            full_simulations: 16,
            full_probability: 1.0,
            noise_weight: 0.0,
            temperature_plies: 0,
            record: false,
            ..AgentConfig::default()
        };
        let external = AgentConfig {
            external: true,
            ..agent.clone()
        };
        let config = EngineConfig {
            agents: [agent, external],
            slots: 4,
            groups: 1,
            games: 4,
            alternate_seats: true,
            openings: vec![],
            shared_tree: false,
            seed: 9,
        };
        let mut engine = Engine::new(config, None);
        let mut state = 1_u64;
        let mut finished = Vec::new();
        while !engine.done() {
            uniform_round(&mut engine, 0);
            for (slot, moves) in engine.waiting_external() {
                let mut game = Game::new();
                for m in moves {
                    game.step(usize::from(m)).unwrap();
                }
                let legal = game.legal_actions();
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                engine
                    .external_move(slot, legal[(state >> 33) as usize % legal.len()])
                    .unwrap();
            }
            finished.extend(engine.drain_finished());
        }
        assert_eq!(finished.len(), 4);
        assert!(finished.iter().any(|g| g.seats == [1, 0]));
    }
}
