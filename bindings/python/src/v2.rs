//! Python boundary for the v2 engine, replay and solver. Python only runs the
//! network; batches cross as byte buffers that NumPy views without copying.
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict};
use tic_tac_toe::duo;
use tic_tac_toe::duo_engine::{AgentConfig, Engine, EngineConfig, FinishedGame};
use tic_tac_toe::duo_mcts::SearchParams;
use tic_tac_toe::duo_replay::Replay;
use tic_tac_toe::duo_solver;
use tic_tac_toe::duo_v2;

fn bytes<T: Copy>(py: Python<'_>, values: &[T]) -> Py<PyByteArray> {
    // SAFETY: plain numeric slices; the receiver reads them as little-endian
    // arrays of the same type (all supported hosts are little-endian).
    let raw = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    PyByteArray::new(py, raw).unbind()
}

fn get<'py, T: FromPyObjectOwned<'py>>(
    d: &Bound<'py, PyDict>,
    key: &str,
    default: T,
) -> PyResult<T> {
    match d.get_item(key)? {
        Some(v) => v.extract().map_err(Into::into),
        None => Ok(default),
    }
}

fn agent(d: &Bound<'_, PyDict>) -> PyResult<AgentConfig> {
    let base = AgentConfig::default();
    let p = SearchParams::default();
    let known = [
        "net", "full_simulations", "fast_simulations", "full_probability", "noise_weight",
        "noise_mass", "temperature_plies", "record", "reuse_tree", "external", "leaf_batch",
        "solver_root_nodes", "solver_leaf_nodes", "solver_max_moves", "c_puct",
        "fpu_reduction", "fpu_root", "virtual_loss", "score_weight", "score_scale",
    ];
    for key in d.keys() {
        let key: String = key.extract()?;
        if !known.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!("unknown agent option {key}")));
        }
    }
    Ok(AgentConfig {
        net: get(d, "net", base.net)?,
        full_simulations: get(d, "full_simulations", base.full_simulations)?,
        fast_simulations: get(d, "fast_simulations", base.fast_simulations)?,
        full_probability: get(d, "full_probability", base.full_probability)?,
        noise_weight: get(d, "noise_weight", base.noise_weight)?,
        noise_mass: get(d, "noise_mass", base.noise_mass)?,
        temperature_plies: get(d, "temperature_plies", base.temperature_plies)?,
        record: get(d, "record", base.record)?,
        reuse_tree: get(d, "reuse_tree", base.reuse_tree)?,
        external: get(d, "external", base.external)?,
        leaf_batch: get(d, "leaf_batch", base.leaf_batch)?,
        solver_root_nodes: get(d, "solver_root_nodes", base.solver_root_nodes)?,
        solver_leaf_nodes: get(d, "solver_leaf_nodes", base.solver_leaf_nodes)?,
        solver_max_moves: get(d, "solver_max_moves", base.solver_max_moves)?,
        params: SearchParams {
            c_puct: get(d, "c_puct", p.c_puct)?,
            fpu_reduction: get(d, "fpu_reduction", p.fpu_reduction)?,
            fpu_root: get(d, "fpu_root", p.fpu_root)?,
            virtual_loss: get(d, "virtual_loss", p.virtual_loss)?,
            score_weight: get(d, "score_weight", p.score_weight)?,
            score_scale: get(d, "score_scale", p.score_scale)?,
        },
    })
}

fn finished_dict<'py>(py: Python<'py>, g: &FinishedGame) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("index", g.index)?;
    d.set_item("moves", g.moves.clone())?;
    d.set_item("scores", g.scores.to_vec())?;
    d.set_item("seats", g.seats.to_vec())?;
    d.set_item("opening_plies", g.opening_plies)?;
    d.set_item("records", g.records.len())?;
    d.set_item("solved_records", g.records.iter().filter(|r| r.solved).count())?;
    Ok(d)
}

#[pyclass(name = "DuoEngine")]
pub struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    /// agents: two dicts of agent options. openings: lists of actions.
    #[new]
    #[pyo3(signature = (agents, slots, games, *, groups=2, alternate_seats=false, openings=vec![], shared_tree=true, seed=0, solver=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        agents: Vec<Bound<'_, PyDict>>,
        slots: usize,
        games: u64,
        groups: usize,
        alternate_seats: bool,
        openings: Vec<Vec<usize>>,
        shared_tree: bool,
        seed: u64,
        solver: bool,
    ) -> PyResult<Self> {
        if agents.len() != 2 || groups == 0 || slots < groups {
            return Err(PyValueError::new_err("need two agents and slots >= groups >= 1"));
        }
        for opening in &openings {
            let mut game = duo::Game::new();
            for &m in opening {
                game.step(m).map_err(|e| PyValueError::new_err(format!("opening: {e}")))?;
            }
        }
        let config = EngineConfig {
            agents: [agent(&agents[0])?, agent(&agents[1])?],
            slots,
            groups,
            games,
            alternate_seats,
            openings,
            shared_tree,
            seed,
        };
        let solver = solver.then_some(duo_solver::solve as tic_tac_toe::duo_engine::Solver);
        Ok(Self {
            inner: Engine::new(config, solver),
        })
    }

    /// Returns (planes u8, scalars f32, legal u16, counts u32, nets u8) buffers.
    #[allow(clippy::type_complexity)]
    fn prepare(
        &self,
        py: Python<'_>,
        group: usize,
    ) -> PyResult<(
        Py<PyByteArray>,
        Py<PyByteArray>,
        Py<PyByteArray>,
        Py<PyByteArray>,
        Py<PyByteArray>,
    )> {
        if group >= self.inner.config.groups {
            return Err(PyValueError::new_err("unknown group"));
        }
        let batch = py.detach(|| self.inner.prepare(group));
        Ok((
            bytes(py, &batch.planes),
            bytes(py, &batch.scalars),
            bytes(py, &batch.legal),
            bytes(py, &batch.counts),
            bytes(py, &batch.nets),
        ))
    }

    fn apply(
        &self,
        py: Python<'_>,
        group: usize,
        probabilities: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let p = PyBuffer::<f32>::get(probabilities)?.to_vec(py)?;
        let v = PyBuffer::<f32>::get(values)?.to_vec(py)?;
        if p.iter().chain(&v).any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("nonfinite network output"));
        }
        let result = py.detach(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.inner.apply(group, &p, &v)
            }))
        });
        result.map_err(|_| PyValueError::new_err("apply: batch shape does not match prepare"))
    }

    fn done(&self) -> bool {
        self.inner.done()
    }
    fn games_started(&self) -> u64 {
        self.inner.games_started()
    }
    fn active_games(&self) -> usize {
        self.inner.active_games()
    }

    /// Finished games as dicts; with `replay`, their records are stored there too.
    #[pyo3(signature = (replay=None))]
    fn drain<'py>(
        &self,
        py: Python<'py>,
        mut replay: Option<PyRefMut<'_, PyReplay>>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        for g in self.inner.drain_finished() {
            out.push(finished_dict(py, &g)?);
            if let Some(r) = replay.as_mut() {
                r.inner.add_finished(g).map_err(PyValueError::new_err)?;
            }
        }
        Ok(out)
    }

    fn waiting_external(&self) -> Vec<(usize, Vec<u16>)> {
        self.inner.waiting_external()
    }

    fn external_move(&self, slot: usize, action: usize) -> PyResult<()> {
        self.inner
            .external_move(slot, action)
            .map_err(PyValueError::new_err)
    }

    fn counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use std::sync::atomic::Ordering::Relaxed;
        let c = &self.inner.counters;
        let d = PyDict::new(py);
        for (name, value) in [
            ("simulations", &c.simulations),
            ("evaluations", &c.evaluations),
            ("exact_backups", &c.exact_backups),
            ("collisions", &c.collisions),
            ("moves", &c.moves),
            ("solver_moves", &c.solver_moves),
            ("solver_leaf_hits", &c.solver_leaf_hits),
            ("solver_nodes", &c.solver_nodes),
        ] {
            d.set_item(name, value.load(Relaxed))?;
        }
        Ok(d)
    }
}

#[pyclass(name = "DuoReplay")]
pub struct PyReplay {
    inner: Replay,
}

#[pymethods]
impl PyReplay {
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            inner: Replay::new(capacity.max(1)),
        }
    }
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: Replay::from_bytes(data).map_err(PyValueError::new_err)?,
        })
    }
    fn to_bytes(&self, py: Python<'_>) -> Py<PyByteArray> {
        let data = py.detach(|| self.inner.to_bytes());
        PyByteArray::new(py, &data).unbind()
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn games(&self) -> usize {
        self.inner.games()
    }
    #[getter]
    fn added_games(&self) -> u64 {
        self.inner.added_games
    }
    #[getter]
    fn added_positions(&self) -> u64 {
        self.inner.added_positions
    }
    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity
    }
    #[setter]
    fn set_capacity(&mut self, capacity: usize) {
        self.inner.capacity = capacity.max(1);
    }
    #[pyo3(signature = (moves, from_ply=0))]
    fn add_moves(&mut self, moves: Vec<usize>, from_ply: usize) -> PyResult<()> {
        self.inner
            .add_moves(&moves, from_ply)
            .map_err(PyValueError::new_err)
    }
    /// Dict of byte buffers: planes u8, scalars f32, legal u16, counts u32,
    /// policy f32, outcome u8, margin f32, ownership u8.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        window: usize,
        seed: u64,
        augment: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if self.inner.is_empty() {
            return Err(PyValueError::new_err("empty replay"));
        }
        let s = py.detach(|| self.inner.sample(n, window, seed, augment));
        let d = PyDict::new(py);
        d.set_item("planes", bytes(py, &s.planes))?;
        d.set_item("scalars", bytes(py, &s.scalars))?;
        d.set_item("legal", bytes(py, &s.legal))?;
        d.set_item("counts", bytes(py, &s.counts))?;
        d.set_item("policy", bytes(py, &s.policy))?;
        d.set_item("outcome", bytes(py, &s.outcome))?;
        d.set_item("margin", bytes(py, &s.margin))?;
        d.set_item("ownership", bytes(py, &s.ownership))?;
        Ok(d)
    }
}

/// Encode one position for the player to move (symmetry 0..4). Returns
/// (planes u8, scalars f32, legal policy indices, legal actions).
#[pyfunction]
#[pyo3(signature = (moves, sym=0))]
pub fn duo_v2_encode(
    py: Python<'_>,
    moves: Vec<usize>,
    sym: usize,
) -> PyResult<(Py<PyByteArray>, Py<PyByteArray>, Vec<u16>, Vec<usize>)> {
    if sym >= duo_v2::SYMMETRIES {
        return Err(PyValueError::new_err("unknown symmetry"));
    }
    let mut game = duo::Game::new();
    for m in moves {
        game.step(m).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if game.is_terminal() {
        return Err(PyValueError::new_err("terminal position"));
    }
    let actions = game.legal_actions();
    let mut planes = vec![0; duo_v2::PLANES * duo::CELLS];
    let mut scalars = vec![0.0; duo_v2::SCALARS];
    let mut legal = Vec::new();
    duo_v2::encode(&game, &actions, sym, &mut planes, &mut scalars, &mut legal);
    Ok((bytes(py, &planes), bytes(py, &scalars), legal, actions))
}

/// Exact solve of the position after `moves`: (margin for the mover, best action,
/// nodes searched), or None when the node budget runs out.
#[pyfunction]
#[pyo3(signature = (moves, budget=1_000_000))]
pub fn duo_solve(
    py: Python<'_>,
    moves: Vec<usize>,
    budget: u64,
) -> PyResult<(Option<(i32, usize)>, u64)> {
    let mut game = duo::Game::new();
    for m in moves {
        game.step(m).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    if game.is_terminal() {
        return Err(PyValueError::new_err("terminal position"));
    }
    let counter = std::sync::atomic::AtomicU64::new(0);
    let result = py.detach(|| duo_solver::solve_with_budget(&game, budget, &counter));
    Ok((result, counter.into_inner()))
}

#[pyfunction]
pub fn duo_v2_shapes() -> (usize, usize, usize, usize) {
    (duo_v2::PLANES, duo_v2::SCALARS, duo_v2::POLICY_SIZE, duo_v2::SYMMETRIES)
}

/// (cell map, policy-index map) for symmetry `sym`; unused policy ids map to 65535.
#[pyfunction]
pub fn duo_v2_symmetry(sym: usize) -> PyResult<(Vec<u16>, Vec<u16>)> {
    if sym >= duo_v2::SYMMETRIES {
        return Err(PyValueError::new_err("unknown symmetry"));
    }
    Ok(duo_v2::symmetry_tables(sym))
}
