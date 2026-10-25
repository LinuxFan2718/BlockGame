//! A thin Python boundary. All board changes and rewards come from the Rust engine.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tic_tac_toe::duo;
use tic_tac_toe::{Game, Outcome, Player};

fn player_from_name(name: &str) -> PyResult<Player> {
    match name {
        "X" => Ok(Player::X),
        "O" => Ok(Player::O),
        _ => Err(PyValueError::new_err("player must be X or O")),
    }
}

#[pyclass(name = "Game")]
struct PyGame {
    inner: Game,
}

#[pymethods]
impl PyGame {
    #[new]
    fn new() -> Self {
        Self { inner: Game::new() }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn current_player(&self) -> Option<&'static str> {
        self.inner.current_player().map(|player| match player {
            Player::X => "X",
            Player::O => "O",
        })
    }

    #[getter]
    fn terminated(&self) -> bool {
        self.inner.outcome().is_terminal()
    }

    #[getter]
    fn outcome(&self) -> &'static str {
        match self.inner.outcome() {
            Outcome::InProgress => "in_progress",
            Outcome::Draw => "draw",
            Outcome::Win(Player::X) => "X",
            Outcome::Win(Player::O) => "O",
        }
    }

    fn legal_actions(&self) -> Vec<usize> {
        self.inner.legal_actions().collect()
    }

    /// Returns (cells, legal_mask, to_move) in the requested player's perspective.
    fn observe(&self, player: &str) -> PyResult<(Vec<i8>, Vec<bool>, i8)> {
        let observation = self.inner.observe(player_from_name(player)?);
        Ok((
            observation.cells.to_vec(),
            observation.legal_mask.to_vec(),
            observation.to_move,
        ))
    }

    /// Place one mark; returns (X reward, O reward, terminated).
    fn step(&mut self, action: usize) -> PyResult<(f32, f32, bool)> {
        let step = self
            .inner
            .step(action)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((
            step.reward_for(Player::X),
            step.reward_for(Player::O),
            step.is_terminal(),
        ))
    }

    fn render(&self) -> String {
        self.inner.to_string()
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGame>()?;
    module.add_class::<PyDuoGame>()?;
    module.add_function(wrap_pyfunction!(duo_pieces, module)?)?;
    Ok(())
}

#[pyclass(name = "DuoGame")]
#[derive(Clone)]
struct PyDuoGame {
    inner: duo::Game,
}

#[pymethods]
impl PyDuoGame {
    #[new]
    fn new() -> Self {
        Self {
            inner: duo::Game::new(),
        }
    }
    fn reset(&mut self) {
        self.inner = duo::Game::new();
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    #[getter]
    fn current_player(&self) -> Option<usize> {
        self.inner.current_player()
    }
    #[getter]
    fn terminated(&self) -> bool {
        self.inner.is_terminal()
    }
    #[getter]
    fn board(&self) -> Vec<u8> {
        self.inner.board().to_vec()
    }
    #[getter]
    fn scores(&self) -> Vec<i16> {
        self.inner.scores().to_vec()
    }
    #[getter]
    fn consecutive_passes(&self) -> u8 {
        self.inner.consecutive_passes()
    }
    #[getter]
    fn frontier_counts(&self) -> Vec<usize> {
        self.inner.frontier_counts().to_vec()
    }
    fn remaining(&self, player: usize) -> PyResult<Vec<bool>> {
        if player > 1 {
            return Err(PyValueError::new_err("player must be 0 or 1"));
        }
        Ok(self.inner.remaining(player).to_vec())
    }
    fn required_starts(&self) -> Vec<usize> {
        self.inner.required_starts()
    }
    fn legal_actions(&self) -> Vec<usize> {
        self.inner.legal_actions()
    }
    /// Candidate row: piece id followed by five board cells (196 pads unused cells).
    fn candidates(&self) -> (Vec<usize>, Vec<Vec<usize>>) {
        let actions = self.inner.legal_actions();
        let features = actions
            .iter()
            .map(|&a| {
                if let Some(p) = duo::placement(a) {
                    let mut row = vec![p.piece];
                    row.extend_from_slice(&p.cells);
                    row.resize(6, duo::CELLS);
                    row
                } else {
                    vec![
                        duo::PIECES,
                        duo::CELLS,
                        duo::CELLS,
                        duo::CELLS,
                        duo::CELLS,
                        duo::CELLS,
                    ]
                }
            })
            .collect();
        (actions, features)
    }
    fn describe(&self, action: usize) -> PyResult<(usize, usize, usize, Vec<usize>)> {
        let p = duo::placement(action).ok_or_else(|| PyValueError::new_err("not a placement"))?;
        Ok((p.piece, p.row, p.col, p.cells.clone()))
    }
    fn step(&mut self, action: usize) -> PyResult<(f32, f32, bool)> {
        let rewards = self
            .inner
            .step(action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((rewards[0], rewards[1], self.inner.is_terminal()))
    }
    fn render(&self) -> String {
        self.inner.to_string()
    }
}

#[pyfunction]
fn duo_pieces() -> Vec<Vec<(i8, i8)>> {
    duo::pieces().to_vec()
}
