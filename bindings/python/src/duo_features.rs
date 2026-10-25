//! Compact observation buffers. No move heuristics or changes to the rules.
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use tic_tac_toe::duo;

pub type Encoded = (Vec<u8>, Vec<u8>, Vec<u8>);
pub type Buffers = (Py<PyByteArray>, Py<PyByteArray>, Py<PyByteArray>);

pub fn encode(game: &duo::Game, actions: &[usize]) -> Encoded {
    let player = game.current_player().expect("nonterminal decision");
    let relative = |cell: usize| if player == 0 { cell } else { 195 - cell };
    let mut state = [0.0_f32; 634];
    for (cell, &color) in game.board().iter().enumerate() {
        if color == player as u8 + 1 {
            state[relative(cell)] = 1.0;
        } else if color != 0 {
            state[196 + relative(cell)] = 1.0;
        }
    }
    for (i, present) in game.remaining(player).iter().enumerate() {
        state[392 + i] = f32::from(*present);
    }
    for (i, present) in game.remaining(1 - player).iter().enumerate() {
        state[413 + i] = f32::from(*present);
    }
    for cell in game.required_starts() {
        state[434 + relative(cell)] = 1.0;
    }
    state[630] = f32::from(player == 0);
    let scores = game.scores();
    // Match Python's float64 arithmetic followed by float32 conversion exactly.
    state[631] = (f64::from(scores[player] + 89) / 109.0) as f32;
    state[632] = (f64::from(scores[1 - player] + 89) / 109.0) as f32;
    state[633] = f32::from(game.consecutive_passes()) / 2.0;
    let observation = state.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ids = actions
        .iter()
        .flat_map(|&action| (action as u32).to_le_bytes())
        .collect();
    let mut candidates = Vec::with_capacity(actions.len() * 12);
    for &action in actions {
        let mut row = [196_u16; 6];
        if let Some(placement) = duo::placement(action) {
            row[0] = placement.piece as u16;
            for (i, &cell) in placement.cells.iter().enumerate() {
                row[i + 1] = relative(cell) as u16;
            }
        } else {
            row[0] = 21;
        }
        candidates.extend(row.iter().flat_map(|x| x.to_le_bytes()));
    }
    (observation, ids, candidates)
}

pub fn buffers(py: Python<'_>, encoded: Encoded) -> Buffers {
    // Fresh writable bytearrays let NumPy/Torch share buffers safely. Mutating
    // a returned observation cannot mutate the Rust game or search tree.
    (
        PyByteArray::new(py, &encoded.0).unbind(),
        PyByteArray::new(py, &encoded.1).unbind(),
        PyByteArray::new(py, &encoded.2).unbind(),
    )
}
