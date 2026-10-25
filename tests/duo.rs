use std::collections::BTreeSet;
use tic_tac_toe::duo::{self, CELLS, Game, PASS, SIZE, STARTS};

// Deliberately cell-by-cell: does not use the engine's cached edge/corner masks.
fn slow_legal(game: &Game, action: usize) -> bool {
    let Some(p) = duo::placement(action) else {
        return false;
    };
    let Some(player) = game.current_player() else {
        return false;
    };
    let color = player as u8 + 1;
    if !game.remaining(player)[p.piece] {
        return false;
    }
    let mut corner = false;
    for &cell in &p.cells {
        if game.board()[cell] != 0 {
            return false;
        }
        let (r, c) = ((cell / SIZE) as i32, (cell % SIZE) as i32);
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                let (nr, nc) = (r + dr, c + dc);
                if !(0..14).contains(&nr) || !(0..14).contains(&nc) || (dr == 0 && dc == 0) {
                    continue;
                }
                if game.board()[nr as usize * SIZE + nc as usize] == color {
                    if dr == 0 || dc == 0 {
                        return false;
                    }
                    corner = true;
                }
            }
        }
    }
    if game.board().contains(&color) {
        corner
    } else {
        game.required_starts()
            .iter()
            .any(|cell| p.cells.contains(cell))
    }
}

#[test]
fn complete_piece_set_and_orientations() {
    assert_eq!(duo::pieces().len(), 21);
    assert_eq!(duo::pieces().iter().map(Vec::len).sum::<usize>(), 89);
    assert_eq!(
        (1..=5)
            .map(|n| duo::pieces().iter().filter(|p| p.len() == n).count())
            .collect::<Vec<_>>(),
        [1, 1, 2, 5, 12]
    );
    assert_eq!(duo::orientations().len(), 91);
    for o in duo::orientations() {
        assert_eq!(o.cells.iter().collect::<BTreeSet<_>>().len(), o.cells.len());
        // All cells are connected through unit edges.
        let mut reached = BTreeSet::from([o.cells[0]]);
        loop {
            let before = reached.len();
            for &(r, c) in &o.cells {
                if reached
                    .iter()
                    .any(|&(rr, cc)| (rr - r).abs() + (cc - c).abs() == 1)
                {
                    reached.insert((r, c));
                }
            }
            if reached.len() == before {
                break;
            }
        }
        assert_eq!(reached.len(), o.cells.len());
    }
}

#[test]
fn starts_passes_and_atomic_errors() {
    let mut game = Game::new();
    let before = game.clone();
    assert!(game.step(PASS).is_err());
    assert!(game.step(usize::MAX).is_err());
    assert!(game.step(0).is_err());
    assert_eq!(game, before);
    assert_eq!(game.scores(), [-89, -89]);
    assert!(game.legal_actions().contains(&STARTS[0])); // monomino orientation 0
    game.step(STARTS[0]).unwrap();
    assert_eq!(game.required_starts(), [STARTS[1]]);
    assert!(!game.legal_actions().contains(&STARTS[0]));
    game.step(STARTS[1]).unwrap();
    assert_eq!(game.scores(), [-88, -88]);
}

#[test]
fn optimized_move_generator_matches_independent_checker_through_games() {
    // Check EVERY possible action at sampled positions in 12 distinct games.
    let mut rng = 813u64;
    for _ in 0..12 {
        let mut game = Game::new();
        let mut used = [BTreeSet::new(), BTreeSet::new()];
        let mut turns = 0;
        while !game.is_terminal() {
            let actions = game.legal_actions();
            let player = game.current_player().unwrap();
            if game.remaining(player)[0] {
                let mono_moves = actions
                    .iter()
                    .filter(|&&a| duo::placement(a).is_some_and(|p| p.piece == 0))
                    .count();
                assert_eq!(game.frontier_counts()[player], mono_moves);
            }
            if turns % 5 == 0 || actions == [PASS] {
                let expected: Vec<_> = (0..PASS).filter(|&a| slow_legal(&game, a)).collect();
                assert_eq!(
                    actions,
                    if expected.is_empty() {
                        vec![PASS]
                    } else {
                        expected
                    }
                );
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let a = actions[rng as usize % actions.len()];
            let player = game.current_player().unwrap();
            if a != PASS {
                let p = duo::placement(a).unwrap();
                assert!(used[player].insert(p.piece));
                assert!(p.cells.iter().all(|&c| c < CELLS));
            }
            let reward = game.step(a).unwrap();
            assert_eq!(reward[0], -reward[1]);
            if !game.is_terminal() {
                assert_eq!(reward, [0.0, 0.0]);
            }
            turns += 1;
            assert!(turns <= 86); // <=42 placements and forced passes between them
        }
        assert!(game.legal_actions().is_empty());
        assert_eq!(game.current_player(), None);
        assert!(game.step(PASS).is_err());
        for (player, used_pieces) in used.iter().enumerate() {
            let area: usize = used_pieces.iter().map(|&p| duo::pieces()[p].len()).sum();
            assert_eq!(
                area,
                game.board()
                    .iter()
                    .filter(|&&c| c == player as u8 + 1)
                    .count()
            );
            if used_pieces.len() != 21 {
                assert_eq!(game.scores()[player], area as i16 - 89);
            }
        }
    }
}
