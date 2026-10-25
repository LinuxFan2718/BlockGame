use std::collections::HashSet;

use tic_tac_toe::{ACTION_COUNT, Agent, Game, MoveError, Outcome, Player, RandomAgent};

fn play(actions: &[usize]) -> Game {
    let mut game = Game::new();
    for &action in actions {
        game.step(action).unwrap();
    }
    game
}

#[test]
fn new_game_has_nine_actions_and_x_starts() {
    let game = Game::new();
    assert_eq!(game.board(), &[None; 9]);
    assert_eq!(game.current_player(), Some(Player::X));
    assert_eq!(game.outcome(), Outcome::InProgress);
    assert_eq!(
        game.legal_actions().collect::<Vec<_>>(),
        (0..9).collect::<Vec<_>>()
    );
    assert_eq!(game.legal_mask(), [true; 9]);
}

#[test]
fn moves_alternate_and_errors_preserve_the_entire_state() {
    let mut game = Game::new();
    let first = game.step(4).unwrap();
    assert_eq!(first.player, Player::X);
    assert_eq!(first.action, 4);
    assert!(!first.is_terminal());
    assert_eq!(first.reward_for(Player::X), 0.0);
    assert_eq!(first.reward_for(Player::O), 0.0);
    assert_eq!(game.current_player(), Some(Player::O));
    let before = game;
    assert_eq!(game.step(4), Err(MoveError::Occupied { action: 4 }));
    assert_eq!(game, before);
    for action in [9, 10, usize::MAX] {
        assert_eq!(game.step(action), Err(MoveError::OutOfBounds { action }));
        assert_eq!(game, before);
    }
    let second = game.step(0).unwrap();
    assert_eq!(second.player, Player::O);
    assert_eq!(game.current_player(), Some(Player::X));
    assert_eq!(game.board()[0], Some(Player::O));
    assert_eq!(game.board()[4], Some(Player::X));
}

#[test]
fn detects_each_of_the_eight_winning_lines() {
    for line in [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ] {
        let other: Vec<_> = (0..9).filter(|cell| !line.contains(cell)).collect();
        let game = play(&[line[0], other[0], line[1], other[1], line[2]]);
        assert_eq!(game.outcome(), Outcome::Win(Player::X), "line {line:?}");
    }
}

#[test]
fn rewards_both_winner_and_loser_and_blocks_moves_after_a_win() {
    for (actions, winner) in [
        (vec![0, 3, 1, 4, 2], Player::X),
        (vec![0, 3, 1, 4, 8, 5], Player::O),
    ] {
        let mut game = Game::new();
        let mut last = None;
        for action in actions {
            last = Some(game.step(action).unwrap());
        }
        let last = last.unwrap();
        assert!(last.is_terminal());
        assert_eq!(last.outcome, Outcome::Win(winner));
        assert_eq!(last.reward_for(winner), 1.0);
        assert_eq!(last.reward_for(winner.opponent()), -1.0);
        assert_eq!(game.current_player(), None);
        assert_eq!(game.legal_actions().count(), 0);
        assert_eq!(game.legal_mask(), [false; 9]);
        let before = game;
        for action in [6, 9, usize::MAX] {
            assert_eq!(
                game.step(action),
                Err(MoveError::GameOver {
                    outcome: last.outcome
                })
            );
            assert_eq!(game, before);
        }
    }
}

#[test]
fn full_board_can_be_a_draw_or_a_win_on_the_last_square() {
    let draw = play(&[0, 1, 2, 4, 3, 5, 7, 6, 8]);
    assert_eq!(draw.outcome(), Outcome::Draw);
    assert_eq!(draw.outcome().reward_for(Player::X), 0.0);
    assert_eq!(draw.outcome().reward_for(Player::O), 0.0);
    assert_eq!(draw.legal_actions().count(), 0);
    let win = play(&[0, 1, 2, 3, 4, 5, 7, 6, 8]);
    assert_eq!(win.outcome(), Outcome::Win(Player::X));
}

#[test]
fn observations_have_explicit_perspectives_and_terminal_masks() {
    let game = play(&[0, 4, 2]);
    let x = game.observe(Player::X);
    let o = game.observe(Player::O);
    assert_eq!(x.perspective, Player::X);
    assert_eq!(o.perspective, Player::O);
    assert_eq!(x.cells, [1, 0, 1, 0, -1, 0, 0, 0, 0]);
    assert_eq!(o.cells, [-1, 0, -1, 0, 1, 0, 0, 0, 0]);
    assert_eq!(x.to_move, -1);
    assert_eq!(o.to_move, 1);
    assert_eq!(
        x.legal_mask,
        [false, true, false, true, false, true, true, true, true]
    );
    assert_eq!(x.legal_mask, o.legal_mask);
    let finished = play(&[0, 3, 1, 4, 2]);
    for player in [Player::X, Player::O] {
        assert_eq!(finished.observe(player).to_move, 0);
        assert_eq!(finished.observe(player).legal_mask, [false; 9]);
    }
}

#[test]
fn snapshots_are_independent_and_reset_starts_a_new_episode() {
    let original = play(&[4, 0]);
    let mut snapshot = original;
    snapshot.step(1).unwrap();
    assert_eq!(original.board()[1], None);
    assert_eq!(snapshot.board()[1], Some(Player::X));
    snapshot.reset();
    assert_eq!(snapshot, Game::new());
    let mut finished = play(&[0, 3, 1, 4, 2]);
    finished.reset();
    assert_eq!(finished, Game::new());
}

#[test]
fn seeded_bots_are_repeatable_and_always_choose_legal_actions() {
    for seed in [0, 1, 42, u64::MAX] {
        let mut a = RandomAgent::new(seed);
        let mut b = RandomAgent::new(seed);
        for _ in 0..100 {
            let mut game = Game::new();
            while let Some(player) = game.current_player() {
                let observation = game.observe(player);
                let action = a.select_action(&observation).unwrap();
                assert_eq!(b.select_action(&observation), Some(action));
                assert!(observation.legal_mask[action]);
                game.step(action).unwrap();
            }
            assert_eq!(a.select_action(&game.observe(Player::X)), None);
            assert_eq!(b.select_action(&game.observe(Player::X)), None);
        }
    }
}

#[test]
fn random_bot_declines_an_opponents_turn_and_handles_one_legal_move() {
    let mut bot = RandomAgent::new(0);
    assert_eq!(bot.select_action(&Game::new().observe(Player::O)), None);
    let game = play(&[0, 1, 2, 4, 3, 5, 7, 6]);
    assert_eq!(bot.select_action(&game.observe(Player::X)), Some(8));
}

// Independent win oracle: generate lines from coordinates, rather than using
// the engine's table. Visit the full legal game tree, stopping at each terminal.
fn reference_outcome(board: &[Option<Player>; ACTION_COUNT]) -> Outcome {
    for player in [Player::X, Player::O] {
        let owns = |row: usize, col: usize| board[3 * row + col] == Some(player);
        let row = (0..3).any(|r| (0..3).all(|c| owns(r, c)));
        let col = (0..3).any(|c| (0..3).all(|r| owns(r, c)));
        let diagonal = (0..3).all(|i| owns(i, i)) || (0..3).all(|i| owns(i, 2 - i));
        if row || col || diagonal {
            return Outcome::Win(player);
        }
    }
    if board.iter().all(Option::is_some) {
        Outcome::Draw
    } else {
        Outcome::InProgress
    }
}

#[test]
fn exhaustive_game_tree_agrees_with_an_independent_rules_oracle() {
    fn visit(game: Game, states: &mut HashSet<Game>, terminals: &mut [usize; 3]) {
        states.insert(game);
        let expected = reference_outcome(game.board());
        assert_eq!(game.outcome(), expected);
        let x_count = game
            .board()
            .iter()
            .filter(|&&cell| cell == Some(Player::X))
            .count();
        let o_count = game
            .board()
            .iter()
            .filter(|&&cell| cell == Some(Player::O))
            .count();
        assert!(x_count == o_count || x_count == o_count + 1);
        if expected.is_terminal() {
            terminals[match expected {
                Outcome::Win(Player::X) => 0,
                Outcome::Win(Player::O) => 1,
                Outcome::Draw => 2,
                Outcome::InProgress => unreachable!(),
            }] += 1;
            assert_eq!(game.current_player(), None);
            assert_eq!(game.legal_mask(), [false; 9]);
            return;
        }
        let player = if x_count == o_count {
            Player::X
        } else {
            Player::O
        };
        assert_eq!(game.current_player(), Some(player));
        let actions: Vec<_> = game.legal_actions().collect();
        assert_eq!(actions.len(), 9 - x_count - o_count);
        for action in 0..9 {
            let legal = game.board()[action].is_none();
            assert_eq!(game.legal_mask()[action], legal);
            assert_eq!(actions.contains(&action), legal);
            let mut next = game;
            if legal {
                let step = next.step(action).unwrap();
                assert_eq!(step.player, player);
                assert_eq!(step.outcome, reference_outcome(next.board()));
                for cell in 0..9 {
                    assert_eq!(
                        next.board()[cell],
                        if cell == action {
                            Some(player)
                        } else {
                            game.board()[cell]
                        }
                    );
                }
                visit(next, states, terminals);
            } else {
                assert_eq!(next.step(action), Err(MoveError::Occupied { action }));
                assert_eq!(next, game);
            }
        }
    }

    let mut states = HashSet::new();
    let mut terminals = [0; 3];
    visit(Game::new(), &mut states, &mut terminals);
    assert_eq!(states.len(), 5_478);
    assert_eq!(terminals, [131_184, 77_904, 46_080]);
}
