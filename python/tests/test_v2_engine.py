import numpy as np
import torch
from blokus_rl._native import DuoGame, DuoReplay, duo_solve, duo_v2_encode
from blokus_rl.v2 import engine as v2
from blokus_rl.v2.benchmark import UniformEvaluator
from blokus_rl.v2.model import POLICY_SIZE, DuoNetV2


def small_agent(**overrides):
    return v2.self_play_agent(full_simulations=24, fast_simulations=6, **overrides)


def test_uniform_self_play_fills_replay_with_valid_games():
    engine = v2.make_engine([small_agent(), small_agent()], 16, 20, seed=3)
    replay = DuoReplay(100000)
    finished = []
    v2.run(engine, UniformEvaluator(), on_finished=finished.extend, replay=replay)
    assert len(finished) == 20
    assert replay.games == sum(1 for g in finished if g["records"])
    for g in finished:
        game = DuoGame()
        for m in g["moves"]:
            game.step(m)
        assert game.terminated and list(game.scores) == g["scores"]
    batch = replay.sample(32, len(replay), 1, True)
    counts = np.frombuffer(batch["counts"], np.uint32)
    policy = np.frombuffer(batch["policy"], np.float32)
    assert len(counts) == 32 and len(policy) == counts.sum()
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)
    for a, b in zip(offsets[:-1], offsets[1:]):
        assert abs(policy[a:b].sum() - 1) < 1e-5


def test_network_probabilities_are_normalized_over_legal_moves():
    torch.manual_seed(0)
    model = DuoNetV2(blocks=1, channels=16, context=8)
    evaluator = v2.Evaluator(model, "cpu")
    engine = v2.make_engine([small_agent(), small_agent()], 4, 4, seed=1)
    batch = engine.prepare(0)
    counts = np.frombuffer(batch[3], np.uint32)
    probabilities, values = evaluator.collect(evaluator.submit(batch))
    assert probabilities.shape == (counts.sum(),) and values.shape == (len(counts),)
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)
    for a, b in zip(offsets[:-1], offsets[1:]):
        assert abs(probabilities[a:b].sum() - 1) < 1e-4
    assert np.all(np.abs(values) <= 1)
    engine.apply(0, probabilities, values)


def test_encoder_policy_indices_cover_the_policy_space():
    planes, scalars, legal, actions = duo_v2_encode([], 0)
    assert len(legal) == len(actions) == 828
    assert max(legal) < POLICY_SIZE
    assert np.frombuffer(planes, np.uint8).size == 9 * 196


def test_external_seat_plays_against_search():
    rng = np.random.default_rng(0)
    agent = v2.play_agent(16)
    engine = v2.make_engine(
        [agent, {**agent, "external": True}], 4, 4, alternate_seats=True, shared_tree=False
    )

    def external(engine):
        for slot, moves in engine.waiting_external():
            game = DuoGame()
            for m in moves:
                game.step(m)
            engine.external_move(slot, int(rng.choice(game.legal_actions())))

    finished = []
    v2.run(engine, UniformEvaluator(), on_finished=finished.extend, external=external)
    assert sorted(tuple(g["seats"]) for g in finished) == [(0, 1), (0, 1), (1, 0), (1, 0)]


def minimax(game):
    """Exact final margin for the player to move, by plain enumeration."""
    mover, best = game.current_player, None
    for action in game.legal_actions():
        child = game.copy()
        child.step(action)
        if child.terminated:
            value = child.scores[mover] - child.scores[1 - mover]
        else:
            value = -minimax(child)
        best = value if best is None else max(best, value)
    return best


def test_solver_matches_python_minimax_late_in_games():
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(6):
        game, moves = DuoGame(), []
        while not game.terminated:
            action = int(rng.choice(game.legal_actions()))
            game.step(action)
            moves.append(action)
        for back in (2, 3, 4):
            position = DuoGame()
            for m in moves[:-back]:
                position.step(m)
            (margin, action), _ = duo_solve(moves[:-back], 1_000_000)
            assert margin == minimax(position)
            assert action in position.legal_actions()
            checked += 1
    assert checked == 18


def test_single_move_choice_for_the_app():
    from blokus_rl.v2.play import choose_move

    torch.manual_seed(0)
    evaluator = v2.Evaluator(DuoNetV2(blocks=1, channels=16, context=8), "cpu")
    game, moves = DuoGame(), []
    for _ in range(6):
        for simulations in (0, 16):
            action = choose_move(evaluator, moves, simulations)
            assert action in game.legal_actions()
        game.step(action)
        moves.append(action)
