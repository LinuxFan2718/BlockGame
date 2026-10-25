import numpy as np
import pytest
import torch
from blokus_rl._native import DuoGame, DuoSearch
from blokus_rl.duo_az import AzConfig, fit
from blokus_rl.duo_env import decision, legacy_decision
from blokus_rl.duo_mcts import search_games
from blokus_rl.duo_model import DuoActorCritic
from blokus_rl.duo_pentobi import DEFAULT_BINARY, Pentobi, match, move_cells, move_string
from blokus_rl.learning import configure_torch


def test_batched_search_reproducibility_budget_and_root_noise():
    configure_torch(642)
    model = DuoActorCritic(16)
    games = [DuoGame(), DuoGame()]
    games[1].step(games[1].legal_actions()[0])
    before = [g.render() for g in games]
    a = search_games(model, games, 32, leaf_batch=4)
    b = search_games(model, games, 32, leaf_batch=4)
    for i, (left, right) in enumerate(zip(a, b, strict=True)):
        assert np.array_equal(left.visits, right.visits)
        assert left.visits.sum() == 32
        assert left.action() in games[i].legal_actions()
        assert np.isclose(left.policy.sum(), 1)
        assert games[i].render() == before[i]
    noisy = search_games(
        model, games, 32, rngs=[np.random.default_rng(1), np.random.default_rng(2)]
    )
    assert not np.allclose(a[0].priors, noisy[0].priors)
    # Soft search targets must update both heads with finite gradients.
    records = [
        {
            "observation": r.observation,
            "candidates": r.candidates,
            "policy_target": r.policy,
            "value_target": float(i * 2 - 1),
        }
        for i, r in enumerate(a)
    ]
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    losses = fit(
        model,
        torch.optim.AdamW(model.parameters()),
        records,
        np.random.default_rng(3),
        AzConfig(gradient_steps=2, minibatch_size=2),
    )
    assert all(np.isfinite(v) for v in losses.values())
    assert not torch.equal(weights["policy_head.weight"], model.policy_head.weight)
    assert not torch.equal(weights["value_head.weight"], model.value_head.weight)


def test_endgame_search_agrees_with_independent_minimax_in_both_perspectives():
    # Reachable position with winning and losing moves for player 1. The oracle
    # explores just this tiny endgame, never supplies training labels.
    prefix = [
        15407,
        11820,
        8573,
        4933,
        4272,
        497,
        898,
        132,
        2249,
        14847,
        11635,
        16628,
        6550,
        8883,
        17483,
        11201,
        9234,
        3867,
        1540,
        14220,
        245,
        7422,
        5491,
        703,
        122,
        12670,
        5038,
    ]
    game = DuoGame()
    for action in prefix:
        game.step(action)

    def solve(state):
        if state.terminated:
            return int(np.sign(state.scores[0] - state.scores[1]))
        values = []
        for action in state.legal_actions():
            child = state.copy()
            child.step(action)
            values.append(solve(child))
        return (max if state.current_player == 0 else min)(values)

    def search(state):
        tree = DuoSearch(state)
        while tree.completed < 2048:
            for token, leaf in tree.request(min(8, 2048 - tree.completed)):
                tree.expand(token, [1.0] * len(leaf.legal_actions()), 0.0)
        stats = tree.stats()
        action = stats["actions"][int(np.argmax(stats["visits"]))]
        child = state.copy()
        child.step(action)
        assert solve(child) == solve(state)
        assert stats["simulations"] == sum(stats["visits"]) == 2048

    assert game.current_player == 1
    search(game)
    # This continuation hands the decision to player 0, testing the sign flip.
    game.step(1078)
    assert game.current_player == 0
    search(game)


def test_coordinate_rotation_round_trip():
    for rotate in (False, True):
        cells = frozenset([0, 1, 14, 60, 135])
        assert move_cells(move_string(cells, rotate), rotate) == cells
        assert move_cells(move_string([], rotate), rotate) == frozenset()


def test_bfloat16_search_crosses_numpy_boundary_safely():
    configure_torch(642)
    game = DuoGame()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = search_games(DuoActorCritic(16), [game], 16)[0]
    assert result.action() in game.legal_actions()
    assert result.policy.dtype == np.float32
    assert np.isfinite(result.priors).all()
    assert result.visits.sum() == 16


def test_native_feature_buffers_match_reference_for_complete_games():
    for seed in range(4):
        game = DuoGame()
        rng = np.random.default_rng(seed)
        while not game.terminated:
            native = decision(game)
            reference = legacy_decision(game)
            assert all(np.array_equal(a, b) for a, b in zip(native, reference, strict=True))
            assert all(a.flags.writeable for a in native)
            game.step(int(rng.choice(game.legal_actions())))
        with pytest.raises(ValueError):
            decision(game)


def test_native_leaf_features_preserve_search_visits_exactly():
    configure_torch(260642)
    model = DuoActorCritic(32)
    games = [DuoGame(), DuoGame()]
    games[1].step(games[1].legal_actions()[123])
    fast = search_games(model, games, 128, native_features=True)
    reference = search_games(model, games, 128, native_features=False)
    for a, b in zip(fast, reference, strict=True):
        assert np.array_equal(a.visits, b.visits)
        assert np.array_equal(a.priors, b.priors)
        assert np.array_equal(a.means, b.means)
        assert a.action() == b.action()


def test_self_play_resume_restores_weights_optimizer_replay_and_rng(tmp_path, monkeypatch):
    from blokus_rl import duo_az
    from blokus_rl.duo_learning import load_duo

    class LocalPool:
        interrupt = False

        def __init__(self, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, tasks):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                raise InterruptedError("simulated interruption between rounds")
            for task in tasks:
                # Actual workers have separate torch RNGs; preserve that property.
                with torch.random.fork_rng():
                    yield function(task)

    monkeypatch.setattr(duo_az, "ProcessPoolExecutor", LocalPool)
    config = AzConfig(
        rounds=2,
        games_per_round=2,
        simulations=4,
        workers=1,
        live_games=2,
        gradient_steps=2,
        minibatch_size=8,
        eval_every=1,
        validation_openings=1,
    )
    duo_az.train(config, tmp_path / "full")
    LocalPool.interrupt = True
    with pytest.raises(InterruptedError):
        duo_az.train(config, tmp_path / "interrupted")
    LocalPool.interrupt = False
    duo_az.train(config, tmp_path / "interrupted", resume=True)
    expected = load_duo(tmp_path / "full" / "final.pt").state_dict()
    actual = load_duo(tmp_path / "interrupted" / "final.pt").state_dict()
    assert all(torch.equal(value, actual[key]) for key, value in expected.items())


@pytest.mark.skipif(not DEFAULT_BINARY.exists(), reason="optional local Pentobi binary")
def test_pentobi_independent_rules_and_final_score():
    # Seed 42's first random largest move chooses the second starting square,
    # exercising rotation as well as forced passes and terminal score agreement.
    configure_torch(42)
    with Pentobi(seed=42) as pentobi:
        result = match(pentobi, "largest", 0, 42, check_legal=True)
    assert len(result["actions"]) > 20
