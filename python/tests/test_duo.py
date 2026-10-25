from unittest.mock import patch

import numpy as np
import pytest
import torch
from blokus_rl._native import DuoGame
from blokus_rl.duo_env import CELLS, LEGACY_STATE_SIZE, PASS, STATE_SIZE, DuoEnv, decision
from blokus_rl.duo_learning import (
    DuoConfig,
    collect_duo,
    evaluate_duo,
    load_duo,
    make_duo_loss,
    optimize_duo,
    pack,
    save_duo,
)
from blokus_rl.duo_model import DuoActorCritic
from blokus_rl.learning import configure_torch
from pettingzoo.test import api_test


def test_native_state_copy_and_atomic_errors():
    game = DuoGame()
    assert game.scores == [-89, -89]
    before = game.render()
    with pytest.raises(ValueError):
        game.step(PASS)
    assert game.render() == before
    state, actions, moves = decision(game)
    assert state.shape == (STATE_SIZE,)
    assert moves.shape == (len(actions), 6)
    assert len(actions) == len(set(actions))
    copy = game.copy()
    copy.step(int(actions[0]))
    assert copy.board != game.board
    assert game.current_player == 0
    assert copy.current_player == 1
    _, actions2, moves2 = decision(copy)
    for action, move in zip(actions2, moves2, strict=True):
        _, _, _, cells = copy.describe(int(action))
        assert sorted(move[1:][move[1:] < CELLS]) == sorted(CELLS - 1 - c for c in cells)


def test_pettingzoo_duo_contract():
    api_test(DuoEnv(), num_cycles=100, verbose_progress=False)


def test_ppo_ragged_moves_and_checkpoint(tmp_path):
    configure_torch(13)
    config = DuoConfig(updates=1, episodes_per_update=2, epochs=1, hidden_size=32)
    model = DuoActorCritic(32)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    records, _ = collect_duo(model, config, np.random.default_rng(13), [])
    # All log probabilities must be from this exact frozen rollout policy,
    # regardless of how many moves each decision had and how it is padded.
    batch = pack(records)
    with torch.no_grad():
        distribution, _ = model.distribution(
            batch["observation"], batch["candidates"], batch["action_mask"]
        )
        assert torch.allclose(
            distribution.log_prob(batch["action"]), batch["action_log_prob"], atol=2e-6
        )
        assert torch.all(distribution.probs[~batch["action_mask"]] == 0)
    stats = optimize_duo(
        model,
        make_duo_loss(model, config),
        torch.optim.Adam(model.parameters(), lr=0.001),
        records,
        config,
    )
    assert all(np.isfinite(x) for x in stats.values())
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
    path = tmp_path / "bot.pt"
    save_duo(path, model, config, 1)
    loaded = load_duo(path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    assert not path.with_suffix(".tmp").exists()


def test_evaluation_frozen_rng_and_both_seats():
    configure_torch(31)
    model = DuoActorCritic(16)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    a = evaluate_duo(model, "largest", 2, 12345)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert a == evaluate_duo(model, "largest", 2, 12345)
    assert a["games"] == 4
    assert set(a["by_seat"]) == {"0", "1"}
    assert a["wins"] + a["draws"] + a["losses"] == 4
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())


def test_neural_opponent_is_sampled_but_learner_is_greedy():
    # Without stochastic opponent moves a deterministic matchup merely repeats
    # the same two games, making the reported game-count interval meaningless.
    from blokus_rl import duo_learning

    configure_torch(32)
    model = DuoActorCritic(16)
    opponent = DuoActorCritic(16)
    observed = []
    original = duo_learning.choose

    def record(game, policy, rng, *, sample=False):
        observed.append((policy is opponent, sample))
        return original(game, policy, rng, sample=sample)

    with patch.object(duo_learning, "choose", record):
        evaluate_duo(model, opponent, 1, 56789)
    assert (True, True) in observed
    assert (False, False) in observed
    assert all(is_opponent == sample for is_opponent, sample in observed)


def test_legacy_checkpoint_migration_preserves_policy(tmp_path):
    configure_torch(9)
    old = DuoActorCritic(16, state_size=LEGACY_STATE_SIZE)
    path = tmp_path / "version1.pt"
    torch.save(
        {
            "game": "blokus_duo",
            "format_version": 1,
            "hidden_size": 16,
            "model_state": old.state_dict(),
        },
        path,
    )
    migrated = load_duo(path)
    state, _, candidates = decision(DuoGame())
    state[-3:] = (0.25, 0.75, 0.5)
    with torch.no_grad():
        old_logits = old.logits(
            torch.from_numpy(state[:LEGACY_STATE_SIZE]), torch.from_numpy(candidates)
        )
        new_logits = migrated.logits(torch.from_numpy(state), torch.from_numpy(candidates))
    assert torch.allclose(old_logits, new_logits, atol=1e-7)
    assert torch.count_nonzero(migrated.trunk[0].weight[:, -3:]) == 0


def test_frontier_baseline_plays_legal_complete_games():
    configure_torch(53)
    result = evaluate_duo("frontier", "largest", 2, 791000)
    assert result["games"] == 4
    assert result["wins"] + result["draws"] + result["losses"] == 4
