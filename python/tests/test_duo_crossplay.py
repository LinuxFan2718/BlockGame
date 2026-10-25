import numpy as np
import pytest
from blokus_rl._native import DuoGame
from blokus_rl.duo_crossplay import evaluate_pair, opening_book, paired_interval
from blokus_rl.duo_league import train_frozen_league
from blokus_rl.duo_learning import DuoConfig, collect_duo
from blokus_rl.duo_model import DuoActorCritic
from blokus_rl.learning import configure_torch


def test_book_replays_and_pair_intervals_resample_openings():
    book = opening_book(4, seed=456)
    assert book == opening_book(4, seed=456)
    assert book != opening_book(4, seed=457)
    for opening in book:
        game = DuoGame()
        for action in opening:
            assert action in game.legal_actions()
            game.step(action)
    # Every opening pair has score 1/2. Resampling games independently would
    # invent sampling uncertainty; resampling openings correctly gives none.
    assert paired_interval([[1, 0]] * 50) == [0.5, 0.5]
    assert paired_interval([[1, 0]]) is None
    assert paired_interval([[1, 1]] * 50) == [1.0, 1.0]
    with pytest.raises(ValueError):
        paired_interval([1, 0])


def test_paired_crossplay_loss_example_and_seats():
    configure_torch(9)
    book = opening_book(3, seed=8765)
    result = evaluate_pair("random", "largest", book, 4567)
    assert result == evaluate_pair("random", "largest", book, 4567)
    assert result["games"] == 6
    assert all(sum(s[k] for k in ("wins", "draws", "losses")) == 3 for s in result["by_seat"])
    if result["loss_example"]:
        example = result["loss_example"]
        game = DuoGame()
        for action in example["actions"]:
            game.step(action)
        assert game.terminated
        assert game.scores == example["scores"]
        seat = example["learner_seat"]
        assert game.scores[seat] < game.scores[1 - seat]
    assert np.isfinite(result["score_rate"])


def test_training_selector_is_once_per_episode_and_neural_rival_stays_frozen():
    import torch

    configure_torch(987)
    model, rival = DuoActorCritic(16), DuoActorCritic(16).requires_grad_(False)
    before = {k: v.clone() for k, v in rival.state_dict().items()}
    calls = []

    def selector(rng):
        calls.append(1)
        return rival

    config = DuoConfig(updates=1, episodes_per_update=2, hidden_size=16)
    records, _ = collect_duo(
        model,
        config,
        np.random.default_rng(4),
        [],
        opponent_selector=selector,
        sample_opponent=False,
    )
    assert len(calls) == 2
    assert records
    assert all(torch.equal(value, before[key]) for key, value in rival.state_dict().items())


def test_frozen_league_rejects_held_out_checkpoint_leakage(tmp_path):
    # A copied checkpoint at a different path must still trigger the guard.
    first, duplicate = tmp_path / "first", tmp_path / "duplicate"
    first.write_bytes(b"same model")
    duplicate.write_bytes(b"same model")
    with pytest.raises(ValueError, match="held-out"):
        train_frozen_league(first, [first], first, duplicate, tmp_path / "out")
    assert not (tmp_path / "out").exists()
