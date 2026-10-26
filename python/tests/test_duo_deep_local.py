import json

import numpy as np
import pytest
from blokus_rl import duo_deep_local as local
from blokus_rl.duo_crossplay import opening_book
from blokus_rl.duo_deep_evaluate import evaluate_many
from blokus_rl.duo_model import DuoActorCritic
from blokus_rl.learning import configure_torch


def test_local_pair_resume_preserves_actual_games_and_rejects_changed_settings(
    tmp_path, monkeypatch
):
    configure_torch(765)
    model = DuoActorCritic(16).eval()
    book = opening_book(3, seed=2561000)
    settings = {"protocol_sha256": "smoke", "case": "smoke", "simulations": 0, "seed": 2561000}
    out = tmp_path / "result.json"
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 2:
            raise RuntimeError("simulated process interruption")
        return evaluate_many(*args, **kwargs)

    monkeypatch.setattr(local, "evaluate_many", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        local.paired_case(model, "largest", book, out, settings, chunk_pairs=1)
    partial = json.loads(out.with_suffix(".partial.json").read_text())
    assert len(partial["chunks"]) == 1 and partial["chunks"][0]["games"] == 2
    with pytest.raises(ValueError, match="settings changed"):
        local.paired_case(model, "largest", book, out, settings | {"seed": 1}, chunk_pairs=1)
    monkeypatch.setattr(local, "evaluate_many", evaluate_many)
    result = local.paired_case(model, "largest", book, out, settings, chunk_pairs=1)
    expected = evaluate_many(
        model, "largest", book, simulations=0, opponent_simulations=0, seed=2561000
    )
    assert result["opening_pair_scores"] == expected["opening_pair_scores"]
    assert (
        result["score_rate_95pct_cluster_bootstrap"]
        == expected["score_rate_95pct_cluster_bootstrap"]
    )
    assert result["wins"] + result["draws"] + result["losses"] == 6
    assert np.isclose(result["mean_margin"], expected["mean_margin"])
    assert local.paired_case(model, "largest", book, out, settings, chunk_pairs=1) == result
    with pytest.raises(ValueError, match="settings changed"):
        local.paired_case(
            model, "largest", book, out, settings | {"protocol_sha256": "new"}, chunk_pairs=1
        )
