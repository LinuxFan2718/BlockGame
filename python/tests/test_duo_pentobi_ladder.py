import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from blokus_rl.duo_learning import DuoConfig, save_duo
from blokus_rl.duo_model import DuoActorCritic
from blokus_rl.duo_pentobi import DEFAULT_BINARY
from blokus_rl.duo_pentobi_ladder import LadderConfig, run, summarize
from blokus_rl.learning import configure_torch


def test_zero_score_bound_respects_opening_pairs():
    games = [{"margin": -1, "seconds": 1.0, "policy_metrics": {}}] * 2
    result = summarize([{"index": i, "games": games} for i in range(100)], 100)
    assert result["games"] == 200 and result["score_rate"] == 0
    assert result["score_rate_95pct_cluster_bootstrap"] == [0.0, 0.0]
    assert np.isclose(result["zero_score_one_sided_95_upper"], 0.0295130496070399)
    for levels in ((0,), (10,), (1, 1), ()):
        with pytest.raises(ValueError):
            LadderConfig(levels=levels)


@pytest.mark.skipif(not DEFAULT_BINARY.exists(), reason="optional local Pentobi31.0")
def test_parallel_ladder_resume_matches_games_and_rejects_changed_configuration(tmp_path):
    configure_torch(2571000)
    model = DuoActorCritic(16)
    checkpoint = tmp_path / "source.pt"
    save_duo(checkpoint, model, DuoConfig(hidden_size=16), 0)
    config = LadderConfig(opening_pairs=2, seed=2571000, simulations=4, workers=2, levels=(1, 2))
    out = tmp_path / "run"
    first = run(config, checkpoint, DEFAULT_BINARY, out)
    # Remove one completed pair and summary to simulate recovery from a partial commit.
    original = json.loads((out / "level-02/pair-001.json").read_text())
    (out / "level-02/pair-001.json").unlink()
    (out / "completed.json").unlink()
    recovered = run(config, checkpoint, DEFAULT_BINARY, out, resume=True)
    actual = json.loads((out / "level-02/pair-001.json").read_text())
    for expected_game, actual_game in zip(original["games"], actual["games"], strict=True):
        assert expected_game["actions"] == actual_game["actions"]
        assert expected_game["margin"] == actual_game["margin"]
    assert recovered["1"]["opening_pair_scores"] == first["1"]["opening_pair_scores"]
    assert recovered["2"]["opening_pair_scores"] == first["2"]["opening_pair_scores"]
    with pytest.raises(ValueError, match="protocol/model/binary changed"):
        run(replace(config, seed=2571001), checkpoint, DEFAULT_BINARY, out, resume=True)
    with torch.no_grad():
        model.value_head.bias.add_(0.1)
    save_duo(checkpoint, model, DuoConfig(hidden_size=16), 0)
    with pytest.raises(ValueError, match="protocol/model/binary changed"):
        run(config, checkpoint, DEFAULT_BINARY, out, resume=True)
