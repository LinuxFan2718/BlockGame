import json

import pytest
from blokus_rl.duo_pentobi import DEFAULT_BINARY
from blokus_rl.duo_pentobi_steps import run


def test_sequential_stop_and_recovery_keep_every_pair(tmp_path):
    checkpoint, binary = tmp_path / "model", tmp_path / "engine"
    checkpoint.write_bytes(b"fixed-model")
    binary.write_bytes(b"fixed-engine")
    config = {
        "first_level": 2,
        "last_level": 9,
        "opening_pairs": 2,
        "seed": 2597000,
        "simulations": 128,
        "device": "cpu",
        "precision": "float32",
    }
    calls = []

    def play(level, index, opening):
        calls.append((level, index))
        # 50% from wins/losses, then 50% from draws, then 25%: stop at4.
        margins = {2: ((1, -1), (1, -1)), 3: ((0, 0), (0, 0)), 4: ((-1, -1), (-1, 1))}
        return {
            "level": level,
            "index": index,
            "games": [
                {
                    "seat": seat,
                    "seed": config["seed"] + index + seat * 1000000,
                    "actions": opening,
                    "margin": margins[level][index][seat],
                    "seconds": 1,
                    "policy_metrics": {},
                }
                for seat in (0, 1)
            ],
        }

    out = tmp_path / "run"
    first = run(config, checkpoint, binary, out, play=play)
    assert list(first) == ["2", "3", "4"]
    assert first["3"]["win_rate"] == 0 and first["3"]["score_rate"] == 0.5
    assert first["4"]["score_rate"] == 0.25
    assert calls == [(level, i) for level in (2, 3, 4) for i in range(2)]
    unchanged = (out / "level-02/pair-000.json").read_bytes()
    (out / "level-03/pair-001.json").unlink()
    (out / "completed.json").unlink()
    calls.clear()
    recovered = run(config, checkpoint, binary, out, play=play, resume=True)
    assert calls == [(3, 1)]
    assert recovered == first
    assert (out / "level-02/pair-000.json").read_bytes() == unchanged
    assert (
        json.loads((out / "completed.json").read_text())["reason"]
        == "first_complete_level_below_half_score"
    )
    with pytest.raises(ValueError, match="protocol changed"):
        run(config | {"seed": 2597001}, checkpoint, binary, out, play=play, resume=True)


@pytest.mark.skipif(not DEFAULT_BINARY.exists(), reason="optional local Pentobi31.0")
def test_new_model_real_pair_is_crosschecked(tmp_path):
    checkpoint = (
        "runs/duo-continuation-final-20261026T2007/duo-continuation-865/frozen-strong1024.pt"
    )
    config = {
        "first_level": 2,
        "last_level": 2,
        "opening_pairs": 1,
        "seed": 2597000,
        "simulations": 4,
        "device": "cpu",
        "precision": "float32",
    }
    result = run(config, checkpoint, DEFAULT_BINARY, tmp_path / "smoke")
    assert result["2"]["games"] == 2
    assert result["2"]["policy_metrics"]["simulations"] > 0
