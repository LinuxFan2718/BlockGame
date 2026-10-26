from dataclasses import replace

import numpy as np
import pytest
import torch
from blokus_rl.duo_continuation import initialize_branch
from blokus_rl.duo_deep import train
from blokus_rl.duo_learning import load_duo, save_duo
from blokus_rl.duo_model import DuoActorCritic
from blokus_rl.learning import configure_torch
from test_duo_deep import small_config


def test_fork_preserves_state_rejects_changed_parent_and_resumes(tmp_path):
    configure_torch(764)
    parent_config = small_config()
    reference = tmp_path / "old-reference.pt"
    save_duo(reference, DuoActorCritic(16), parent_config, 0)
    parent = tmp_path / "parent"
    train(parent_config, parent, reference)
    source, frozen = parent / "resume.pt", parent / "final.pt"
    source_bytes, frozen_bytes = source.read_bytes(), frozen.read_bytes()
    config = replace(parent_config, seed=765, full_simulations=8, validation_seed=2581000)
    left, right = tmp_path / "left", tmp_path / "right"
    initialize_branch(config, left, frozen, source)
    a = torch.load(left / "resume.pt", weights_only=False)
    b = torch.load(source, weights_only=False)
    assert a["round"] == a["games"] == 0 and a["best"] == 0.5
    assert all(torch.equal(value, b["model_state"][key]) for key, value in a["model_state"].items())
    assert len(a["replay"]) == len(b["replay"])
    assert all(
        np.array_equal(x["observation"], y["observation"])
        for x, y in zip(a["replay"], b["replay"], strict=True)
    )
    for key, value in a["optimizer"]["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, b["optimizer"]["state"][key][name])
    initialize_branch(config, right, frozen, source)
    train(config, left, frozen, resume=True)
    train(config, right, frozen, resume=True, stop_after_round=1)
    train(config, right, frozen, resume=True)
    for name in ("final.pt", "best.pt"):
        x, y = load_duo(left / name).state_dict(), load_duo(right / name).state_dict()
        assert all(torch.equal(value, y[key]) for key, value in x.items())
    assert source.read_bytes() == source_bytes and frozen.read_bytes() == frozen_bytes
    with pytest.raises(ValueError, match="already exists"):
        initialize_branch(config, left, frozen, source)
    with pytest.raises(ValueError, match="unsupported setting"):
        initialize_branch(replace(config, learning_rate=0.01), tmp_path / "bad-lr", frozen, source)
    assert not (tmp_path / "bad-lr").exists()
    different = tmp_path / "different.pt"
    save_duo(different, DuoActorCritic(16), parent_config, 99)
    with pytest.raises(ValueError, match="weights do not match"):
        initialize_branch(config, tmp_path / "bad-parent", different, source)
    assert not (tmp_path / "bad-parent").exists()
