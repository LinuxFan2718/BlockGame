import importlib.util
from dataclasses import replace
from pathlib import Path

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


def test_selected_best_resets_optimizer_keeps_replay_and_recovers(tmp_path):
    # Match the cloud loader: the copied helper still needs its package context.
    path = Path(__file__).parents[1] / "blokus_rl/duo_continuation.py"
    spec = importlib.util.spec_from_file_location("blokus_rl.teacher_continuation", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    initialize_selected = helper.initialize_branch
    configure_torch(91)
    config = small_config()
    reference = tmp_path / "reference.pt"
    save_duo(reference, DuoActorCritic(16), config, 0)
    parent = tmp_path / "parent"
    train(config, parent, reference)
    saved = torch.load(parent / "resume.pt", weights_only=False)
    selected = parent / "initial.pt"
    saved["best_checkpoint"] = torch.load(selected, weights_only=True)
    source = tmp_path / "source.pt"
    torch.save(saved, source)
    before = source.read_bytes()
    child_config = replace(config, seed=1066)
    with pytest.raises(ValueError, match="weights do not match"):
        initialize_branch(child_config, tmp_path / "reject", selected, source)
    left, right = tmp_path / "left", tmp_path / "right"
    for out in (left, right):
        initialize_selected(child_config, out, selected, source, selected_best=True)
        state = torch.load(out / "resume.pt", weights_only=False)
        assert not state["optimizer"]["state"]
        assert len(state["replay"]) == len(saved["replay"])
        assert all(
            torch.equal(v, state["model_state"][k])
            for k, v in saved["best_checkpoint"]["model_state"].items()
        )
        assert all(
            np.array_equal(a["visits"], b["visits"])
            for a, b in zip(state["replay"], saved["replay"], strict=True)
        )
    train(child_config, left, selected, resume=True)
    train(child_config, right, selected, resume=True, stop_after_round=1)
    train(child_config, right, selected, resume=True)
    for name in ("final.pt", "best.pt"):
        a, b = load_duo(left / name).state_dict(), load_duo(right / name).state_dict()
        assert all(torch.equal(v, b[k]) for k, v in a.items())
    assert source.read_bytes() == before
