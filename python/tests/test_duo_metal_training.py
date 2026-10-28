from dataclasses import replace

import numpy as np
import pytest
import torch
from blokus_rl._native import DuoGame
from blokus_rl.duo_conv import DuoConvActorCritic
from blokus_rl.duo_deep import DeepConfig, fit_indexed, train
from blokus_rl.duo_env import decision
from blokus_rl.duo_learning import load_duo, save_duo
from blokus_rl.duo_shared import indexed
from blokus_rl.duo_training_device import (
    metal_rng_state,
    prepare_training_device,
    restore_metal_rng,
)
from blokus_rl.learning import configure_torch


def test_metal_rejects_unsupported_precision_and_graphs_before_creating_run(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        DeepConfig(device="mps", precision="bfloat16")
    config = DeepConfig(device="mps", precision="float32")
    with pytest.raises(ValueError, match="CUDA Graphs"):
        train(config, tmp_path / "bad", tmp_path / "missing.pt", inference_backend="cuda_graphs")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="mps_rng"):
        restore_metal_rng("mps", {})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="actual Metal GPU required")
def test_metal_rng_and_optimizer_roundtrip(tmp_path):
    configure_torch(1090)
    prepare_training_device("mps", "float32")
    rng_state = metal_rng_state("mps")
    expected = torch.rand(16, device="mps")
    restore_metal_rng("mps", {"mps_rng": rng_state})
    assert torch.equal(expected, torch.rand(16, device="mps"))
    obs, actions, candidates = decision(DuoGame())
    rows = indexed([{"observation":obs, "candidates":candidates}])
    rows[0].update(visits=np.ones(len(actions), np.uint16), value_target=1.0)
    config = DeepConfig(architecture="conv", device="mps", precision="float32", minibatch_size=4)
    model = DuoConvActorCritic().to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    rng = np.random.default_rng(1090)
    fit_indexed(model, optimizer, rows, rng, config, 2)
    checkpoint = tmp_path / "optimizer.pt"
    torch.save({"weights":model.state_dict(), "optimizer":optimizer.state_dict()}, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recovered = DuoConvActorCritic().to("mps")
    recovered.load_state_dict(restored["weights"])
    second = torch.optim.AdamW(recovered.parameters(), lr=.001)
    second.load_state_dict(restored["optimizer"])
    fit_indexed(model, optimizer, rows, np.random.default_rng(9), config, 1)
    fit_indexed(recovered, second, rows, np.random.default_rng(9), config, 1)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, recovered.state_dict()[key], atol=1e-6, rtol=1e-5)
    assert {int(v["step"].item()) for v in second.state.values()} == {3}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="actual Metal GPU required")
def test_metal_full_trainer_interrupted_recovery(tmp_path):
    configure_torch(1090)
    config = DeepConfig(
        architecture="conv", device="mps", precision="float32", seed=1090,
        rounds=2, games_per_round=2, workers=1, live_games=2,
        fast_probability=0, full_simulations=4, fast_simulations=2,
        inference_batch=8, leaf_batch=2, minibatch_size=8, training_ratio=1,
        eval_every=1, validation_openings=1, validation_simulations=4,
    )
    reference = tmp_path / "reference.pt"
    save_duo(reference, DuoConvActorCritic(), config, 0)
    train(config, tmp_path / "full", reference)
    train(config, tmp_path / "recovered", reference, stop_after_round=1)
    saved = torch.load(tmp_path / "recovered/resume.pt", map_location="cpu", weights_only=False)
    assert isinstance(saved["mps_rng"], torch.Tensor)
    assert saved["runtime"]["deterministic_algorithms"].startswith("warn_only")
    train(config, tmp_path / "recovered", reference, resume=True)
    left = load_duo(tmp_path / "full/final.pt").state_dict()
    right = load_duo(tmp_path / "recovered/final.pt").state_dict()
    for key, value in left.items():
        torch.testing.assert_close(value, right[key], atol=1e-6, rtol=1e-5)
    final = torch.load(tmp_path / "recovered/resume.pt", map_location="cpu", weights_only=False)
    assert final["games"] == 4 and final["round"] == 2
