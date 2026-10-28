from dataclasses import replace

import numpy as np
import pytest
import torch
from blokus_rl._native import DuoGame
from blokus_rl.duo_crossplay import evaluate_pair, opening_book
from blokus_rl.duo_deep import DeepConfig, fit_indexed, train
from blokus_rl.duo_deep_evaluate import evaluate_many
from blokus_rl.duo_env import decision
from blokus_rl.duo_learning import load_duo, save_duo
from blokus_rl.duo_mcts import evaluate_records
from blokus_rl.duo_model import DuoActorCritic
from blokus_rl.duo_pentobi import DEFAULT_BINARY
from blokus_rl.duo_replay import replay_records
from blokus_rl.duo_residual import DuoResidualActorCritic, candidate_indices_numpy
from blokus_rl.duo_shared import SharedSearchPool, collect, evaluate_indexed, indexed
from blokus_rl.learning import configure_torch


def small_config(**kwargs):
    return replace(
        DeepConfig(),
        device="cpu",
        precision="float32",
        channels=8,
        blocks=1,
        hidden_size=16,
        rounds=2,
        games_per_round=2,
        workers=1,
        live_games=2,
        full_simulations=4,
        fast_simulations=2,
        fast_probability=0,
        leaf_batch=2,
        inference_batch=8,
        minibatch_size=8,
        training_ratio=1,
        eval_every=1,
        validation_openings=1,
        validation_simulations=4,
        **kwargs,
    )


def test_residual_canonical_action_encoding_across_complete_game():
    configure_torch(764)
    model = DuoResidualActorCritic(16, 8, 1).eval()
    game, rng = DuoGame(), np.random.default_rng(51)
    while not game.terminated:
        _, _, candidates = decision(game)
        expected = model.candidate_indices(torch.from_numpy(candidates)).numpy()
        assert np.array_equal(expected, candidate_indices_numpy(candidates))
        assert expected.max() < 17837
        game.step(int(rng.choice(game.legal_actions())))


def test_residual_inventory_batching_serialization_and_gradients(tmp_path):
    configure_torch(764)
    config = small_config()
    model = DuoResidualActorCritic(16, 8, 1).eval()
    observation, _, candidates = decision(DuoGame())
    rows = [{"observation": observation, "candidates": candidates}]
    reference, values = evaluate_records(model, rows)
    probabilities, actual_values = evaluate_indexed(model, indexed(rows))
    assert np.allclose(reference, probabilities, atol=1e-7)
    assert np.allclose(values, actual_values)
    repeated, _ = evaluate_indexed(model, indexed(rows * 3))
    assert np.allclose(repeated[0], probabilities[0], atol=1e-6)
    altered = torch.from_numpy(observation.copy())
    altered[392] = 0
    assert not torch.allclose(model(altered)[0], model(torch.from_numpy(observation))[0])
    save_duo(tmp_path / "model.pt", model, config, 0)
    loaded = load_duo(tmp_path / "model.pt")
    assert loaded.blocks == 1 and loaded.channels == 8
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    records = indexed(rows)
    records[0].update(visits=np.ones(len(candidates), dtype=np.uint16), value_target=1.0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses = fit_indexed(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.001),
        records,
        np.random.default_rng(2),
        config,
        2,
    )
    assert all(np.isfinite(x) for x in losses.values())
    assert not torch.equal(before["policy_head.weight"], model.policy_head.weight)
    assert not torch.equal(before["value_head.2.weight"], model.value_head[2].weight)


def test_shared_process_inference_matches_direct_collection():
    configure_torch(764)
    config = replace(small_config(), workers=2, live_games=1)
    model = DuoResidualActorCritic(16, 8, 1).eval()
    direct = []
    for worker in range(2):
        rows, _, _ = collect(
            config,
            config.seed + 1000000 + worker * 10000,
            1,
            lambda rows: evaluate_indexed(model, indexed(rows)),
        )
        direct.extend(rows)
    with SharedSearchPool(config) as pool:
        remote, scores, metrics = pool.collect(model, 1, 2)
    assert len(scores) == 2 and len(remote) == len(direct)
    assert metrics["gpu_positions"] == metrics["neural_positions"]
    assert metrics["gpu_batches"] > 1
    for a, b in zip(direct, remote, strict=True):
        assert a["value_target"] == b["value_target"]
        assert np.array_equal(a["observation"], b["observation"])
        assert np.array_equal(a["indices"], b["indices"])
        assert np.array_equal(a["visits"], b["visits"])


def test_batched_raw_evaluation_matches_sequential_reference():
    configure_torch(764)
    policy = DuoResidualActorCritic(16, 8, 1).eval()
    opponent = DuoActorCritic(16).eval()
    book = opening_book(2, seed=2563000)
    sequential = evaluate_pair(policy, opponent, book, seed=2563000)
    batched = evaluate_many(policy, opponent, book, simulations=0, opponent_simulations=0)
    assert batched["opening_pair_scores"] == sequential["opening_pair_scores"]
    assert [batched[k] for k in ("wins", "draws", "losses")] == [
        sequential[k] for k in ("wins", "draws", "losses")
    ]


@pytest.mark.parametrize("legacy_replay", (False, True))
def test_deep_resume_preserves_learning_state(tmp_path, legacy_replay):
    configure_torch(764)
    config = small_config()
    reference = tmp_path / "reference.pt"
    save_duo(reference, DuoActorCritic(16), config, 0)
    train(config, tmp_path / "full", reference)
    train(config, tmp_path / "interrupted", reference, stop_after_round=1)
    if legacy_replay:
        path = tmp_path / "interrupted/resume.pt"
        saved = torch.load(path, weights_only=False)
        saved["replay"] = replay_records(saved["replay"])
        saved.pop("resume_format_version", None)
        torch.save(saved, path)
    # A crash can leave an ahead-of-resume selected file. The atomic round bundle
    # owns selection as well as optimizer/model/replay state.
    save_duo(tmp_path / "interrupted/best.pt", DuoActorCritic(16), config, 999)
    train(config, tmp_path / "interrupted", reference, resume=True)
    a = load_duo(tmp_path / "full/final.pt").state_dict()
    b = load_duo(tmp_path / "interrupted/final.pt").state_dict()
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    a = load_duo(tmp_path / "full/best.pt").state_dict()
    b = load_duo(tmp_path / "interrupted/best.pt").state_dict()
    assert all(torch.equal(v, b[k]) for k, v in a.items())


def test_time_limited_run_selects_and_saves_a_real_checkpoint(tmp_path):
    configure_torch(764)
    config = replace(small_config(), max_seconds=0.001, eval_every=8)
    reference = tmp_path / "reference.pt"
    save_duo(reference, DuoActorCritic(16), config, 0)
    out = tmp_path / "limited"
    train(config, out, reference)
    import json

    completion = json.loads((out / "completed.json").read_text())
    assert completion["reason"] == "time_budget" and completion["games"] == 2
    selected = torch.load(out / "best.pt", weights_only=True)
    assert selected["update"] == 1
    assert (out / "final-validation.json").exists()


@pytest.mark.skipif(not DEFAULT_BINARY.exists(), reason="optional local Pentobi")
def test_resumable_pentobi_evaluation_guards_model_and_keeps_complete_pairs(tmp_path):
    from blokus_rl.duo_deep_evaluate import evaluate_pentobi

    configure_torch(764)
    model = DuoResidualActorCritic(16, 8, 1).eval()
    out = tmp_path / "pentobi.json"
    book = opening_book(1, seed=2561000)
    first = evaluate_pentobi(model, DEFAULT_BINARY, book, out, simulations=4)
    assert first["games"] == 2 and sum(first[k] for k in ("wins", "draws", "losses")) == 2
    recovered = evaluate_pentobi(model, DEFAULT_BINARY, book, out, simulations=4)
    assert recovered["games_detail"] == first["games_detail"]
    with torch.no_grad():
        model.pass_head.bias.add_(0.1)
    with pytest.raises(ValueError, match="configuration/model changed"):
        evaluate_pentobi(model, DEFAULT_BINARY, book, out, simulations=4)
