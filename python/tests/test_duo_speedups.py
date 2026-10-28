"""Correctness and recovery boundaries for inference and replay optimizations."""

from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
import torch
from blokus_rl._native import DuoGame
from blokus_rl.duo_env import CELLS, PASS, decision
from blokus_rl.duo_indices import action_index_table, policy_indices
from blokus_rl.duo_inference import FrozenEvaluationCache, InferenceBuffer
from blokus_rl.duo_replay import pack_replay, replay_length, replay_records
from blokus_rl.duo_residual import DuoResidualActorCritic, candidate_indices_numpy
from blokus_rl.duo_shared import SharedSearchPool
from blokus_rl.learning import configure_torch
from test_duo_deep import small_config


def test_action_lookup_exhaustive_both_observers_and_pass():
    game = DuoGame()
    actions, rows = [], []
    for action in range(PASS):
        try:
            piece, _, _, cells = game.describe(action)
        except ValueError:
            continue
        row = np.full(6, CELLS, dtype=np.int16)
        row[0], row[1 : 1 + len(cells)] = piece, cells
        actions.append(action)
        rows.append(row)
    rows.append(np.asarray([21, CELLS, CELLS, CELLS, CELLS, CELLS], np.int16))
    actions.append(PASS)
    actions, candidates = np.asarray(actions), np.asarray(rows)
    for player in (0, 1):
        relative = candidates.copy()
        if player:
            relative[:, 1:] = np.where(relative[:, 1:] < CELLS, CELLS - 1 - relative[:, 1:], CELLS)
        assert np.array_equal(policy_indices(actions, player), candidate_indices_numpy(relative))
    assert action_index_table().nbytes == 2 * (PASS + 1) * 2
    assert not action_index_table().flags.writeable


def test_shared_buffer_round_trip_copies_results_and_unlinks():
    owner = InferenceBuffer(3)
    name = owner.memory.name
    reader = InferenceBuffer(**owner.descriptor())
    try:
        observation, actions, _ = decision(DuoGame())
        indices = policy_indices(actions, 0)
        records = [{"observation": observation, "indices": indices}]
        reader.write_request(records)
        actual = owner.read_request(1)
        assert np.array_equal(actual[0]["observation"], observation)
        assert np.array_equal(actual[0]["indices"], indices)
        probabilities = np.full((1, len(indices)), 1 / len(indices), np.float32)
        width = owner.write_response(probabilities, np.asarray([0.25], np.float32))
        first, values = reader.read_response(1, width)
        owner.write_response(np.zeros_like(probabilities), np.zeros(1, np.float32))
        assert np.array_equal(first, probabilities) and values[0] == 0.25
        del actual
    finally:
        reader.close()
        owner.close()
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_cache_keys_include_inventory_scores_passes_player_and_action_order():
    observation, actions, _ = decision(DuoGame())
    base = {"observation": observation, "indices": policy_indices(actions, 0)}
    evaluated = []

    def evaluate(rows):
        evaluated.extend(rows)
        p = np.zeros((len(rows), max(len(r["indices"]) for r in rows)), np.float32)
        for i, row in enumerate(rows):
            p[i, : len(row["indices"])] = 1 / len(row["indices"])
        return p, np.asarray([r["observation"][392] for r in rows], np.float32)

    cache = FrozenEvaluationCache(3)
    p, v, counters = cache.evaluate([base, base], evaluate)
    assert len(evaluated) == 1 and counters["cache_hits"] == counters["cache_misses"] == 1
    assert np.array_equal(p[0], p[1]) and v[0] == v[1]
    _, _, counters = cache.evaluate([base], evaluate)
    assert counters["cache_hits"] == 1 and counters["cache_misses"] == 0
    for field in (392, 630, 631, 633):
        changed = {"observation": observation.copy(), "indices": base["indices"]}
        changed["observation"][field] += 0.25
        _, _, counters = cache.evaluate([changed], evaluate)
        assert counters["cache_misses"] == 1
        assert len(cache.entries) <= 3
    _, _, counters = cache.evaluate([base | {"indices": base["indices"][::-1]}], evaluate)
    assert counters["cache_misses"] == 1
    # A new model generation must get a fresh cache, even for identical inputs.
    _, _, counters = FrozenEvaluationCache(3).evaluate([base], evaluate)
    assert counters["cache_misses"] == 1


def assert_records_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        for key in ("observation", "indices", "visits"):
            assert np.array_equal(a[key], b[key])
        assert a["value_target"] == b["value_target"]


def test_queue_shared_and_cached_collection_preserve_exact_cpu_targets():
    configure_torch(764)
    config = replace(small_config(), workers=2, live_games=2, games_per_round=4)
    model = DuoResidualActorCritic(16, 8, 1).eval()
    results = []
    for transport, cache in (("queue", 0), ("shared_memory", 0), ("shared_memory", 128)):
        with SharedSearchPool(config, transport=transport, cache_entries=cache) as pool:
            records, scores, metrics = pool.collect(model, 1, 4)
            # Each call uses a fresh cache; no entry survives a model update.
            _, _, next_metrics = pool.collect(model, 2, 4)
        assert metrics["gpu_positions"] + metrics["cache_hits"] == metrics["neural_positions"]
        assert next_metrics["cache_misses"] > 0
        if cache:
            assert metrics["cache_hits"] > 0
        results.append((records, scores, metrics["simulations"]))
    for records, scores, simulations in results[1:]:
        assert scores == results[0][1] and simulations == results[0][2]
        assert_records_equal(records, results[0][0])


def replay_fixture():
    game = DuoGame()
    rng = np.random.default_rng(2026)
    records = []
    while not game.terminated:
        observation, actions, _ = decision(game)
        records.append(
            {
                "observation": observation,
                "indices": policy_indices(actions, game.current_player),
                "visits": np.arange(1, len(actions) + 1, dtype=np.uint16),
                "value_target": (len(records) % 3) - 1.0,
            }
        )
        game.step(int(rng.choice(actions)))
    return records


def test_packed_replay_exact_roundtrip_legacy_and_corruption(tmp_path):
    records = replay_fixture()
    assert replay_records(records) is records
    saved = pack_replay(records)
    path = tmp_path / "replay.pt"
    torch.save(saved, path)
    loaded = torch.load(path, weights_only=True)
    assert replay_length(loaded) == len(records)
    assert_records_equal(records, replay_records(loaded))
    assert replay_records(pack_replay([])) == []
    bad = dict(loaded)
    bad["offsets"] = loaded["offsets"].clone()
    bad["offsets"][-1] -= 1
    with pytest.raises(ValueError, match="invalid packed"):
        replay_records(bad)
    records[0]["observation"][392] = 0.25
    with pytest.raises(ValueError, match="nonbinary"):
        pack_replay(records)
