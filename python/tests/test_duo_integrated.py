"""Integrated native leaves and bulk transport retain ordering and target semantics."""

from dataclasses import replace

import numpy as np
from blokus_rl._native import DuoGame, DuoSearch, duo_policy_indices
from blokus_rl.duo_env import PASS, STATE_SIZE, decode_decision, decode_indexed_decision
from blokus_rl.duo_indices import policy_indices
from blokus_rl.duo_inference import IndexedBlock, fill_blocks, slice_blocks
from blokus_rl.duo_residual import DuoResidualActorCritic
from blokus_rl.duo_shared import SharedSearchPool
from blokus_rl.learning import configure_torch
from test_duo_deep import small_config
from test_duo_speedups import assert_records_equal


def test_native_policy_indices_match_exhaustive_geometry_table():
    game = DuoGame()
    actions = [PASS]
    for action in range(PASS):
        try:
            game.describe(action)
        except ValueError:
            continue
        actions.append(action)
    for player in (0, 1):
        assert np.array_equal(duo_policy_indices(actions, player), policy_indices(actions, player))


def test_indexed_leaves_preserve_all_stats_under_arbitrary_priors():
    rng = np.random.default_rng(1070)
    game = DuoGame()
    for ply in range(30):
        if game.terminated:
            break
        if ply % 5 == 0:
            left, right = DuoSearch(game), DuoSearch(game)
            while left.completed < 64:
                a = left.request_decisions(min(8, 64 - left.completed))
                b = right.request_indexed_decisions(min(8, 64 - right.completed))
                assert len(a) == len(b)
                for (ta, da), (tb, db) in zip(a, b, strict=True):
                    oa, aa, ca = decode_decision(da)
                    ob, ab, indices = decode_indexed_decision(db)
                    assert ta == tb and np.array_equal(oa, ob) and np.array_equal(aa, ab)
                    assert np.array_equal(indices, policy_indices(aa, int(oa[630] == 0)))
                    assert len(db[2]) == len(ca) * 2  # Six geometry integers no longer cross.
                    priors, value = rng.random(len(aa)).tolist(), float(rng.uniform(-1, 1))
                    left.expand(ta, priors, value)
                    right.expand(tb, priors, value)
            assert left.stats() == right.stats()
        game.step(int(rng.choice(game.legal_actions())))


def test_bulk_staging_strided_sources_unequal_widths_dummy_rows_and_slicing():
    a = IndexedBlock(
        np.ones((2, STATE_SIZE), np.float32),
        np.asarray([[1, 2, 99], [3, 99, 99]], np.uint16),
        np.asarray([2, 1]),
    )
    b = IndexedBlock(
        np.full((1, STATE_SIZE), 2, np.float32),
        np.asarray([[4, 5, 6, 7]], np.uint16),
        np.asarray([4]),
    )
    obs, ids, mask = (
        np.empty((8, STATE_SIZE), np.float32),
        np.empty((8, 64), np.int64),
        np.empty((8, 64), bool),
    )
    assert fill_blocks([a, b], obs, ids, mask) == (3, 4)
    assert ids[:3, :4].tolist() == [[1, 2, 0, 0], [3, 0, 0, 0], [4, 5, 6, 7]]
    assert not obs[3:].any() and mask[3:, 0].all() and not mask[:, 4:].any()
    pieces = slice_blocks([a, b], 1, 3)
    assert [len(x) for x in pieces] == [1, 1]
    assert sum((x.records() for x in pieces), [])[0]["indices"].tolist() == [3]
    fill_blocks([a.slice(1, 2)], obs, ids, mask)
    assert not ids[:, 1:].any() and not mask[:, 1:].any()


def test_bulk_compact_ready_transport_preserves_complete_cpu_targets_across_rounds():
    configure_torch(1070)
    config = replace(small_config(), workers=2, live_games=2, games_per_round=4, batch_wait_ms=0)
    model = DuoResidualActorCritic(16, 8, 1).eval()
    results = []
    for bulk, compact, ready in ((False, False, False), (True, True, True)):
        with SharedSearchPool(
            config, bulk_inference=bulk, compact_native=compact, ready_batching=ready
        ) as pool:
            rows, scores, counters = pool.collect(model, 1, 4)
            pool.collect(model, 2, 4)
        assert counters["gpu_positions"] == counters["neural_positions"]
        if bulk:
            assert counters["bulk_requests"] > 0
        results.append((rows, scores, counters["simulations"]))
    assert results[0][1:] == results[1][1:]
    assert_records_equal(results[0][0], results[1][0])
