"""CPU contracts for optional CUDA Graph staging; device checks run remotely."""

import numpy as np
import pytest
from blokus_rl.duo_cuda_graphs import BATCH_BUCKETS, bucket, fill_staging
from blokus_rl.duo_env import PASS, STATE_SIZE


def test_buckets_bound_padding_and_refuse_unsupported_shapes():
    assert bucket(1, BATCH_BUCKETS) == 8
    assert bucket(65, BATCH_BUCKETS) == 128
    assert bucket(256, BATCH_BUCKETS) == 256
    assert bucket(257, BATCH_BUCKETS) is None
    with pytest.raises(ValueError):
        bucket(0, BATCH_BUCKETS)


def test_staging_clears_previous_data_and_keeps_dummy_rows_finite():
    obs = np.full((8, STATE_SIZE), 5, np.float32)
    indices = np.full((8, 64), 9, np.int64)
    mask = np.ones((8, 64), bool)
    rows = [
        {"observation": np.ones(STATE_SIZE, np.float32), "indices": np.array([0, PASS], np.uint16)}
    ]
    assert fill_staging(rows, obs, indices, mask) == (1, 2)
    assert np.all(obs[0] == 1) and not obs[1:].any()
    assert indices[0, 1] == PASS and not indices[1:].any()
    assert mask[0, :2].all() and not mask[0, 2:].any()
    assert mask[1:, 0].all() and not mask[1:, 1:].any()
    rows[0]["indices"] = np.array([17], np.uint16)
    fill_staging(rows, obs, indices, mask)
    assert not mask[:, 1:].any() and not indices[:, 1:].any()
    rows[0]["indices"] = np.array([PASS + 1], np.int64)
    with pytest.raises(ValueError):
        fill_staging(rows, obs, indices, mask)
