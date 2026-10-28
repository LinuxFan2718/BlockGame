"""The result-copy fast path preserves output values, ownership and padding rules."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from blokus_rl.duo_cuda_graphs import GraphBucket


@pytest.mark.parametrize(
    "batch,n,width,actual_width",
    [
        (8, 1, 64, 40),
        (32, 24, 256, 200),
        (128, 85, 1024, 900),
        (256, 129, 2048, 1500),
        (256, 256, 2048, 1500),
    ],
)
def test_graph_result_copies_preserve_values_and_owned_outputs(
    monkeypatch, batch, n, width, actual_width
):
    # CPU stand-ins exercise exactly the production copy/view method. The real
    # CUDA transfer/synchronization path is checked by the separate GPU probe.
    b = GraphBucket.__new__(GraphBucket)
    b.batch, b.width = batch, width
    b.probabilities = torch.arange(batch * width, dtype=torch.float32).reshape(batch, width)
    b.values = torch.linspace(-1, 1, batch)
    b.host_probabilities = torch.full((batch, width), -999.0)
    b.host_values = torch.full((batch,), -999.0)
    b.output_arrays = b.host_probabilities.numpy(), b.host_values.numpy()
    b.output_views = {batch: (b.host_probabilities, b.probabilities, b.host_values, b.values)}
    synced = []
    b.graph = SimpleNamespace(replay=lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: synced.append(True)),
    )
    expected_p = b.probabilities[:n, :actual_width].numpy().copy()
    expected_v = b.values[:n].numpy().copy()
    p, v = b.finish(n, actual_width)
    assert synced == [True]
    assert np.array_equal(p, expected_p) and np.array_equal(v, expected_v)
    assert p.flags.owndata and v.flags.owndata
    if (batch - n) * (width + 1) * 4 >= 65536:
        assert torch.all(b.host_probabilities[n:] == -999)
        assert torch.all(b.host_values[n:] == -999)
    else:
        assert torch.equal(b.host_probabilities, b.probabilities)
    # A smaller call after larger data must not expose stale rows/columns or
    # mutate arrays returned to a previous caller.
    b.probabilities.add_(1)
    b.values.add_(1)
    next_p, next_v = b.finish(1, 3)
    assert np.array_equal(next_p, b.probabilities[:1, :3].numpy())
    assert np.array_equal(next_v, b.values[:1].numpy())
    assert np.array_equal(p, expected_p) and np.array_equal(v, expected_v)
    b.probabilities[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        b.finish(1, 3)
