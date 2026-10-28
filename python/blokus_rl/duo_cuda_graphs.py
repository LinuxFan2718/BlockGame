"""Opt-in CUDA Graph evaluator for one frozen Duo model; not a training default.

Static CUDA tensors and pinned staging remain alive with each graph. Returned
NumPy arrays own their memory, so a later replay cannot overwrite a caller's
policy/value arrays. Device copies complete before the host consumes results.
"""

import bisect
import time

import numpy as np
import torch

from .duo_env import PASS, STATE_SIZE

BATCH_BUCKETS = (8, 16, 32, 64, 128, 256)
WIDTH_BUCKETS = (64, 256, 1024, 2048)


def bucket(value, choices):
    if value < 1:
        raise ValueError("positive graph dimension required")
    index = bisect.bisect_left(choices, value)
    return None if index == len(choices) else choices[index]


def fill_staging(records, observations, indices, mask, *, validate_indices=True):
    """CPU-only helper, also tested without a CUDA device."""
    n = len(records)
    if not 0 < n <= len(observations):
        raise ValueError("invalid graph batch length")
    width = max(len(r["indices"]) for r in records)
    if not 0 < width <= indices.shape[1]:
        raise ValueError("invalid graph action width")
    observations.fill(0)
    indices.fill(0)
    mask.fill(False)
    # Dummy padded rows must have a finite softmax as well.
    mask[:, 0] = True
    np.stack([r["observation"] for r in records], out=observations[:n])
    for i, row in enumerate(records):
        legal = np.asarray(row["indices"])
        if len(legal) == 0 or (validate_indices and (np.any(legal < 0) or np.any(legal > PASS))):
            raise ValueError("invalid legal policy index")
        indices[i, : len(legal)] = legal
        mask[i, : len(legal)] = True
    return n, width


class GraphBucket:
    def __init__(self, model, batch, width, *, precision="bfloat16"):
        if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
            raise ValueError("CUDA model and device required")
        self.batch, self.width = batch, width
        self.model, self.precision = model, precision
        self.host_observations = torch.zeros((batch, STATE_SIZE), pin_memory=True)
        self.host_indices = torch.zeros((batch, width), dtype=torch.int64, pin_memory=True)
        self.host_mask = torch.zeros((batch, width), dtype=torch.bool, pin_memory=True)
        self.host_mask[:, 0] = True
        self.host_probabilities = torch.empty((batch, width), pin_memory=True)
        self.host_values = torch.empty(batch, pin_memory=True)
        self.observations = self.host_observations.to("cuda")
        self.indices = self.host_indices.to("cuda")
        self.mask = self.host_mask.to("cuda")
        self.host_arrays = tuple(
            t.numpy() for t in (self.host_observations, self.host_indices, self.host_mask)
        )
        started = time.monotonic()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.probabilities, self.values = self.forward()
        torch.cuda.synchronize()
        self.setup_seconds = time.monotonic() - started

    def forward(self):
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.precision == "bfloat16", cache_enabled=False
        ):
            scores, values = self.model(self.observations)
            logits = scores.gather(-1, self.indices).float().masked_fill(~self.mask, -torch.inf)
            return torch.softmax(logits, -1), values.flatten().clamp(-1, 1).float()

    @torch.no_grad()
    def stage(self, records):
        # Production Rust/table producers already validate these indices. Avoid
        # re-scanning every legal list inside this latency-sensitive boundary.
        n, width = fill_staging(records, *self.host_arrays, validate_indices=False)
        self.observations.copy_(self.host_observations, non_blocking=True)
        self.indices.copy_(self.host_indices, non_blocking=True)
        self.mask.copy_(self.host_mask, non_blocking=True)
        return n, width

    @torch.no_grad()
    def evaluate(self, records):
        n, width = self.stage(records)
        self.graph.replay()
        self.host_probabilities.copy_(self.probabilities, non_blocking=True)
        self.host_values.copy_(self.values, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        p = self.host_probabilities[:n, :width].numpy().copy()
        v = self.host_values[:n].numpy().copy()
        if not np.isfinite(p).all() or not np.isfinite(v).all():
            raise RuntimeError("nonfinite graphed inference")
        return p, v


class FrozenGraphEvaluator:
    def __init__(self, model, *, fallback, precision="bfloat16"):
        self.model, self.precision, self.fallback = model, precision, fallback
        self.graphs = {}
        self.calls = self.fallback_calls = self.positions = self.padded_positions = 0
        self.fallback_positions = 0
        self.shapes = {}

    def prepare(self):
        for n in BATCH_BUCKETS:
            for width in WIDTH_BUCKETS:
                self.graphs[n, width] = GraphBucket(self.model, n, width, precision=self.precision)

    def __call__(self, records):
        n = len(records)
        width = max(len(r["indices"]) for r in records)
        batch_bucket, width_bucket = bucket(n, BATCH_BUCKETS), bucket(width, WIDTH_BUCKETS)
        self.calls += 1
        self.positions += n
        if batch_bucket is None or width_bucket is None:
            self.fallback_calls += 1
            self.fallback_positions += n
            return self.fallback(self.model, records, self.precision)
        key = batch_bucket, width_bucket
        if key not in self.graphs:
            self.graphs[key] = GraphBucket(self.model, *key, precision=self.precision)
        self.padded_positions += batch_bucket
        self.shapes[str(key)] = self.shapes.get(str(key), 0) + 1
        return self.graphs[key].evaluate(records)

    def counters(self):
        return {
            "calls": self.calls,
            "fallback_calls": self.fallback_calls,
            "fallback_positions": self.fallback_positions,
            "requested_positions": self.positions,
            "padded_positions": self.padded_positions,
            "shapes": dict(self.shapes),
            "capture_seconds": sum(g.setup_seconds for g in self.graphs.values()),
            "graphs": len(self.graphs),
        }
