"""Bounded synthetic result-copy diagnostic; no games, weights or training changes.

Run inside an existing CUDA container. Compare the current production method to
host NumPy-view reuse and active-row copies, preserving synchronization and owned
outputs. Timings describe this copy boundary, not full game/search throughput.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
from blokus_rl.duo_cuda_graphs import GraphBucket


@torch.no_grad()
def cached_numpy(self, n, width):
    self.graph.replay()
    self.host_probabilities.copy_(self.probabilities, non_blocking=True)
    self.host_values.copy_(self.values, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    p = self.output_arrays[0][:n, :width].copy()
    v = self.output_arrays[1][:n].copy()
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise RuntimeError("nonfinite graphed inference")
    return p, v


@torch.no_grad()
def active_rows(self, n, width):
    self.graph.replay()
    views = self.output_views.get(n)
    if views is None:
        views = (
            self.host_probabilities[:n],
            self.probabilities[:n],
            self.host_values[:n],
            self.values[:n],
        )
        self.output_views[n] = views
    host_p, device_p, host_v, device_v = views
    host_p.copy_(device_p, non_blocking=True)
    host_v.copy_(device_v, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    p = self.output_arrays[0][:n, :width].copy()
    v = self.output_arrays[1][:n].copy()
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise RuntimeError("nonfinite graphed inference")
    return p, v


@torch.no_grad()
def threshold_rows(self, n, width):
    self.graph.replay()
    key = n if (self.batch - n) * (self.width + 1) * 4 >= 65536 else self.batch
    views = self.output_views.get(key)
    if views is None:
        views = (
            self.host_probabilities[:key],
            self.probabilities[:key],
            self.host_values[:key],
            self.values[:key],
        )
        self.output_views[key] = views
    host_p, device_p, host_v, device_v = views
    host_p.copy_(device_p, non_blocking=True)
    host_v.copy_(device_v, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    p = self.output_arrays[0][:n, :width].copy()
    v = self.output_arrays[1][:n].copy()
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise RuntimeError("nonfinite graphed inference")
    return p, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 300:
        raise ValueError("bounded diagnostic only")
    torch.set_num_threads(1)
    modes = {
        "baseline": GraphBucket.finish,
        "cached_numpy": cached_numpy,
        "active_rows": active_rows,
        "threshold_rows": threshold_rows,
    }
    cases = []
    started = time.monotonic()
    for batch, n, width, actual_width in [
        (8, 1, 64, 40),
        (32, 24, 256, 200),
        (128, 85, 1024, 900),
        (256, 129, 2048, 1500),
        (256, 256, 2048, 1500),
    ]:
        b = GraphBucket.__new__(GraphBucket)
        b.batch, b.width = batch, width
        b.probabilities = torch.arange(batch * width, device="cuda", dtype=torch.float32).reshape(
            batch, width
        ) / (batch * width)
        b.values = torch.linspace(-1, 1, batch, device="cuda")
        b.host_probabilities = torch.empty((batch, width), pin_memory=True)
        b.host_values = torch.empty(batch, pin_memory=True)
        b.output_arrays = (b.host_probabilities.numpy(), b.host_values.numpy())
        b.output_views = {batch: (b.host_probabilities, b.probabilities, b.host_values, b.values)}
        b.graph = SimpleNamespace(replay=lambda: None)
        expected = (
            b.probabilities[:n, :actual_width].cpu().numpy().copy(),
            b.values[:n].cpu().numpy().copy(),
        )
        # Compare all methods and prove that later host-buffer reuse cannot mutate returns.
        for fn in modes.values():
            p, v = fn(b, n, actual_width)
            assert np.array_equal(p, expected[0]) and np.array_equal(v, expected[1])
            b.host_probabilities.fill_(-999)
            b.host_values.fill_(-999)
            assert np.array_equal(p, expected[0]) and np.array_equal(v, expected[1])
            for _ in range(10):
                fn(b, n, actual_width)
        times = {k: [] for k in modes}
        for repeat in range(4):
            order = list(modes) if repeat % 2 == 0 else list(reversed(modes))
            for name in order:
                before = time.perf_counter()
                for _ in range(args.iterations):
                    modes[name](b, n, actual_width)
                times[name].append((time.perf_counter() - before) * 1e6 / args.iterations)
        cases.append(
            {
                "batch_bucket": batch,
                "active_rows": n,
                "width_bucket": width,
                "actual_width": actual_width,
                "baseline_copy_bytes": (batch * width + batch) * 4,
                "active_copy_bytes": (n * width + n) * 4,
                "microseconds_per_call": times,
                "median_microseconds": {k: statistics.median(v) for k, v in times.items()},
            }
        )
        del b
    print(
        json.dumps(
            {
                "passed": True,
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "seconds": time.monotonic() - started,
                "iterations_per_repeat": args.iterations,
                "repeats_reversed_order": 4,
                "cases": cases,
                "scope": "synthetic result-copy component under concurrent study load; no neural inference, games or training; no whole-workload gain established",
            }
        )
    )


if __name__ == "__main__":
    main()
