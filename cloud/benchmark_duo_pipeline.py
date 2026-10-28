"""Local evaluation of further Duo speed ideas; no training or remote calls.

The prototypes here deliberately do not alter the production inference path.
Component timings on a Mac are not estimates of L40S end-to-end speedup.
"""

import argparse
import copy
import cProfile
import hashlib
import io
import json
import platform
import pstats
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from blokus_rl.duo_deep import DeepConfig
from blokus_rl.duo_learning import load_duo
from blokus_rl.duo_shared import collect, evaluate_indexed, pack_indexed
from blokus_rl.learning import configure_torch


def measure(cases, iterations, repeats=7):
    """Warm all paths, then alternate their order; retain every timing."""
    results = {key: [] for key in cases}
    for function in cases.values():
        for _ in range(3):
            function()
    for repeat in range(repeats):
        order = list(cases) if repeat % 2 == 0 else list(reversed(cases))
        for key in order:
            started = time.perf_counter()
            for _ in range(iterations):
                cases[key]()
            results[key].append((time.perf_counter() - started) * 1e6 / iterations)
    return {
        key: {"microseconds": values, "median_microseconds": statistics.median(values)}
        for key, values in results.items()
    }


class ReusablePack:
    """Prototype reusable host staging, including masks and Tensor wrappers."""

    def __init__(self, records):
        self.n = len(records)
        self.width = max(len(r["indices"]) for r in records)
        self.observations = np.empty((self.n, 634), np.float32)
        self.indices = np.zeros((self.n, self.width), np.int64)
        self.mask = np.zeros_like(self.indices, dtype=bool)
        self.positions = np.arange(self.width)[None, :]
        self.tensors = tuple(
            torch.from_numpy(x) for x in (self.observations, self.indices, self.mask)
        )

    def __call__(self, records):
        counts = np.fromiter((len(r["indices"]) for r in records), np.int64, count=self.n)
        np.stack([r["observation"] for r in records], out=self.observations)
        self.indices.fill(0)
        for i, row in enumerate(records):
            self.indices[i, : len(row["indices"])] = row["indices"]
        np.less(self.positions, counts[:, None], out=self.mask)
        return self.tensors


def packing(records):
    output = {}
    for size in (32, 128):
        rows = [records[i % len(records)] for i in range(size)]
        expected = pack_indexed(rows, "cpu")
        reusable = ReusablePack(rows)
        actual = reusable(rows)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
        # Four already-populated worker buffers, rather than one dict per leaf.
        # Match the wide, strided index layout of the real shared buffers.
        source_observations = expected[0].numpy().copy()
        source_indices = np.zeros((size, 17837), np.uint16)
        source_indices[:, : reusable.width] = expected[1].numpy()
        counts = np.asarray([len(r["indices"]) for r in rows], np.uint16)

        def bulk():
            for start in range(0, size, 32):
                end = min(size, start + 32)
                np.copyto(reusable.observations[start:end], source_observations[start:end])
                np.copyto(reusable.indices[start:end], source_indices[start:end, : reusable.width])
            np.less(reusable.positions, counts[:, None], out=reusable.mask)
            return reusable.tensors

        actual = bulk()
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
        output[size] = {
            "max_legal_actions": expected[1].shape[-1],
            "exact": True,
            "timings": measure(
                {
                    "current": lambda: pack_indexed(rows, "cpu"),
                    "reused_staging": lambda: reusable(rows),
                    "bulk_request_blocks": bulk,
                },
                400,
            ),
        }
    return output


def batch_efficiency(model, records):
    rows = [records[i % len(records)] for i in range(256)]

    def evaluate(size):
        outputs = []
        for start in range(0, len(rows), size):
            subset = rows[start : start + size]
            p, v = evaluate_indexed(model, subset)
            outputs.extend(
                (p[i, : len(row["indices"])].copy(), float(v[i])) for i, row in enumerate(subset)
            )
        return outputs

    expected = evaluate(32)
    for size in (8, 64, 128):
        actual = evaluate(size)
        assert all(
            np.allclose(a, c, rtol=2e-5, atol=2e-6) and abs(b - d) <= 2e-6
            for (a, b), (c, d) in zip(expected, actual, strict=True)
        )
    return {
        "positions_per_case": 256,
        "outputs_close": True,
        "timings": measure(
            {str(size): lambda n=size: evaluate(n) for size in (8, 32, 64, 128)}, 2, 5
        ),
    }


def responses(records):
    """Compare current row copies/repacking with direct batched view copies."""
    rows = [records[i % len(records)] for i in range(128)]
    counts = [len(r["indices"]) for r in rows]
    rng = np.random.default_rng(1068)
    probabilities = rng.random((128, max(counts)), dtype=np.float32)
    values = rng.random(128, dtype=np.float32)
    groups = [(offset, min(offset + 32, 128)) for offset in range(0, 128, 32)]
    outputs = [
        (
            np.empty((end - start, max(counts[start:end])), np.float32),
            np.empty(end - start, np.float32),
        )
        for start, end in groups
    ]

    def current():
        p = [probabilities[i, :n].copy() for i, n in enumerate(counts)]
        v = values.tolist()
        for (start, end), (outp, outv) in zip(groups, outputs, strict=True):
            padded = np.zeros_like(outp)
            for i, value in enumerate(p[start:end]):
                padded[i, : len(value)] = value
            outp[:] = padded
            outv[:] = np.asarray(v[start:end], np.float32)
        return outputs

    def direct():
        for (start, end), (outp, outv) in zip(groups, outputs, strict=True):
            # The current interface promises zeros for padding; retain that too.
            np.copyto(outp, probabilities[start:end, : outp.shape[-1]])
            for i, n in enumerate(counts[start:end]):
                outp[i, n:] = 0
            np.copyto(outv, values[start:end])
        return outputs

    expected = [(a.copy(), b.copy()) for a, b in current()]
    actual = direct()
    assert all(
        np.array_equal(a, c) and np.array_equal(b, d)
        for (a, b), (c, d) in zip(expected, actual, strict=True)
    )
    return {"exact": True, "timings": measure({"current": current, "direct": direct}, 700)}


def neural(model, records):
    rows = [records[i % len(records)] for i in range(32)]
    observations, indices, mask = pack_indexed(rows, "cpu")
    channels_last = copy.deepcopy(model).to(memory_format=torch.channels_last).eval()

    def forward(network, mode):
        with mode():
            scores, value = network(observations)
            probabilities = scores.gather(-1, indices).float().masked_fill(~mask, -torch.inf)
            return torch.softmax(probabilities, -1).numpy(), value.clamp(-1, 1).numpy()

    expected = forward(model, torch.no_grad)
    cases = {
        "no_grad": lambda: forward(model, torch.no_grad),
        "inference_mode": lambda: forward(model, torch.inference_mode),
        "channels_last_inference": lambda: forward(channels_last, torch.inference_mode),
    }
    differences = {}
    for name, function in cases.items():
        actual = function()
        differences[name] = max(
            float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)
        )
        assert all(
            np.allclose(a, b, atol=2e-6, rtol=2e-5) for a, b in zip(expected, actual, strict=True)
        )
    return {"batch": 32, "max_absolute_error": differences, "timings": measure(cases, 15, 7)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    configure_torch(1068)
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    model = load_duo(args.model).cpu().eval()
    config = replace(
        DeepConfig(),
        workers=1,
        live_games=4,
        full_simulations=128,
        inference_batch=128,
        games_per_round=8,
        seed=1068,
        device="cpu",
        architecture="conv",
    )
    sampled = []
    calls = []

    def evaluator(rows):
        if len(sampled) < 2048:
            sampled.extend(
                {"observation": r["observation"].copy(), "indices": r["indices"].copy()}
                for r in rows[: 2048 - len(sampled)]
            )
        started = time.perf_counter()
        result = evaluate_indexed(model, rows)
        calls.append((len(rows), time.perf_counter() - started))
        return result

    # Warm lookup construction and kernels before profiling the real search path.
    collect(config, 1068000, 1, lambda rows: evaluate_indexed(model, rows))
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    records, scores, metrics = collect(config, 1068001, 8, evaluator)
    profile.disable()
    seconds = time.perf_counter() - started
    profile.dump_stats(args.out / "collection.prof")
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(45)
    (args.out / "collection-profile.txt").write_text(stream.getvalue())
    np.savez_compressed(
        args.out / "sampled-leaves.npz",
        observations=np.stack([r["observation"] for r in sampled]),
        indices=np.concatenate([r["indices"] for r in sampled]),
        offsets=np.cumsum([0] + [len(r["indices"]) for r in sampled]),
    )
    print(
        json.dumps(
            {
                "stage": "profile_complete",
                "seconds": seconds,
                "games": len(scores),
                "sampled_leaves": len(sampled),
            }
        ),
        flush=True,
    )
    input_bytes = sum(634 * 4 + len(r["indices"]) * 16 for r in sampled)
    candidate_bytes = sum(len(r["indices"]) * 12 for r in sampled)
    output = {
        "source_commit": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "model_sha256": before,
        "gradient_updates": 0,
        "games": len(scores),
        "profile": {
            "seconds": seconds,
            "metrics": metrics,
            "retained_roots": len(records),
            "inference_calls": len(calls),
            "mean_inference_batch": statistics.mean(n for n, _ in calls),
            "inference_seconds": sum(t for _, t in calls),
        },
        "leaf_encoding": {
            "sampled_leaves": len(sampled),
            "current_bytes": input_bytes,
            "candidate_bytes": candidate_bytes,
            "candidate_fraction": candidate_bytes / input_bytes,
        },
        "packing": packing(sampled),
        "responses": responses(sampled),
        "neural": neural(model, sampled),
        "batch_efficiency": batch_efficiency(model, sampled),
        "limitations": "CPU components and CPU profile only; no CUDA, IPC or end-to-end Modal gain measured.",
    }
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == before
    (args.out / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
