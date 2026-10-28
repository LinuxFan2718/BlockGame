"""Frozen CUDA Graph correctness, inference ceiling and actual collection."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from blokus_rl.duo_cuda_graphs import WIDTH_BUCKETS, FrozenGraphEvaluator, GraphBucket, bucket
from blokus_rl.duo_deep import DeepConfig
from blokus_rl.duo_learning import load_duo
from blokus_rl.duo_shared import SharedSearchPool, evaluate_indexed
from blokus_rl.learning import configure_torch


def phase(path, name, event):
    with (path / "phases.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"monotonic": time.monotonic(), "phase": name, "event": event}) + "\n"
        )


def fixture(path):
    data = np.load(path)
    return [
        {
            "observation": data["observations"][i],
            "indices": data["indices"][data["offsets"][i] : data["offsets"][i + 1]],
        }
        for i in range(len(data["observations"]))
    ]


@torch.no_grad()
def correctness(model, rows):
    graph = FrozenGraphEvaluator(model, fallback=evaluate_indexed)
    graph.prepare()
    checks = []
    for count in (1, 9, 17, 33, 65, 127, 256):
        subset = rows[:count]
        expected = evaluate_indexed(model, subset, "bfloat16")
        actual = graph(subset)
        errors = [float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)]
        # Same padded shapes isolate graph capture from normal bf16 batch-shape effects.
        key = (
            bucket(count, tuple(sorted(set(n for n, _ in graph.graphs)))),
            bucket(max(len(r["indices"]) for r in subset), WIDTH_BUCKETS),
        )
        item = graph.graphs[key]
        item.stage(subset)
        padded_eager = tuple(t.cpu().numpy() for t in item.forward())
        item.graph.replay()
        padded_graph = tuple(t.cpu().numpy() for t in (item.probabilities, item.values))
        capture_error = max(
            float(np.max(np.abs(a - b))) for a, b in zip(padded_eager, padded_graph, strict=True)
        )
        assert capture_error <= 1e-6, capture_error
        assert errors[0] <= 0.002 and errors[1] <= 0.02, errors
        assert np.allclose(actual[0].sum(-1), 1, atol=2e-6)
        saved = tuple(a.copy() for a in actual)
        graph(rows[-count:])
        assert all(np.array_equal(a, b) for a, b in zip(saved, actual, strict=True))
        checks.append(
            {
                "rows": count,
                "padding_max_abs_error": errors,
                "same_shape_capture_max_abs_error": capture_error,
            }
        )
    return checks


@torch.no_grad()
def gpu_time(function, iterations):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.monotonic()
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return {
        "iterations": iterations,
        "wall_seconds": time.monotonic() - wall,
        "cuda_event_seconds": start.elapsed_time(end) / 1000,
    }


def micro(model, rows, out):
    phase(out, "correctness", "start")
    checks = correctness(model, rows)
    phase(out, "correctness", "end")
    (out / "correctness.json").write_text(json.dumps(checks, indent=2) + "\n")
    cases = []
    for count in (32, 128, 512, 2048):
        selected = rows[:count]
        width = max(len(r["indices"]) for r in selected)
        item = GraphBucket(model, count, width)
        item.stage(selected)
        torch.cuda.synchronize()
        methods = {"eager": item.forward, "graph": item.graph.replay}
        for repeat in range(3):
            order = tuple(methods) if repeat % 2 == 0 else tuple(reversed(methods))
            for name in order:
                tag = f"micro-{count}-{name}-{repeat + 1}"
                calibration = gpu_time(methods[name], 10)
                iterations = max(16, min(20000, int(0.7 * 10 / calibration["wall_seconds"])))
                phase(out, tag, "start")
                result = gpu_time(methods[name], iterations)
                phase(out, tag, "end")
                result.update(
                    {
                        "rows": count,
                        "width": width,
                        "method": name,
                        "repeat": repeat + 1,
                        "positions_per_second": count * iterations / result["wall_seconds"],
                        "capture_seconds": item.setup_seconds,
                    }
                )
                cases.append(result)
                print(
                    json.dumps(
                        {"phase": tag, "positions_per_second": result["positions_per_second"]}
                    ),
                    flush=True,
                )
        if count == 2048:
            for name in ("eager", "graph"):
                tag = "sustained-2048-" + name
                phase(out, tag, "start")
                started, iterations = time.monotonic(), 0
                with torch.no_grad():
                    while time.monotonic() - started < 10:
                        for _ in range(50):
                            methods[name]()
                        iterations += 50
                        torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                phase(out, tag, "end")
                cases.append(
                    {
                        "sustained": True,
                        "rows": count,
                        "method": name,
                        "iterations": iterations,
                        "wall_seconds": elapsed,
                        "positions_per_second": count * iterations / elapsed,
                    }
                )
    return {"cases": cases, "correctness": checks, "gradient_updates": 0}


def collection(model, args, rows):
    import blokus_rl._native as native
    import blokus_rl.duo_shared as shared

    graphed = "graph" in args.variant
    config = DeepConfig(
        architecture="conv",
        seed=1069,
        rounds=1,
        games_per_round=384,
        workers=6,
        live_games=64 if "wide" in args.variant else 32,
        full_simulations=1024,
        fast_simulations=32,
        device="cuda",
        precision="bfloat16",
    )
    graph = None
    if graphed:
        phase(args.out.parent, args.tag + "-capture", "start")
        graph = FrozenGraphEvaluator(model, fallback=evaluate_indexed)
        graph.prepare()
        phase(args.out.parent, args.tag + "-capture", "end")
        shared.evaluate_indexed = lambda _model, records, _precision: graph(records)
    with SharedSearchPool(config, transport="shared_memory", cache_entries=0) as pool:
        phase(args.out.parent, args.tag + "-warmup", "start")
        _, _, warmup = pool.collect(model, 0, 12)
        phase(args.out.parent, args.tag + "-warmup", "end")
        before = graph.counters() if graph else None
        phase(args.out.parent, args.tag, "start")
        records, scores, metrics = pool.collect(model, args.repeat, 384)
        phase(args.out.parent, args.tag, "end")
    assert len(scores) == 384 and all(int(r["visits"].sum()) == 1024 for r in records)
    assert metrics["gpu_positions"] == metrics["neural_positions"]
    digest = hashlib.sha256()
    for row in records:
        for key, dtype in (
            ("observation", np.float32),
            ("indices", np.uint16),
            ("visits", np.uint16),
        ):
            digest.update(np.asarray(row[key], dtype=dtype).tobytes())
        digest.update(np.asarray([row["value_target"]], np.float64).tobytes())
    return {
        "variant": args.variant,
        "repeat": args.repeat,
        "config": asdict(config),
        "games": len(scores),
        "recorded_roots": len(records),
        "record_sha256": digest.hexdigest(),
        "metrics": metrics,
        "warmup_metrics": warmup,
        "warmup_games": 12,
        "graph_before": before,
        "graph_after": graph.counters() if graph else None,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "gradient_updates": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="micro")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tag", default="micro")
    args = parser.parse_args()
    configure_torch(1069)
    torch.set_float32_matmul_precision("high")
    model = load_duo(args.model).cuda().eval()
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    rows = fixture(args.fixture)
    started = time.monotonic()
    result = (
        micro(model, rows, args.out.parent)
        if args.variant == "micro"
        else collection(model, args, rows)
    )
    result.update(
        {
            "model_sha256": before,
            "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "total_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == before
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "case": args.tag,
                "seconds": result["total_seconds"],
                "collection_seconds": result.get("metrics", {}).get("collect_seconds"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
