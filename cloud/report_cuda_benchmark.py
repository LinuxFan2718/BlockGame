"""Audit one finished/stopped CUDA Graph stage and summarize saved telemetry."""

import argparse
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
RESUME_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
VARIANTS = ("current-eager", "current-graph", "native-eager", "native-graph", "native-graph-wide")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def telemetry(run):
    phases = {}
    for line in (run / "phases.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["event"] not in phases.setdefault(row["phase"], {})
        phases[row["phase"]][row["event"]] = row["monotonic"]
    samples = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
    report = {}
    for name, span in phases.items():
        assert set(span) == {"start", "end"} and span["end"] >= span["start"]
        trim = 1.2 if span["end"] - span["start"] >= 3 else 0
        selected = [
            r
            for r in samples
            if span["start"] + trim <= r["monotonic"] <= span["end"] and "gpu" in r
        ]
        if not selected:
            report[name] = {"samples": 0}
            continue
        values = np.asarray([[float(v.strip()) for v in r["gpu"].split(",")] for r in selected])
        entry = {
            "samples": len(selected),
            "phase_seconds": span["end"] - span["start"],
            "gpu_busy_mean": float(values[:, 0].mean()),
            "gpu_busy_median": float(np.median(values[:, 0])),
            "gpu_busy_p95": float(np.quantile(values[:, 0], 0.95)),
            "memory_io_mean": float(values[:, 1].mean()),
            "vram_peak_mib": float(values[:, 2].max()),
            "vram_capacity_mib": float(values[0, 3]),
            "power_mean_watts": float(values[:, 4].mean()),
            "sm_clock_mean_mhz": float(values[:, 5].mean()),
        }
        cpu = [r for r in selected if "cpu_ns" in r]
        if len(cpu) > 1:
            entry["cpu_cores_mean"] = (
                (cpu[-1]["cpu_ns"] - cpu[0]["cpu_ns"])
                / 1e9
                / (cpu[-1]["monotonic"] - cpu[0]["monotonic"])
            )
        memory = [r["memory_bytes"] for r in selected if "memory_bytes" in r]
        if memory:
            entry["host_memory_peak_gib"] = max(memory) / 1024**3
        report[name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--apps", type=Path, required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--billing", type=Path, required=True)
    parser.add_argument("--gpu", choices=("l40s", "h100"), required=True)
    args = parser.parse_args()
    run = args.run
    execution = load(run / "execution.json")
    ledger = load(ROOT / f"runs/modal-cuda-graphs-{args.gpu}-r1-cost-ledger.jsonl")
    assert execution == ledger and execution["gpu"] == args.gpu
    assert not (run / "failure.json").exists()
    assert sha(ROOT / "cloud/modal_cuda_benchmark.py") == execution["runner_sha256"]
    for name, h in execution["payload_sha256"].items():
        assert sha(ROOT / "runs/duo-cuda-payload-r1" / name) == h
    apps = load(args.apps)
    assert all(int(a["tasks"]) == 0 for a in apps)
    assert any(a["app_id"] == args.app_id and a["state"] == "stopped" for a in apps)
    finish = load(run / "finished.json")
    hardware = load(run / "hardware.json")
    assert finish["model_sha256"] == MODEL_SHA and finish["source_resume_sha256"] == RESUME_SHA
    assert finish["gradient_updates"] == 0 and 0 < finish["seconds"] < 1800
    assert ("L40S" if args.gpu == "l40s" else "H100") in hardware["gpu"]
    build = load(run / "native-build.json")
    for name, h in build["optimized_source_sha256"].items():
        assert h == execution["payload_sha256"]["native/" + name]
    assert build["baseline_extension_sha256"] != build["optimized_extension_sha256"]
    native = load(run / "native-correctness.json")
    assert (
        native["exact"] and native["random_games"] == 32 and native["arbitrary_prior_roots"] == 12
    )
    checks = load(run / "correctness.json")
    assert [c["rows"] for c in checks] == [1, 9, 17, 33, 65, 127, 256]
    for check in checks:
        assert check["same_shape_capture_max_abs_error"] <= 1e-6
        assert (
            check["padding_max_abs_error"][0] <= 0.002 and check["padding_max_abs_error"][1] <= 0.02
        )
    micro = load(run / "micro.json")
    assert (
        micro["correctness"] == checks
        and micro["model_sha256"] == MODEL_SHA
        and micro["gradient_updates"] == 0
    )
    assert len(micro["cases"]) == 26
    assert micro["fixture_sha256"] == sha(run / "inputs.npz")
    for count in (32, 128, 512, 2048):
        for mode in ("eager", "graph"):
            cases = [
                r
                for r in micro["cases"]
                if r["rows"] == count and r["method"] == mode and not r.get("sustained")
            ]
            assert sorted(r["repeat"] for r in cases) == [1, 2, 3]
            for r in cases:
                assert (
                    16 <= r["iterations"] <= 20000
                    and r["wall_seconds"] > 0
                    and r["cuda_event_seconds"] > 0
                )
                assert (
                    abs(r["positions_per_second"] - count * r["iterations"] / r["wall_seconds"])
                    < 1e-6
                )
    order = [
        (repeat, variant)
        for repeat, variants in ((1, VARIANTS), (2, tuple(reversed(VARIANTS))))
        for variant in variants
    ]
    assert [(r["repeat"], r["variant"]) for r in finish["results"]] == order
    summaries = {}
    for row in finish["results"]:
        variant, repeat = row["variant"], row["repeat"]
        assert load(run / f"{repeat}-{variant}.json") == row
        assert row["model_sha256"] == MODEL_SHA and row["fixture_sha256"] == sha(run / "inputs.npz")
        assert row["games"] == 384 and row["warmup_games"] == 12 and row["gradient_updates"] == 0
        config = row["config"]
        for k, v in {
            "seed": 1069,
            "workers": 6,
            "games_per_round": 384,
            "full_simulations": 1024,
            "fast_simulations": 32,
            "fast_probability": 0.8,
            "leaf_batch": 8,
            "inference_batch": 256,
            "device": "cuda",
            "precision": "bfloat16",
            "live_games": 64 if "wide" in variant else 32,
        }.items():
            assert config[k] == v, (k, config[k])
        expected = (
            build["optimized_extension_sha256"]
            if variant.startswith("native")
            else build["baseline_extension_sha256"]
        )
        assert row["native_sha256"] == expected
        m = row["metrics"]
        assert m["simulations"] >= row["recorded_roots"] * 1024 > 0
        assert (m["simulations"] - row["recorded_roots"] * 1024) % 32 == 0
        assert m["gpu_positions"] == m["neural_positions"] > 0 and m["cache_hits"] == 0
        if "graph" in variant:
            before, after = row["graph_before"], row["graph_after"]
            assert before["graphs"] == after["graphs"] == 24
            assert (
                after["requested_positions"] - before["requested_positions"]
                == m["neural_positions"]
            )
            padded = after["padded_positions"] - before["padded_positions"]
            fallback = after["fallback_positions"] - before["fallback_positions"]
            physical = padded + fallback
            assert physical >= m["neural_positions"]
        else:
            physical = m["neural_positions"]
        summaries.setdefault(variant, []).append(
            {
                "repeat": repeat,
                "seconds": m["collect_seconds"],
                "simulations": m["simulations"],
                "neural_positions": m["neural_positions"],
                "physical_neural_positions": physical,
                "mean_gpu_batch": m["mean_gpu_batch"],
                "capture_seconds": row["graph_after"]["capture_seconds"]
                if row["graph_after"]
                else 0,
            }
        )
    aggregate = {}
    for variant, cases in summaries.items():
        seconds = sum(c["seconds"] for c in cases)
        positions = sum(c["neural_positions"] for c in cases)
        aggregate[variant] = {
            "cases": cases,
            "mean_seconds": seconds / 2,
            "games_per_second": 768 / seconds,
            "simulations_per_second": sum(c["simulations"] for c in cases) / seconds,
            "neural_positions_per_second": positions / seconds,
            "padding_factor": sum(c["physical_neural_positions"] for c in cases) / positions,
            "rate_estimated_usd_per_1000_games": execution["hourly_usd"]
            * seconds
            / 3600
            / 768
            * 1000,
        }
    costs = {}
    for row in load(args.billing):
        costs[row["object_id"]] = costs.get(row["object_id"], Decimal(0)) + Decimal(row["cost"])
    actual = costs[args.app_id]
    assert 0 < actual < Decimal("4.19642")
    stop = ROOT / Path((ROOT / "runs/duo-teacher-user-stopped-current.txt").read_text().strip())
    preserved = load(stop / "stop-verified.json")
    for name, h in preserved["preserved"].items():
        assert sha(ROOT / name) == h
    for branch, row in preserved["retained"].items():
        assert sha(stop / "duo-teacher-search-1066" / branch / "resume.pt") == row["resume_sha256"]
    report = {
        "verified": True,
        "gpu": args.gpu,
        "app_id": args.app_id,
        "run": str(run.resolve()),
        "collection": aggregate,
        "telemetry_by_phase": telemetry(run),
        "metered_usd": str(actual),
        "combined_reservation_usd": execution["combined_reserved_usd"],
        "model_sha256": MODEL_SHA,
        "source_resume_sha256": RESUME_SHA,
        "preserved_checkpoints_unchanged": True,
        "gradient_updates": 0,
        "micro_summary": {
            f"{n}-{mode}": statistics.median(
                r["positions_per_second"]
                for r in micro["cases"]
                if r["rows"] == n and r["method"] == mode and not r.get("sustained")
            )
            for n in (32, 128, 512, 2048)
            for mode in ("eager", "graph")
        },
    }
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / f"runs/cuda-graphs-{args.gpu}-verified.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "telemetry_by_phase"}, indent=2))


if __name__ == "__main__":
    main()
