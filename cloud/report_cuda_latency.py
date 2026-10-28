"""Audit the optional final H100 latency diagnostic after stopped compute."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

from report_cuda_benchmark import MODEL_SHA, ROOT, load, sha, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--apps", type=Path, required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--billing", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    ledger = load(ROOT / "runs/modal-cuda-latency-cost-ledger.jsonl")
    assert load(run / "execution.json") == ledger
    assert sha(ROOT / "cloud/modal_cuda_latency.py") == ledger["runner_sha256"]
    assert sha(ROOT / "cloud/duo_cuda_layout_probe.py") == ledger["helper_sha256"]
    assert sha(run / "inputs.npz") == ledger["fixture_sha256"]
    assert not (run / "failure.json").exists()
    apps = load(args.apps)
    assert all(int(a["tasks"]) == 0 for a in apps)
    assert any(a["app_id"] == args.app_id and a["state"] == "stopped" for a in apps)
    assert "H100" in load(run / "hardware.json")["gpu"]
    h100 = load(ROOT / "runs/cuda-graphs-h100-verified.json")
    expected_native = load(Path(h100["run"]) / "native-build.json")["optimized_extension_sha256"]
    assert sha(Path(h100["run"]) / "inputs.npz") == ledger["fixture_sha256"]
    finished = load(run / "finished.json")
    assert (
        finished["gradient_updates"] == 0
        and finished["model_sha256"] == MODEL_SHA
        and 0 < finished["seconds"] < 780
    )
    layout_check = load(run / "layout-check.json")
    assert layout_check["weights_exact"] and layout_check["model_sha256"] == MODEL_SHA
    assert [r["rows"] for r in layout_check["checks"]] == [1, 33, 128, 256]
    settings = ((1.0, "nchw"), (0.1, "nchw"), (0.0, "nchw"))
    if finished["channels_last_enabled"]:
        assert layout_check["passed"]
        assert not (run / "layout-failure.json").exists()
        assert load(run / "channels-last-micro/micro.json")["layout"] == "channels_last"
        settings += ((0.1, "channels_last"),)
    for layout in ("nchw", "channels-last"):
        path = run / (layout + "-micro/micro.json")
        if not path.exists():
            assert layout == "channels-last" and not finished["channels_last_enabled"]
            continue
        micro = load(path)
        assert micro["model_sha256"] == MODEL_SHA and micro["gradient_updates"] == 0
        assert micro["fixture_sha256"] == ledger["fixture_sha256"] and len(micro["cases"]) == 26
        assert [r["rows"] for r in micro["correctness"]] == [1, 9, 17, 33, 65, 127, 256]
        for check in micro["correctness"]:
            assert check["same_shape_capture_max_abs_error"] <= 1e-6
            assert (
                check["padding_max_abs_error"][0] <= 0.002
                and check["padding_max_abs_error"][1] <= 0.02
            )
        for n in (32, 128, 512, 2048):
            for method in ("eager", "graph"):
                rows = [
                    r
                    for r in micro["cases"]
                    if r["rows"] == n and r["method"] == method and not r.get("sustained")
                ]
                assert sorted(r["repeat"] for r in rows) == [1, 2, 3]
        for row in micro["cases"]:
            assert row["iterations"] > 0 and row["wall_seconds"] > 0
            if row.get("sustained"):
                assert row["rows"] == 2048 and row["wall_seconds"] >= 10
            else:
                assert row["cuda_event_seconds"] > 0
            assert (
                abs(
                    row["positions_per_second"]
                    - row["rows"] * row["iterations"] / row["wall_seconds"]
                )
                < 1e-6
            )
    expected_order = [
        (rep, delay, layout)
        for rep, variants in ((3, settings), (4, tuple(reversed(settings))))
        for delay, layout in variants
    ]
    assert [
        (r["repeat"], r["config"]["batch_wait_ms"], r["layout"]) for r in finished["results"]
    ] == expected_order
    groups = {}
    for row in finished["results"]:
        config = row["config"]
        repeat = row["repeat"]
        delay = config["batch_wait_ms"]
        layout = row["layout"]
        assert row["delay_ms"] == delay
        assert load(run / f"{repeat}-{layout}-wait-{delay}.json") == row
        for key, value in {
            "seed": 1069,
            "workers": 6,
            "live_games": 64,
            "full_simulations": 1024,
            "fast_simulations": 32,
            "fast_probability": 0.8,
            "inference_batch": 256,
            "leaf_batch": 8,
            "device": "cuda",
            "precision": "bfloat16",
        }.items():
            assert config[key] == value
        assert row["games"] == 384 and row["warmup_games"] == 12 and row["gradient_updates"] == 0
        assert row["native_sha256"] == expected_native and row["model_sha256"] == MODEL_SHA
        assert row["fixture_sha256"] == ledger["fixture_sha256"]
        m = row["metrics"]
        before, after = row["graph_before"], row["graph_after"]
        assert (m["simulations"] - row["recorded_roots"] * 1024) % 32 == 0
        assert m["simulations"] >= row["recorded_roots"] * 1024 > 0
        assert m["neural_positions"] == m["gpu_positions"] and m["cache_hits"] == 0
        assert after["requested_positions"] - before["requested_positions"] == m["neural_positions"]
        physical = (
            after["padded_positions"]
            - before["padded_positions"]
            + after["fallback_positions"]
            - before["fallback_positions"]
        )
        assert physical >= m["neural_positions"]
        groups.setdefault(f"{layout}-{delay}", []).append(
            {
                "repeat": repeat,
                "seconds": m["collect_seconds"],
                "simulations": m["simulations"],
                "neural_positions": m["neural_positions"],
                "physical_neural_positions": physical,
                "mean_batch": m["mean_gpu_batch"],
            }
        )
    summary = {}
    for delay, cases in groups.items():
        seconds = sum(c["seconds"] for c in cases)
        summary[delay] = {
            "cases": cases,
            "mean_seconds": seconds / 2,
            "games_per_second": 768 / seconds,
            "simulations_per_second": sum(c["simulations"] for c in cases) / seconds,
            "padding_factor": sum(c["physical_neural_positions"] for c in cases)
            / sum(c["neural_positions"] for c in cases),
            "rate_estimated_usd_per_1000_games": ledger["hourly_usd"] * seconds / 3600 / 768 * 1000,
        }
    by_app = {}
    for row in load(args.billing):
        by_app[row["object_id"]] = by_app.get(row["object_id"], Decimal(0)) + Decimal(row["cost"])
    ids = set(ledger["prior_apps"]) | {args.app_id}
    total = sum(by_app[name] for name in ids)
    assert 0 < total < Decimal("5")
    stop = ROOT / Path((ROOT / "runs/duo-teacher-user-stopped-current.txt").read_text().strip())
    preserved = load(stop / "stop-verified.json")
    for name, h in preserved["preserved"].items():
        assert sha(ROOT / name) == h
    for branch, row in preserved["retained"].items():
        assert sha(stop / "duo-teacher-search-1066" / branch / "resume.pt") == row["resume_sha256"]
    report = {
        "verified": True,
        "run": str(run.resolve()),
        "app_id": args.app_id,
        "latency": summary,
        "telemetry_by_phase": telemetry(
            run, recursive=True, allow_incomplete=(run / "layout-failure.json").exists()
        ),
        "channels_last_enabled": finished["channels_last_enabled"],
        "layout_check": layout_check,
        "metered_usd": str(by_app[args.app_id]),
        "combined_actual_usd": str(total),
        "cost_by_phase_app": {name: str(by_app[name]) for name in sorted(ids)},
        "preserved_models_unchanged": True,
        "gradient_updates": 0,
    }
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "runs/cuda-latency-verified.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "telemetry_by_phase"}, indent=2))


if __name__ == "__main__":
    main()
