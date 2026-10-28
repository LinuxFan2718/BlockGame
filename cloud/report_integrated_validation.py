"""Read-only local audit of the integrated GPU validation; never launch games."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--apps", type=Path, required=True)
    parser.add_argument("--app-id", required=True)
    args = parser.parse_args()
    root, checks = args.run, args.run / "checks"
    apps = read(args.apps)
    assert all(int(a["tasks"]) == 0 for a in apps)
    assert any(a["app_id"] == args.app_id and a["state"] == "stopped" for a in apps)
    assert not (root / "failure.json").exists()
    assert read(root / "wrapper-finished.json")["seconds"] < 1200
    execution = read(root / "execution.json")
    ledger = [
        json.loads(line)
        for line in (ROOT / "runs/modal-integrated-cost-ledger.jsonl").read_text().splitlines()
    ]
    assert execution == next(r for r in ledger if r["stage"] == execution["stage"])
    manifest = read(ROOT / "runs/integrated-payload-1070-sha256.json")
    assert execution["payload_sha256"] == manifest
    for name, expected in manifest.items():
        assert (
            hashlib.sha256((ROOT / "runs/integrated-payload-1070" / name).read_bytes()).hexdigest()
            == expected
        )
    native = read(root / "native-verified.json")
    assert native == {
        "random_games": 32,
        "positions": 964,
        "arbitrary_prior_roots": 12,
        "simulations_per_implementation": 3072,
        "exact": True,
    }
    numerical = read(checks / "numerical-checks.json")
    for key in ("initial", "after_optimizer_updates"):
        assert [c["rows"] for c in numerical[key]] == [1, 9, 17, 33, 65, 127, 256]
        for c in numerical[key]:
            assert c["bulk_matches_row_graph_exactly"]
            assert c["max_abs_error"][0] <= 0.002 and c["max_abs_error"][1] <= 0.02
    assert numerical["discarded_gradient_steps"] == 2
    assert read(checks / "recovery-check.json")["exact_single_worker_gpu_recovery"]
    finished = read(checks / "finished.json")
    assert (
        finished["model_sha256"]
        == "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
    )
    assert (
        finished["source_resume_sha256"]
        == "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
    )
    assert (
        finished["discarded_smoke_games"] == 16 and finished["discarded_optimizer_probe_steps"] == 2
    )
    order = [(full, mode) for full in (1024, 2048) for mode in ("legacy", "compact", "bulk")]
    assert [(r["repeat"], r["full_simulations"], r["mode"]) for r in finished["cases"]] == [
        (rep, f, m)
        for rep, sequence in ((1, order), (2, list(reversed(order))))
        for f, m in sequence
    ]
    means = {}
    for row in finished["cases"]:
        assert read(checks / (row["tag"] + ".json")) == row
        assert row["games"] == 256 and row["warmup_games"] == 12
        metrics = row["metrics"]
        assert (
            metrics["gpu_positions"]
            == metrics["neural_positions"]
            == metrics["graph_requested_positions"]
            > 0
        )
        assert metrics["graph_fallback_positions"] == 0
        fast = metrics["simulations"] - row["recorded_roots"] * row["full_simulations"]
        assert fast >= 0 and fast % 32 == 0
        assert metrics["neural_positions"] <= metrics["simulations"]
        assert metrics["graph_padded_positions"] >= metrics["graph_requested_positions"]
        assert 0 < metrics["collect_seconds"] < 150
        assert row["recorded_roots"] > 0
        c = row["config"]
        assert (
            c["seed"],
            c["live_games"],
            c["workers"],
            c["inference_batch"],
            c["leaf_batch"],
            c["batch_wait_ms"],
        ) == (1070, 64, 6, 256, 8, 0)
    for mode in ("legacy", "compact", "bulk"):
        means[mode] = {
            str(full): float(
                np.mean(
                    [
                        r["metrics"]["collect_seconds"] + r["metrics"]["graph_capture_seconds"]
                        for r in finished["cases"]
                        if r["mode"] == mode and r["full_simulations"] == full
                    ]
                )
            )
            for full in (1024, 2048)
        }
    eligible = [
        m for m in means if all(means[m][f] <= means["legacy"][f] for f in ("1024", "2048"))
    ]
    selected = min(eligible, key=lambda m: sum(means[m].values()))
    settings = next(r["runtime"] for r in finished["cases"] if r["mode"] == selected)
    runtime = dict(inference_backend="cuda_graphs", evaluation_backend="cuda_graphs", **settings)
    result = {
        "passed": True,
        "run": str(root.resolve()),
        "app_id": args.app_id,
        "payload_sha256": manifest,
        "selected_mode": selected,
        "selected_runtime": runtime,
        "mean_collection_plus_capture_seconds": means,
        "games_per_case": 256,
        "caveat": "two ordered repeats; timing variability, no learning evidence; all smoke data discarded",
    }
    (ROOT / "runs/integrated-validation-verified.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
