"""Audit completed timing artifacts and plot the M2/L40S comparison; no games."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = "ap-gL3LWke1O5eLx7eMCNp1yR"
COMMIT = "3d3a8031ac907b2dabd3b2c8b69c564c0ae105b2"
MODEL = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
STATE = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
NATIVE = "257b5a55c2c9fefcf3af5f15e240c7b18245abe49c44b781956b2b5351dac30e"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean_range(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def audit(mac, remote):
    sources = [mac / "finished.json", remote / "eager/finished.json", remote / "production/finished.json"]
    for folder in (mac, remote, remote / "eager", remote / "production"):
        require(not (folder / "failure.json").exists(), f"failure marker: {folder}")
    require(read(remote / "finished.json")["benchmark_optimizer_updates_discarded"] is True,
            "missing wrapper completion")
    execution = read(remote / "execution.json")
    ledger = [json.loads(line) for line in (ROOT / "runs/modal-metal-training-cost-ledger.jsonl").read_text().splitlines()]
    require(execution == ledger[0], "execution/reservation mismatch")
    require(execution["source_commit"] == COMMIT, "wrong source commit")
    require(execution["timeout_seconds"] == 900 and execution["reserved_usd"] < 1.50, "budget changed")
    expected_payload = read(ROOT / "runs/metal-training-payload-1090-sha256.json")
    require(execution["payload_sha256"] == expected_payload, "payload manifest changed")
    for name, expected in expected_payload.items():
        require(sha(ROOT / "runs/metal-training-payload-1090" / (name + ".py")) == expected,
                f"payload changed: {name}")
    runner = subprocess.check_output(["git", "show", COMMIT + ":cloud/modal_metal_training.py"], cwd=ROOT)
    require(hashlib.sha256(runner).hexdigest() == execution["runner_sha256"], "runner lineage mismatch")
    cases = [read(path) for path in sources]
    base = cases[0]
    reference_config = {k: v for k, v in base["config"].items() if k not in ("device", "precision")}
    summaries = []
    for index, (path, result) in enumerate(zip(sources, cases)):
        require(result["model_sha256"] == MODEL and result["source_state_sha256"] == STATE, "wrong inputs")
        require(result["initial_tensor_sha256"] == base["initial_tensor_sha256"], "different starting tensors")
        require(result["parameters"] == 52978 and result["initial_replay_positions"] == 100000, "wrong network/replay")
        require(result["hardware"]["mps_cpu_fallback"] == "0", "CPU fallback was enabled")
        require({k: v for k, v in result["config"].items() if k not in ("device", "precision")} == reference_config,
                "scientific configuration differs")
        require(result["config"]["workers"] == 6 and result["config"]["live_games"] == 64, "wrong parallelism")
        require(result["warmup"]["games"] == 12 and result["warmup"]["optimizer_steps"] == 2, "wrong warmup")
        require(result["backend"] == ("cuda_graphs" if index == 2 else "eager"), "wrong inference backend")
        require(result["config"]["device"] == ("mps" if index == 0 else "cuda"), "wrong device")
        require(result["config"]["precision"] == ("bfloat16" if index == 2 else "float32"), "wrong precision")
        require(result["deterministic_algorithms"] == ("warn_only" if index == 0 else "strict"), "wrong determinism mode")
        for name, digest in result["source_sha256"].items():
            if name == "_native":
                if index:
                    require(digest == NATIVE, "wrong Linux native build")
            else:
                require(digest == base["source_sha256"][name], f"different source: {name}")
                committed = subprocess.check_output(["git", "show", COMMIT + ":python/blokus_rl/" + name + ".py"], cwd=ROOT)
                require(hashlib.sha256(committed).hexdigest() == digest, f"source lineage: {name}")
        require(len(result["fit"]) == len(result["rounds"]) == 2, "incomplete repetitions")
        for repeat, row in enumerate(result["fit"], 1):
            require(row["repeat"] == repeat and row["steps"] == 64 and row["examples"] == 16384, "fit accounting")
            require(row["seconds"] > 0 and all(math.isfinite(v) for v in row["losses"].values()), "invalid fit")
            require(row["final_tensor_sha256"] != result["initial_tensor_sha256"], "model did not update")
        for repeat, row in enumerate(result["rounds"], 1):
            m = row["metrics"]
            require(row["repeat"] == repeat and row["games"] == 256 and row["checkpoint_readback_verified"], "round accounting")
            require(row["simulations"] == row["recorded_roots"] * 1024 + row["fast_roots"] * 32 == m["simulations"], "search budget")
            require(row["gradient_steps"] == math.ceil(3 * row["recorded_roots"] / 256), "gradient ratio")
            require(row["gradient_examples"] == 256 * row["gradient_steps"], "gradient examples")
            require(m["neural_positions"] == m["gpu_positions"] == m["cache_misses"] > 0 and m["cache_hits"] == 0, "neural counters/cache")
            require(row["collection_seconds"] == m["collect_seconds"] > 0, "collection timing")
            require(row["round_cycle_seconds"] >= row["collection_seconds"] + row["fit_seconds"] + row["checkpoint_seconds"], "cycle timing")
            require(all(math.isfinite(v) for v in row["losses"].values()), "nonfinite round loss")
            if index == 2:
                require(m["graph_fallback_positions"] == 0 and m["graph_requested_positions"] == m["neural_positions"], "graph counters")
        summaries.append({
            "label": ("M2 Metal", "L40S float32 eager", "L40S production")[index],
            "artifact": str(path.relative_to(ROOT)), "artifact_sha256": sha(path),
            "fit_64_seconds": mean_range([r["seconds"] for r in result["fit"]]),
            **{key: mean_range([r[key] for r in result["rounds"]]) for key in
               ("collection_seconds", "round_cycle_seconds", "fit_seconds", "checkpoint_seconds")},
            "games_per_second": 512 / sum(r["collection_seconds"] for r in result["rounds"]),
            "simulations_per_second": sum(r["simulations"] for r in result["rounds"]) / sum(r["collection_seconds"] for r in result["rounds"]),
            "neural_positions_per_second": sum(r["metrics"]["neural_positions"] for r in result["rounds"]) / sum(r["collection_seconds"] for r in result["rounds"]),
            "rounds": result["rounds"], "hardware": result["hardware"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
    apps = read(ROOT / "runs/metal-training-apps-after.json")
    require(all(int(row["tasks"]) == 0 and row["state"] == "stopped" for row in apps), "active compute remains")
    require(any(row["app_id"] == APP for row in apps), "missing benchmark app")
    billing = read(ROOT / "runs/metal-training-billing-after.json")
    costs = [row for row in billing if row["object_id"] == APP]
    require(costs and any(row["resource"] == "L40S" for row in costs), "missing GPU metering")
    actual = sum(Decimal(row["cost"]) for row in costs)
    require(actual < Decimal("1.50"), "actual phase cap exceeded")
    account = Decimal(read(ROOT / "runs/metal-training-summary-after.json")["metered_cost"])
    require(account < Decimal(80), "credit reserve breached")
    protected = read(ROOT / "runs/control-replication-preserved-before.json")
    for name, digest in protected.items():
        require(sha(ROOT / name) == digest, f"preserved model changed: {name}")
    return {"cases": summaries, "execution": execution, "app_id": APP, "billing_resources": costs,
            "actual_provider_usd": str(actual), "account_metered_usd": str(account),
            "protected_checkpoints_verified": len(protected),
            "checks": "source/input lineage, fixed work, finite updates, counters, checkpoint readback, stopped compute, provider cap, prior model hashes"}


def plot(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = report["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    colors = ["#147d79", "#5165aa", "#a85b28"]
    for ax, metric, title in zip(axes, ("fit_64_seconds", "collection_seconds"),
                                 ("64 optimizer steps", "256 self-play games")):
        means = [c[metric]["mean"] for c in cases]
        ax.bar(range(3), means, color=colors, width=.62)
        ax.errorbar(range(3), means, yerr=[[c[metric]["mean"]-c[metric]["min"] for c in cases],
                                         [c[metric]["max"]-c[metric]["mean"] for c in cases]],
                    color="#222222", fmt="none", capsize=5)
        for i, value in enumerate(means):
            ax.text(i, value + max(means)*.035, f"{value:.2f}s", ha="center", fontsize=11)
        ax.set_xticks(range(3), ["M2\nMetal float32", "L40S\nEager float32", "L40S\nGraphs bfloat16"])
        ax.set_ylim(0, max(means)*1.24)
        ax.set_ylabel("Seconds · lower is faster")
        ax.set_title(title, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Duo training workload on Apple M2 and Modal L40S", fontsize=14, fontweight="bold")
    fig.supxlabel("Means and observed ranges of two repetitions · 52,978 parameters · same parent/replay", fontsize=9)
    fig.savefig(ROOT / "docs/assets/metal-training-results.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.mac.resolve(), args.remote.resolve())
    plot(report)
    output = ROOT / "runs/metal-training-verified.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verification": str(output), "actual_provider_usd": report["actual_provider_usd"],
                      "cases": [{k: v for k, v in c.items() if k not in ("rounds", "hardware")} for c in report["cases"]]}, indent=2))


if __name__ == "__main__":
    main()
