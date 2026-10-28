"""Summarize saved, read-only Modal telemetry without attaching to or changing a job."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def summarize(directory):
    directory = Path(directory)
    telemetry = [json.loads(s) for s in (directory / "telemetry.jsonl").read_text().splitlines()]
    progress = [
        json.loads(s) for s in (directory / "control1024/progress.jsonl").read_text().splitlines()
    ]
    gpu = np.asarray(
        [[float(v.strip()) for v in r["gpu"].split(",")] for r in telemetry if "gpu" in r]
    )
    cpu_intervals = [
        (b["monotonic"] - a["monotonic"], (b["cpu_ns"] - a["cpu_ns"]) / 1e9)
        for a, b in zip(telemetry, telemetry[1:])
        if "cpu_ns" in a and "cpu_ns" in b and b["cpu_ns"] >= a["cpu_ns"]
    ]

    def total(key):
        return sum(r["metrics"].get(key, 0) for r in progress)

    collection, capture = total("collect_seconds"), total("graph_capture_seconds")
    fitting = sum(r["training_seconds"] for r in progress)
    validation = sum(r.get("validation", {}).get("seconds", 0) for r in progress)
    timed = sum(r["round_seconds"] for r in progress)
    result = {
        "scope": "in-progress sample; telemetry and saved rounds have distinct end points; no strength outcomes",
        "samples": len(telemetry),
        "utc_start": telemetry[0]["utc"],
        "utc_end": telemetry[-1]["utc"],
        "cpu_cores_mean": sum(v for _, v in cpu_intervals) / sum(dt for dt, _ in cpu_intervals),
        "gpu_busy_mean": float(gpu[:, 0].mean()),
        "gpu_busy_median": float(np.median(gpu[:, 0])),
        "gpu_busy_p95": float(np.percentile(gpu[:, 0], 95)),
        "gpu_memory_controller_busy_mean": float(gpu[:, 1].mean()),
        "vram_mib_peak": float(gpu[:, 2].max()),
        "vram_mib_capacity": float(gpu[:, 3].max()),
        "host_ram_gib_peak": max(r.get("memory_bytes", 0) for r in telemetry) / 2**30,
        "rounds": len(progress),
        "games": progress[-1]["games"],
        "collection_seconds": collection,
        "inference_service_seconds": total("inference_seconds"),
        "inference_fraction_of_collection": total("inference_seconds") / collection,
        "graph_capture_seconds": capture,
        "fitting_seconds": fitting,
        "validation_seconds": validation,
        "timed_round_seconds": timed,
        "other_timed_round_seconds": timed - collection - capture - fitting - validation,
        "setup_commit_boundary_residual_seconds": progress[-1]["elapsed_seconds"] - timed,
        "elapsed_through_last_round": progress[-1]["elapsed_seconds"],
        "actual_mean_gpu_batch": total("gpu_positions") / total("gpu_batches"),
        "simulations": total("simulations"),
        "neural_positions": total("neural_positions"),
        "gradient_steps": sum(r["gradient_steps"] for r in progress),
    }
    return result, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--figure", type=Path, default=ROOT / "docs/assets/control-replication-live-profile.png"
    )
    args = parser.parse_args()
    result, rows = summarize(args.snapshot)
    (args.snapshot / "performance-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    times = [datetime.fromisoformat(r["utc"]) for r in rows]
    axes[0].plot(
        times, [float(r["gpu"].split(",")[0]) for r in rows], label="GPU busy", linewidth=1
    )
    cpu_times, cpu = [], []
    for a, b in zip(rows, rows[1:]):
        if "cpu_ns" in a and "cpu_ns" in b:
            cpu_times.append(datetime.fromisoformat(b["utc"]))
            cpu.append(
                (b["cpu_ns"] - a["cpu_ns"]) / 1e9 / (b["monotonic"] - a["monotonic"]) / 8 * 100
            )
    axes[0].plot(cpu_times, cpu, label="CPU / 8 cores", linewidth=1)
    axes[0].set(title="Busy time / CPU allocation", ylim=(0, 100), ylabel="Percent")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(
        times,
        [r.get("memory_bytes", 0) / (24 * 2**30) * 100 for r in rows],
        label="Host RAM / 24 GiB",
    )
    axes[1].plot(
        times,
        [float(r["gpu"].split(",")[2]) / float(r["gpu"].split(",")[3]) * 100 for r in rows],
        label="VRAM capacity",
    )
    axes[1].set(title="Memory capacity used", ylim=(0, 100), ylabel="Percent")
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo("America/New_York")))
        ax.set_xlabel("28 October 2026, EDT")
    labels = [
        "Collection",
        "Validation",
        "Fitting",
        "Graph capture",
        "Other round work",
        "Commit/setup residual*",
    ]
    values = [
        result[k]
        for k in (
            "collection_seconds",
            "validation_seconds",
            "fitting_seconds",
            "graph_capture_seconds",
            "other_timed_round_seconds",
            "setup_commit_boundary_residual_seconds",
        )
    ]
    axes[2].barh(
        labels, values, color=["#277c9e", "#718096", "#b46a39", "#b46a39", "#a4afb9", "#718096"]
    )
    axes[2].invert_yaxis()
    axes[2].set(
        title=f"First {result['rounds']} saved rounds",
        xlabel="Seconds (inference is within collection)",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Live control replication: host/inference coordination remains the constraint")
    fig.supxlabel(
        "GPU busy is not peak FLOPs. *Residual includes Volume commits, setup and timing boundaries; not isolated commit time.",
        fontsize=8,
    )
    target = args.figure
    fig.savefig(target, dpi=180)
    print(
        json.dumps(
            {
                "summary": str(args.snapshot / "performance-summary.json"),
                "figure": str(target),
                "cpu_cores": result["cpu_cores_mean"],
                "gpu_busy_percent": result["gpu_busy_mean"],
            }
        )
    )


if __name__ == "__main__":
    main()
