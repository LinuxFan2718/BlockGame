"""Plot verified CUDA Graph comparisons; reads only saved local artifacts."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
reports = {
    gpu: json.loads((ROOT / f"runs/cuda-graphs-{gpu}-verified.json").read_text())
    for gpu in ("l40s", "h100")
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
variants = ["current-eager", "current-graph", "native-eager", "native-graph", "native-graph-wide"]
labels = [
    "Current eager",
    "Current + graphs",
    "Native eager",
    "Native + graphs",
    "Native + graphs, live64",
]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
colors = {"l40s": "#167f87", "h100": "#d28c44"}
for index, (gpu, report) in enumerate(reports.items()):
    for y, key in enumerate(variants):
        value = report["collection"][key]
        mean = value["mean_seconds"]
        times = [r["seconds"] for r in value["cases"]]
        position = y + (index - 0.5) * 0.32
        axes[0].barh(
            position, mean, height=0.29, color=colors[gpu], label=gpu.upper() if y == 0 else None
        )
        axes[0].errorbar(
            mean,
            position,
            xerr=[[mean - min(times)], [max(times) - mean]],
            fmt="none",
            color="#23333c",
            capsize=2,
        )
        axes[0].text(max(times) + 1, position, f"{mean:.1f}", va="center", fontsize=8)
    for mode, style in [("eager", "--"), ("graph", "-")]:
        batches = [32, 128, 512, 2048]
        values = [report["micro_summary"][f"{n}-{mode}"] / 1e6 for n in batches]
        axes[1].plot(
            batches, values, style, marker="o", color=colors[gpu], label=f"{gpu.upper()} {mode}"
        )
axes[0].set_yticks(range(len(variants)), labels)
axes[0].invert_yaxis()
axes[0].set_xlabel("Seconds per 384 games (lower is faster)")
axes[0].set_xlim(
    0, max(r["collection"]["current-eager"]["mean_seconds"] for r in reports.values()) * 1.15
)
axes[0].set_title("Actual self-play collection", loc="left", fontweight="bold")
axes[0].legend(frameon=False)
axes[1].set_xscale("log", base=2)
axes[1].set_xticks([32, 128, 512, 2048], [32, 128, 512, 2048])
axes[1].set_xlabel("Neural batch size")
axes[1].set_ylabel("Million positions / second")
axes[1].set_title("GPU-resident inference ceiling", loc="left", fontweight="bold")
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle(
    "Frozen Blokus Duo: graphs speed inference; the actor pipeline limits delivery",
    fontweight="bold",
    fontsize=12,
)
fig.text(
    0.5,
    -0.04,
    "2 collection repeats per case · full1024 / fast32 · 8 CPU cores / 24 GiB · zero learning",
    ha="center",
    fontsize=9,
)
for ext in ("png", "svg"):
    fig.savefig(
        ROOT / f"docs/assets/cuda-graphs-results.{ext}",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

path = ROOT / "runs/cuda-latency-verified.json"
if path.exists():
    report = json.loads(path.read_text())
    run = Path(report["run"])
    keys = ["nchw-1.0", "nchw-0.1", "nchw-0.0"]
    names = ["NCHW, 1 ms", "NCHW, 0.1 ms", "NCHW, no wait"]
    if report["channels_last_enabled"]:
        keys += ["channels_last-0.1"]
        names += ["Channels last, 0.1 ms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for y, key in enumerate(keys):
        value = report["latency"][key]
        rate = value["games_per_second"]
        rates = [384 / r["seconds"] for r in value["cases"]]
        axes[0].barh(y, rate, color="#167f87", height=0.58)
        axes[0].errorbar(
            rate,
            y,
            xerr=[[rate - min(rates)], [max(rates) - rate]],
            fmt="none",
            color="#203440",
            capsize=3,
        )
        axes[0].text(max(rates) + 0.12, y, f"{rate:.2f}", va="center")
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(report["latency"][k]["games_per_second"] for k in keys) * 1.18)
    axes[0].set_xlabel("Completed games / second (higher is faster)")
    axes[0].set_title("H100: same frozen search workload", loc="left", fontweight="bold")
    for layout, color in [("nchw", "#d28c44"), ("channels-last", "#167f87")]:
        source = run / (layout + "-micro/micro.json")
        if not source.exists():
            continue
        cases = json.loads(source.read_text())["cases"]
        for mode, style in [("eager", "--"), ("graph", "-")]:
            batches = [32, 128, 512, 2048]
            values = [
                statistics.median(
                    c["positions_per_second"]
                    for c in cases
                    if c["rows"] == n and c["method"] == mode and not c.get("sustained")
                )
                / 1e6
                for n in batches
            ]
            axes[1].plot(batches, values, style, marker="o", color=color, label=f"{layout} {mode}")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([32, 128, 512, 2048], [32, 128, 512, 2048])
    axes[1].set_xlabel("Neural batch size")
    axes[1].set_ylabel("Million positions / second")
    axes[1].set_title("Same-device layout comparison", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "H100 follow-up: request latency and convolution layout", fontsize=12, fontweight="bold"
    )
    fig.text(
        0.5,
        -0.045,
        "Two order-reversed collection repeats · live64 · full1024 / fast32 · unchanged weights",
        ha="center",
        fontsize=9,
    )
    for ext in ("png", "svg"):
        fig.savefig(
            ROOT / f"docs/assets/cuda-latency-results.{ext}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
