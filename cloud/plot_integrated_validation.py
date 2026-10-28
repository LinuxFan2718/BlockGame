"""Compact scientific figure from the independently verified GPU speed check."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
verified = json.loads((ROOT / "runs/integrated-validation-verified.json").read_text())
rows = json.loads((Path(verified["run"]) / "checks/finished.json").read_text())["cases"]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), constrained_layout=True)
colors = {"legacy": "#718096", "compact": "#277c9e", "bulk": "#b46a39"}
for i, (mode, color) in enumerate(colors.items()):
    x = np.arange(2) + (i - 1) * 0.23
    speeds, lower, upper, batches = [], [], [], []
    for full in (1024, 2048):
        group = [r for r in rows if r["mode"] == mode and r["full_simulations"] == full]
        values = [
            256 / (r["metrics"]["collect_seconds"] + r["metrics"]["graph_capture_seconds"])
            for r in group
        ]
        center = float(np.mean(values))
        speeds.append(center)
        lower.append(center - min(values))
        upper.append(max(values) - center)
        batches.append(float(np.mean([r["metrics"]["mean_gpu_batch"] for r in group])))
    axes[0].bar(
        x, speeds, 0.21, color=color, label=mode, yerr=np.asarray([lower, upper]), capsize=3
    )
    axes[1].bar(x, batches, 0.21, color=color, label=mode)
for ax in axes:
    ax.set_xticks([0, 1], ["1,024", "2,048"])
    ax.set_xlabel("Full-search simulations per move")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Games / second, including graph capture")
axes[1].set_ylabel("Mean actual inference batch")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Integrated L40S pipeline: frozen-model collection", fontsize=13)
fig.supxlabel(
    "256 measured games × 2 reversed-order repeats; whiskers show repeat range", fontsize=9
)
out = ROOT / "docs/assets/integrated-validation.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=180)
print(out)
