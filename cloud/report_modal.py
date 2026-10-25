"""Render saved hardware measurements; does not launch cloud work."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODES = {"cpu": "CPU", "cuda": "CUDA fp32", "cuda_bfloat16": "CUDA bf16"}
COLORS = {"cpu": "#64748b", "cuda": "#2563eb", "cuda_bfloat16": "#059669"}


def render(paths, local_cpu=None):
    runs = [json.loads(path.read_text()) for path in paths]
    local = json.loads(local_cpu.read_text()) if local_cpu else None
    lines = [
        "# Modal hardware measurements — 25 October 2026",
        "",
        "Timing benchmark only: these measurements do not establish playing strength.",
        "All searches use 128 simulations/move, eight leaves/tree/batch, one Python",
        "coordinator and one PyTorch CPU thread. Every mode starts from the same",
        "weights within its architecture. The additive model is frozen frontier342;",
        "the CNN is seeded, untrained. Synthetic gradient updates are discarded.",
        "CUDA fp32 permits TF32; bf16 uses autocast. Precision can change search",
        "paths. Cloud CPU baselines run on the GPU host; the Mac was timed separately.",
        "",
        "![Hardware throughput](assets/modal-throughput.png)",
        "",
        "## Complete search pipeline",
        "",
        "32 concurrent games, four random opening plies, then four searched plies:",
        "128 decisions and 16,384 simulations per row. This includes Rust legality,",
        "tree traversal, observations, ragged candidate packing, transfers and inference.",
        "A timing row is one workload execution, not a confidence interval.",
        "",
        "| Host GPU | Model | Execution | Decisions/s | Neural positions | Neural batches |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for run in runs:
        for row in run["search"]:
            if row["live_games"] != 32:
                continue
            assert row["simulations"] == 16384, row
            lines.append(
                f"| {run['gpu']} | {row['architecture']} | {MODES[row['mode']]} | "
                f"{row['decisions_per_second']:.2f} | {row['neural_positions']} | "
                f"{row['inference_batches']} |"
            )
    if local:
        for architecture in ("additive", "conv"):
            rows = [
                r
                for r in local["search"]
                if r["architecture"] == architecture and r["live_games"] == 32
            ]
            seconds = float(np.median([r["seconds"] for r in rows]))
            lines.append(
                f"| Dennis's Mac | {architecture} | CPU, median of {len(rows)} | "
                f"{128 / seconds:.2f} | {rows[0]['neural_positions']} | "
                f"{rows[0]['inference_batches']} |"
            )
    lines += [
        "",
        "## Neural inference and synthetic gradient timing",
        "",
        "Inference uses the initial board with 828 legal moves, batch128; median of",
        "ten timings after three warmups, including copying outputs back to CPU.",
        "Training includes packing, optimizer setup on first step, and twelve updates",
        "on repeated initial-board records; this is not a representative learning curve.",
        "",
        "| Host GPU | Model | Execution | Batch128 inference ms | 12 gradient steps s |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for run in runs:
        training = {(r["architecture"], r["mode"]): r for r in run["training"]}
        for row in run["measurements"]:
            if row["batch"] != 128:
                continue
            train = training[(row["architecture"], row["mode"])]
            lines.append(
                f"| {run['gpu']} | {row['architecture']} | {MODES[row['mode']]} | "
                f"{row['milliseconds']:.3f} | {train['seconds']:.3f} |"
            )
    lines += ["", "## Run records", ""]
    for path, run in zip(paths, runs, strict=True):
        lines.append(
            f"- `{path.relative_to(ROOT)}`: {run['gpu']}, "
            f"PyTorch{run['torch']}, CUDA{run['cuda']}, "
            f"{run['seconds']:.1f}s measured workload; "
            f"${run['estimated_execution_usd']:.4f} estimated execution resources."
        )
    if local:
        lines.append(
            f"- `{local_cpu.relative_to(ROOT)}`: local Mac CPU, PyTorch{local['torch']}; "
            "one eight-game warmup plus two 32-game measurements per architecture."
        )
    lines += [
        "",
        "Resource estimates exclude image builds, startup, idle time and metering",
        "differences. Account billing snapshots and the reservation ledger are saved",
        "under runs/. The experiment note records the final budget assessment.",
        "",
    ]
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "blokus-modal-20261025"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for index, architecture in enumerate(("additive", "conv")):
        for column, kind in enumerate(("inference", "search")):
            labels, values, colors = [], [], []
            for run in runs:
                rows = run["measurements"] if kind == "inference" else run["search"]
                for row in rows:
                    if row["architecture"] != architecture:
                        continue
                    if kind == "inference" and row["batch"] != 128:
                        continue
                    if kind == "search" and row["live_games"] != 32:
                        continue
                    label = run["budget"]["gpu"].upper() + " / " + MODES[row["mode"]]
                    labels.append(label)
                    values.append(
                        128000 / row["milliseconds"]
                        if kind == "inference"
                        else row["decisions_per_second"]
                    )
                    colors.append(COLORS[row["mode"]])
            if local and kind == "search":
                seconds = [
                    r["seconds"]
                    for r in local["search"]
                    if r["architecture"] == architecture and r["live_games"] == 32
                ]
                labels.append("Mac / CPU")
                values.append(128 / float(np.median(seconds)))
                colors.append("#7c3aed")
            ax = axes[index, column]
            ax.barh(labels, values, color=colors)
            ax.invert_yaxis()
            ax.set_title(
                f"{architecture.upper()} — {'batch128 neural inference' if column == 0 else 'complete search'}"
            )
            ax.set_xlabel("Positions / second" if column == 0 else "Decisions / second")
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            ax.bar_label(ax.containers[0], fmt="%.1f", padding=3, fontsize=8)
            ax.set_xlim(right=max(values) * 1.22)
    fig.suptitle("GPU speed must survive the complete search pipeline", fontsize=15)
    assets = ROOT / "docs/assets"
    assets.mkdir(exist_ok=True)
    fig.savefig(assets / "modal-throughput.png", dpi=160)
    fig.savefig(assets / "modal-throughput.svg", metadata={"Date": None})
    svg = assets / "modal-throughput.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    (ROOT / "docs/modal-results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--local-cpu", type=Path)
    args = parser.parse_args()
    render(
        [path.resolve() for path in args.results],
        args.local_cpu.resolve() if args.local_cpu else None,
    )
