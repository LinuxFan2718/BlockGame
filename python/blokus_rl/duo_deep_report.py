"""Render the completed bounded deep/shallow study without launching computation."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .duo_deep import atomic_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blokus-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"residual": "Deep residual CNN", "conv": "Shallow CNN"}
COLORS = {"residual": "#a23b72", "conv": "#187b80"}
APP_IDS = {
    "study": "ap-H3fgjNS4auEbHTqNTsQXon",
    "L4 screen": "ap-RoaDV7u20Yt9BGIxn9Kqk8",
    "L40S screen": "ap-9R1Y2Qelk553FZkSeB6RlK",
}


def read(path):
    return json.loads(Path(path).read_text())


def training_summary(root, architecture):
    rows = [
        json.loads(line)
        for line in (root / architecture / "progress.jsonl").read_text().splitlines()
    ]
    completed = read(root / architecture / "completed.json")
    assert [r["round"] for r in rows] == list(range(1, completed["rounds"] + 1))
    protocol = read(root / architecture / "protocol.json")
    selected = torch.load(root / f"frozen-{architecture}.pt", map_location="cpu", weights_only=True)
    totals = {
        key: sum(r[key] for r in rows)
        for key in (
            "new_training_positions",
            "gradient_steps",
            "gradient_examples",
            "training_seconds",
        )
    }
    totals |= {
        key: sum(r["metrics"][key] for r in rows)
        for key in (
            "simulations",
            "neural_positions",
            "gpu_batches",
            "gpu_positions",
            "collect_seconds",
        )
    }
    return (
        completed
        | totals
        | {
            "parameters": protocol["parameters"],
            "selected_round": selected["update"],
            "mean_gpu_batch": totals["gpu_positions"] / totals["gpu_batches"],
            "rows": rows,
        }
    )


def table(labelled):
    lines = [
        "| Agent vs opponent | W/D/L | Score | Paired 95% interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, row in labelled:
        low, high = row["score_rate_95pct_cluster_bootstrap"]
        lines.append(
            f"| {label} | {row['wins']}/{row['draws']}/{row['losses']} | "
            f"{row['score_rate']:.2%} | {low:.2%}–{high:.2%} |"
        )
    return lines


def draw(cloud, local, training, destination):
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "blokus-deep-20261026"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    ax = axes[0, 0]
    for architecture, values in training.items():
        rows = [r for r in values["rows"] if "validation" in r]
        ax.plot(
            [r["games"] / 1000 for r in rows],
            [r["validation"]["score_rate"] * 100 for r in rows],
            "o-",
            color=COLORS[architecture],
            label=NAMES[architecture],
            markersize=3,
        )
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set(
        title="Validation: checkpoint selection only",
        xlabel="Self-play games (thousands)",
        ylabel="Score vs original reference (%)",
        ylim=(0, 100),
    )
    ax.legend(fontsize=8)
    categories = [
        (
            axes[0, 1],
            "Primary: deep vs shallow",
            [
                ("Search 128 each", cloud["residual_vs_conv_search128"], "residual"),
                ("Raw policies", cloud["residual_vs_conv_raw"], "residual"),
            ],
        ),
        (
            axes[1, 0],
            "External test: Pentobi 31.0 level 1",
            [
                (f"{NAMES[a]} · {mode}", cloud[f"{a}_vs_pentobi_{budget}"], a)
                for a in NAMES
                for mode, budget in (("raw", 0), ("search128", 128))
            ],
        ),
        (
            axes[1, 1],
            "Local test vs preserved warm round 8",
            [
                (f"{NAMES[a]} · {mode}", local[f"{a}_{mode}_vs_warm_round8"], a)
                for a in NAMES
                for mode in ("raw", "search128")
            ],
        ),
    ]
    for ax, title, rows in categories:
        for y, (label, row, architecture) in enumerate(rows):
            score = row["score_rate"] * 100
            low, high = np.asarray(row["score_rate_95pct_cluster_bootstrap"]) * 100
            ax.errorbar(
                score,
                y,
                xerr=[[score - low], [high - score]],
                fmt="o",
                capsize=3,
                color=COLORS[architecture],
            )
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.axvline(50, color="gray", linestyle="--", linewidth=1)
        ax.set(
            title=title,
            xlabel="Score (%) · nominal paired 95% interval",
            xlim=(0, 100),
            ylim=(-0.5, len(rows) - 0.5),
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Cold-start Blokus Duo: 32,768 self-play games per model · one training seed", fontsize=13
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def render(run, local_dir, billing, out):
    run, local_dir, out = Path(run), Path(local_dir), Path(out)
    assert (run / "finished.json").exists() and (local_dir / "completed.json").exists()
    protocol, local_protocol = (
        read(run / "heldout-protocol.json"),
        read(local_dir / "protocol.json"),
    )
    assert all(
        local_protocol["checkpoint_sha256"][a] == protocol["checkpoint_sha256"][a] for a in NAMES
    )
    training = {a: training_summary(run, a) for a in NAMES}
    cloud = {
        p.stem.removeprefix("heldout-"): read(p)
        for p in run.glob("heldout-*.json")
        if p.name != "heldout-protocol.json" and not p.name.endswith(".partial.json")
    }
    local = {
        name: read(local_dir / f"{name}.json")
        for name in read(local_dir / "completed.json")["cases"]
    }
    costs = {
        name: sum(float(r["cost"]) for r in read(billing) if r["object_id"] == app)
        for name, app in APP_IDS.items()
    }
    total_cost = sum(costs.values())
    samples = []
    for line in (run / "telemetry.jsonl").read_text().splitlines():
        row = json.loads(line)
        value = row.get(
            "gpu_busy_percent,memory_io_percent,memory_used_mib,memory_total_mib,power_watts"
        )
        if value:
            try:
                samples.append([float(v.strip()) for v in value.split(",")])
            except ValueError:
                pass
    telemetry = np.asarray(samples)
    image_path = out.parent / "assets/deep-results.png"
    draw(cloud, local, training, image_path)
    primary = cloud["residual_vs_conv_search128"]
    lines = [
        "# Cold-start Blokus Duo study — 26 October 2026",
        "",
        "**The deeper network did not help in this bounded study.** The shallow CNN",
        "learned entirely from self-play and was substantially stronger than the deep",
        "model and the original reference. Pentobi level 1 still beats our agents.",
        "",
        f"The prespecified primary comparison scored **{primary['score_rate']:.2%}** for",
        "deep search128 against shallow search128 (paired 95% interval",
        f"{primary['score_rate_95pct_cluster_bootstrap'][0]:.2%}–{primary['score_rate_95pct_cluster_bootstrap'][1]:.2%}).",
        "The success criterion was a lower bound above 50%; it failed. This is evidence",
        "about these two trained agents at this budget, not a general verdict on deep CNNs.",
        "",
        "![Study results](assets/deep-results.png)",
        "",
        "## What was trained",
        "",
        "Both models began with random weights and played both seats. Rust supplied",
        "legal moves and terminal win/draw/loss. Search supplied policy visit targets;",
        "the final game outcome supplied value targets. No heuristic, Pentobi or other",
        "expert supplied training labels. During training, the original reference was",
        "used for checkpoint selection, never for training targets.",
        "",
        "Both used 128 rounds × 256 games, six CPU search workers and a shared L40S",
        "inference server. On 80% of moves, search used 32 simulations; on 20%, 256 with",
        "root noise. Only the latter positions entered replay. The fixed settings were",
        "100,000 replay positions, about 3 gradient examples per new recorded position,",
        "minibatch 256, AdamW learning rate 0.001, clipped gradients and bf16 GPU autocast.",
        "",
        "Deep: 10 residual blocks × 64 channels, GroupNorm and inventory FiLM. Shallow:",
        "four 32-channel convolution layers. Depth, width, normalization and inventory",
        "conditioning design changed together, along with coordinate inputs and tanh",
        "versus linear value outputs. This does not isolate a causal depth effect.",
        "",
        "| Actual full training budget | Deep | Shallow |",
        "| --- | ---: | ---: |",
    ]
    fields = [
        ("Parameters", "parameters"),
        ("Games", "games"),
        ("Recorded full-search positions", "new_training_positions"),
        ("Search simulations", "simulations"),
        ("Neural positions evaluated", "neural_positions"),
        ("Gradient steps", "gradient_steps"),
        ("Gradient examples", "gradient_examples"),
        ("Selected round (validation)", "selected_round"),
    ]
    for label, key in fields:
        lines.append(f"| {label} | {training['residual'][key]:,} | {training['conv'][key]:,} |")
    lines += [
        f"| Training elapsed, including checkpoint/commit overhead | {training['residual']['elapsed_seconds'] / 3600:.3f}h | {training['conv']['elapsed_seconds'] / 3600:.3f}h |",
        f"| Mean shared inference batch | {training['residual']['mean_gpu_batch']:.1f} | {training['conv']['mean_gpu_batch']:.1f} |",
        "",
        "Equal games and collection/training rules do not imply equal wall time or",
        "identical neural/gradient counts; game trajectories differ. The test uses",
        "selected checkpoints, not necessarily the final weights. All checkpoints",
        "are preserved, including the weaker deep network.",
        "",
        "## Held-out cloud results",
        "",
        "Each row contains 200 games: 100 four-ply random opening positions, each",
        "played with seats swapped. Test seed 2569000 was separate from the 24-pair",
        "validation seed 2563000 used every 8 rounds for checkpoint selection. Frozen",
        "checkpoint hashes were recorded before test openings were generated.",
        "",
        "Score is wins plus half draws. Intervals resample whole opening pairs",
        "10,000 times. They are nominal 95%, without correction for the many secondary",
        "comparisons, and do not include uncertainty across independent training runs.",
        "",
    ]
    labelled = [
        ("Deep search128 vs shallow search128 (primary)", cloud["residual_vs_conv_search128"]),
        ("Deep raw vs shallow raw", cloud["residual_vs_conv_raw"]),
    ]
    for a in NAMES:
        labelled += [
            (f"{NAMES[a]} {mode} vs original {mode}", cloud[f"{a}_vs_reference_{mode}"])
            for mode in ("raw", "search128")
        ]
        labelled += [
            (f"{NAMES[a]} {mode} vs Pentobi 31.0 level 1", cloud[f"{a}_vs_pentobi_{budget}"])
            for mode, budget in (("raw", 0), ("search128", 128))
        ]
    lines += table(labelled)
    lines += [
        "",
        "Searched neural-vs-neural rows give both sides 128 simulations/move.",
        "Pentobi uses its own level 1 search budget, one thread, no book/resignation;",
        "this is an outside playing-strength benchmark, not matched wall-clock or",
        "search-node compute. Its pinned 31.0 engine independently checks every game",
        "and final margin against Rust. Pentobi remained evaluation-only. Per-case JSON",
        "stores actual search/neural counters; raw inference and Pentobi node counts",
        "were not collected. Rust verifies the requested simulation count at every root.",
        "",
        "## Local comparisons and learning checks",
        "",
        "Same frozen selected weights and opening book, evaluated locally in CPU",
        "float32, using 10-pair atomic chunks. Cloud uses GPU bf16 autocast; precision",
        "and inference grouping can change decisions. The old warm-round8 checkpoint",
        "was copied and hashed locally and was never uploaded for this study.",
        "",
    ]
    labelled = []
    for a in NAMES:
        labelled += [
            (f"{NAMES[a]} raw vs its untrained initial network", local[f"{a}_raw_vs_initial"]),
            (f"{NAMES[a]} raw vs largest-piece heuristic", local[f"{a}_raw_vs_largest"]),
            (f"{NAMES[a]} raw vs warm-round8 raw", local[f"{a}_raw_vs_warm_round8"]),
            (
                f"{NAMES[a]} search128 vs warm-round8 search128",
                local[f"{a}_search128_vs_warm_round8"],
            ),
        ]
    lines += table(labelled)
    lines += [
        "",
        "An untrained neural network is not a uniform-random player. These checks",
        "describe learning and baseline strength; they did not select checkpoints.",
        "Old reports used different opening seeds and must not be treated as direct",
        "head-to-head rankings against today's results.",
        "",
        "## Normal starting board: descriptive only",
        "",
        "Exactly two deterministic seat-swapped games per case, with no population CI.",
        "",
        "| Agent vs original reference, same search budget | W/D/L |",
        "| --- | ---: |",
    ]
    for a in NAMES:
        for budget in (0, 128):
            row = read(run / f"descriptive-{a}_normal_start_{budget}.json")
            lines.append(
                f"| {NAMES[a]}, {budget} simulations | {row['wins']}/{row['draws']}/{row['losses']} |"
            )
    lines += [
        "",
        "## Hardware, costs and recovery",
        "",
        f"The main call finished in {read(run / 'finished.json')['seconds'] / 3600:.3f} hours and",
        "the app is stopped with zero tasks. No training retry was necessary.",
        f"Provider per-app metered snapshot: main study **${costs['study']:.4f}**;",
        f"L4 screen ${costs['L4 screen']:.4f}; L40S screen ${costs['L40S screen']:.4f};",
        f"phase app total **${total_cost:.4f}**. This is metered usage covered by existing",
        "credits, distinct from $18.45155 conservative reservations and the $35 phase",
        "ceiling. Volume storage/download accounting can arrive later. Resource rates",
        "included the GPU, eight capped CPU cores and 24 GiB capped RAM.",
        "",
        f"Telemetry sampled about every 5 seconds: mean GPU busy {telemetry[:, 0].mean():.1f}%,",
        f"median {np.median(telemetry[:, 0]):.1f}%, peak GPU memory in use",
        f"{telemetry[:, 2].max() / 1024:.2f} GiB of {telemetry[:, 3].max() / 1024:.1f} GiB.",
        "These whole-call samples include collection, training, checkpoint IO and",
        "evaluation. GPU memory capacity use and GPU compute utilization are different",
        "measurements; filling memory is not the objective.",
        "",
        "The first artifact download failed because the Modal CLI treated a nonexistent",
        "destination as a file while recursively creating a directory. Retrying into",
        "that empty directory succeeded, without restarting paid compute. The final",
        "client log-tail timeout occurred after the remote call returned; finished.json,",
        "all test artifacts and provider stopped state confirm completion. Neither was",
        "a training failure. Frozen hashes and earlier certified model hashes verified.",
        "",
        "All 68 Python tests passed, including a real interrupted/resumed local evaluation",
        "that preserved games, seed offsets and paired intervals and rejected changed",
        "settings. Existing rules were unchanged. Cloud artifacts have a local SHA256",
        "manifest and compressed backup; local results are backed up after completion.",
        "",
        "## Interpretation and next hypothesis",
        "",
        "Batched self-play can cold-start a useful Blokus Duo policy without an outside",
        "teacher. Search adds substantial strength. The deeper architecture failed this",
        "comparison despite equal game budgets; its cause is unproven. More parameters",
        "alone did not provide a better agent at this training ratio and duration.",
        "",
        "The next useful experiment is a small prespecified multi-seed replication of",
        "the shallow setup, with fresh validation/test openings and a fixed compute",
        "budget, before tuning architecture or training against Pentobi. A separate",
        "training-ratio ablation could test whether the deep model needs more updates",
        "per collected position. Neither experiment has been launched. This bounded",
        "study stops here; one seed cannot establish SOTA or superhuman strength.",
        "",
        "## Reproduction and artifacts",
        "",
        "Methods: [deep-self-play.md](deep-self-play.md). Rolling audit:",
        "[deep-status.md](deep-status.md). Architecture explanation:",
        "[alphablokus-explanation.md](alphablokus-explanation.md).",
        "",
        "```sh",
        ".venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_deep.py --stage study --gpu l40s --run-name duo-deep-study-764",
        f".venv/bin/python -m blokus_rl.duo_deep_local --run {run} --out {local_dir} --threads 1",
        f".venv/bin/python -m blokus_rl.duo_deep_report --run {run} --local {local_dir} --billing {billing} --out {out}",
        "```",
        "",
        "The cloud command is the original command, not an instruction to restart a",
        "finished run. Resume local evaluation only explicitly with --resume and the",
        "same frozen protocol after checking that no duplicate is active.",
        "",
        f"- Cloud artifact directory: `{run}`.",
        f"- Local comparison directory: `{local_dir}`.",
        f"- Provider snapshot: `{billing}`.",
        "- Original training source commit: `6963505b5a96318dcd136368388d7cac1d886e72`.",
        "- Private Volume: `blokus-deep-selfplay-20261026`; no new model was published.",
        "",
        "Frozen selected checkpoint SHA256:",
        "",
        *[f"- {NAMES[a]}: `{h}`." for a, h in protocol["checkpoint_sha256"].items()],
        "",
    ]
    out.write_text("\n".join(lines))
    atomic_json(
        local_dir / "report-summary.json",
        {
            "primary": primary,
            "costs_usd_snapshot": costs,
            "training": {
                a: {k: v for k, v in row.items() if k != "rows"} for a, row in training.items()
            },
            "cloud_cases": cloud,
            "local_cases": local,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--billing", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/deep-results.md"))
    args = parser.parse_args()
    render(args.run, args.local, args.billing, args.out)


if __name__ == "__main__":
    main()
