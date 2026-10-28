"""Append independently audited full-study results to the morning report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def pct(value):
    return f"{100 * value:.2f}%"


def interval(values):
    return "–".join(pct(v) for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.verification.read_text())
    names = {
        "parent": "Preserved parent",
        "control1024": "Control 1,024",
        "teacher2048": "Teacher 2,048",
    }
    colors = ["#718096", "#277c9e", "#b46a39"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for i, (name, label) in enumerate(names.items()):
        y, lo, hi = [], [], []
        for level in (3, 4):
            row = result["external"][f"{name}-pentobi{level}"]
            center = row["score_rate"] * 100
            limits = np.asarray(row["score_rate_95pct_cluster_bootstrap"]) * 100
            if row.get("zero_score_one_sided_95_upper") is not None:
                limits[1] = row["zero_score_one_sided_95_upper"] * 100
            y.append(center)
            lo.append(center - limits[0])
            hi.append(limits[1] - center)
        axes[0].errorbar(
            np.arange(2) + (i - 1) * 0.13,
            y,
            yerr=[lo, hi],
            marker="o",
            capsize=3,
            label=label,
            color=colors[i],
        )
    axes[0].axhline(50, color="#999999", linewidth=0.8, linestyle="--")
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Pentobi 3", "Pentobi 4"],
        ylabel="Score %, draws half",
        ylim=(0, 100),
    )
    axes[0].legend(frameon=False, fontsize=8)
    labels, differences, lo, hi = [], [], [], []
    for key, label in [
        ("teacher2048-minus-parent-pentobi4", "Teacher − parent"),
        ("teacher2048-minus-control1024-pentobi4", "Teacher − control"),
        ("control1024-minus-parent-pentobi4", "Control − parent"),
    ]:
        row = result["paired_differences"][key]
        center = row["score_difference"] * 100
        limits = np.asarray(row["nominal_paired_95pct_interval"]) * 100
        labels.append(label)
        differences.append(center)
        lo.append(center - limits[0])
        hi.append(limits[1] - center)
    axes[1].errorbar(differences, np.arange(3), xerr=[lo, hi], fmt="o", color=colors[2], capsize=3)
    axes[1].axvline(0, color="#999999", linewidth=0.8, linestyle="--")
    axes[1].set(
        yticks=np.arange(3),
        yticklabels=labels,
        xlabel="Paired Pentobi 4 difference, percentage points",
    )
    axes[1].invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fresh seed 1070: held-out results at equal 2,048 playing simulations")
    caption = "100 whole-opening pairs; nominal 95% bootstrap intervals; one parent/training seed"
    if any(
        row.get("zero_score_one_sided_95_upper") is not None for row in result["external"].values()
    ):
        caption += " (zero scores: one-sided upper)"
    fig.supxlabel(caption, fontsize=9)
    figure = ROOT / "docs/assets/integrated-training-results.png"
    fig.savefig(figure, dpi=180)
    lines = [
        "\n## Verified full-study results\n",
        f"Primary verdict: **{result['primary_verdict']}**. All 1,200 external action traces were independently replayed. All 800 head-to-head outcomes and counters were audited; those games have no saved action traces.\n",
        "![Held-out results](assets/integrated-training-results.png)\n",
        "| Case | W–D–L | Win rate | Score (draw half) | Nominal 95% score interval | Mean margin | Game seconds | Actual actor simulations | Neural positions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        for level in (3, 4):
            row = result["external"][f"{name}-pentobi{level}"]
            ci = interval(row["score_rate_95pct_cluster_bootstrap"])
            if row.get("zero_score_one_sided_95_upper") is not None:
                ci = "All zero; conservative one-sided 95% upper " + pct(
                    row["zero_score_one_sided_95_upper"]
                )
            metrics = row["policy_metrics"]
            lines.append(
                f"| {names[name]} vs level {level} | {row['wins']}–{row['draws']}–{row['losses']} | {pct(row['wins'] / row['games'])} | {pct(row['score_rate'])} | {ci} | {row['mean_margin']:.2f} | {row['sum_game_seconds']:.1f} | {metrics['simulations']:,} | {metrics['neural_positions']:,} |"
            )
    lines += [
        "\nAll cases use 2,048 simulations per searched actor decision. Game seconds include both agents and game work, not pure neural or GPU latency.\n",
        "| Paired comparison | Score difference | Nominal 95% whole-opening interval |",
        "|---|---:|---:|",
    ]
    for name, row in result["paired_differences"].items():
        lines.append(
            f"| {name} | {100 * row['score_difference']:+.2f}pp | {100 * row['nominal_paired_95pct_interval'][0]:+.2f} to {100 * row['nominal_paired_95pct_interval'][1]:+.2f}pp |"
        )
    lines += [
        "\n| Branch | Completed rounds / new games | Recorded roots | Gradients | Actual simulations | Collection seconds | Training seconds | Total branch seconds / soft cap | Selected additional round / games |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["training"].items():
        lines.append(
            f"| {names[name]} | {row['rounds']} / {row['games']:,} | {row['new_training_positions']:,} | {row['gradient_steps']:,} | {row['simulations']:,} | {row['collect_seconds']:.1f} | {row['training_seconds']:.1f} | {row['elapsed_seconds']:.1f} / {row['soft_cap_seconds']} | {row['selected_additional_round']} / {row['selected_additional_games']:,} |"
        )
    lines += [
        "\n| Head-to-head | W–D–L | Score | Nominal 95% whole-opening interval | Mean margin | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["head_to_head"].items():
        lines.append(
            f"| {name} | {row['wins']}–{row['draws']}–{row['losses']} | {pct(row['score_rate'])} | {interval(row['score_rate_95pct_cluster_bootstrap'])} | {row['mean_margin']:.2f} | {row['seconds']:.1f} |"
        )
    lines += [
        "\nRaw and searched results above remain separate. Validation chose checkpoints; the held-out results did not. The full verification JSON retains validation rows, runtime/source hashes, optimizer/replay checks, exact actor decisions and all paired scores.\n",
        "| Provider app | Actual metering USD |",
        "|---|---:|",
    ]
    for name, cost in result["provider_metering_usd"].items():
        lines.append(f"| {name} | ${cost} |")
    lines += [
        f"\nCombined net reservation: ${result['reserved_usd']:.8f} under $23; reservation is not actual spend. Whole study helper runtime: {result['finished']['seconds']:.1f}s. Billing and app lifetime are separate.\n",
        "A promising primary outcome warrants an independently seeded replication with a prespecified budget; an inconclusive or negative result warrants diagnosis before further training. No next paid run was launched automatically.\n",
    ]
    target = ROOT / "docs/integrated-overnight.md"
    text = target.read_text()
    marker = "\n## Verified full-study results\n"
    tail = ""
    if marker in text:
        start = text.index(marker)
        notes = text.find("\n## Interpretation, experience and resource limits\n", start)
        if notes >= 0:
            tail = text[notes:]
        text = text[:start]
    target.write_text(text + "\n".join(lines) + "\n" + tail)
    print(figure)


if __name__ == "__main__":
    main()
