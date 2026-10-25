"""Rebuild an evidence table and scientific figure from saved Blokus evaluations."""

import json
import os
import shutil
import tempfile
from pathlib import Path


def build_report(root=Path("runs"), destination=Path("docs/blokus-duo-results.md")):
    paths = (
        sorted(root.glob("duo-pilot-seed-*/summary.json"))
        + sorted(root.glob("duo-curriculum-*/summary.json"))
        + sorted(root.glob("duo-longer-*/summary.json"))
        + sorted(root.glob("duo-frontier-training-*/summary.json"))
    )
    runs = [(p.parent.name, json.loads(p.read_text())) for p in paths]
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Blokus Duo results",
        "",
        "Generated from saved summary.json files. Each policy was selected on validation",
        "games before the held-out tests below. W/D/L count strict wins, draws and losses.",
        "The learner is greedy; neural opponents sample their frozen distributions.",
        "",
        "## Cold-start pilots",
        "",
        "Each pilot starts from random weights and trains for 1,600 games against random",
        "legal play. Different seeds independently initialize weights and experience.",
        "",
        "| Seed | Initial vs random W/D/L | Selected vs random W/D/L | Selected vs largest W/D/L | Selected vs sampled initial W/D/L |",
        "|---|---|---|---|---|",
    ]

    def counts(x):
        return f"{x['wins']}/{x['draws']}/{x['losses']}"

    for name, run in runs:
        if "pilot" not in name:
            continue
        h = run["held_out"]
        lines.append(
            f"| {run['config']['seed']} | {counts(h['initial']['random'])} | "
            f"{counts(h['selected']['random'])} | {counts(h['selected']['largest'])} | "
            f"{counts(h['selected_vs_initial'])} |"
        )
    lines += [
        "",
        "Each entry has 200 games, balanced between seats. All three pilots improve",
        "against random and their sampled initial policy. This is a small cold-start",
        "learning result; it does not establish human-level play.",
        "",
        "## Follow-up curricula",
        "",
        "Each follow-up adds 6,400 games. Seed 142 arms share the seed-42 pilot parent.",
        "League seeds 143/144 use independent pilot parents 43/44. Selection maximizes",
        "mean validation score rate against random and largest-piece play. The margin",
        "arm changes the learning objective to terminal score difference / 109.",
        "",
        "| Run | Selected update | vs random W/D/L | vs largest W/D/L | Largest strict-win 95% interval | Largest wins by seat 0 / 1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, run in runs:
        if "curriculum" not in name:
            continue
        h = run["held_out"]["selected"]
        lower, upper = h["largest"]["win_rate_95pct_wilson"]
        seats = h["largest"]["by_seat"]
        lines.append(
            f"| {name.removeprefix('duo-curriculum-')} | {run['best_update']} | "
            f"{counts(h['random'])} | {counts(h['largest'])} | {lower:.1%}–{upper:.1%} | "
            f"{seats['0']['win_rate']:.1%} / {seats['1']['win_rate']:.1%} |"
        )
    lines += [
        "",
        "Follow-up entries have 400 games each, balanced between seats. The intervals",
        "describe game sampling uncertainty for a fixed policy/opponent, not uncertainty",
        "across training seeds. These arms are exploratory comparisons; test results",
        "informed later hypotheses, which use new final test seeds.",
        "",
        "![Blokus learning results](assets/blokus-duo-results.png)",
        "",
        "## What the evidence supports",
        "",
        "- We can train neural Blokus Duo policies from random initialization with the same PyTorch/TorchRL stack.",
        "- Baseline choice matters: beating random is easier than beating largest-piece play.",
        "- Compare selected and final checkpoints; additional training can regress.",
        "- These finite benchmarks cannot certify optimality, broad robustness or human-level strength.",
        "",
        "No human demonstrations, minimax targets or tree-search policy targets were used for these runs.",
        "Largest/mixed curricula use a handcrafted opponent, which is a declared source",
        "of human domain knowledge. The league uses sampled historical neural policies.",
        "",
        "The original pilot deterministic-neural match repeated only two trajectories.",
        "It was corrected to sampled neural opponents; original summaries are preserved",
        "as summary-before-neural-opponent-fix.json and are superseded.",
        "",
        "## Reproduce",
        "",
        "See [pilot protocol](blokus-duo-protocol.md),",
        "[follow-up protocol](blokus-duo-curriculum-protocol.md), and",
        "[training walkthrough](blokus-duo.md). Run artifacts and weights live under",
        "the ignored runs/ directory; the source tree does not bundle model weights.",
        "",
        "Rebuild this report with `uv run blokus-rl duo-report`.",
        "",
    ]
    extra = []
    for name, run in runs:
        if not name.startswith("duo-longer-"):
            continue
        extra += [
            "## Longer-budget training",
            "",
            f"`{name}` adds {run['training_games']:,} games; selected update {run['best_update']}.",
            "The parent and selected checkpoint are tested on the same fresh seeds.",
            "",
            "| Policy | Opponent | W/D/L | Strict-win 95% interval |",
            "|---|---|---|---|",
        ]
        for label in ("initial", "selected"):
            for opponent, result in run["held_out"][label].items():
                lower, upper = result["win_rate_95pct_wilson"]
                extra.append(
                    f"| {label} | {opponent} | {counts(result)} | {lower:.1%}–{upper:.1%} |"
                )
        gain = (
            run["held_out"]["selected"]["largest"]["win_rate"]
            - run["held_out"]["initial"]["largest"]["win_rate"]
        )
        passed = gain >= 0.05 and run["held_out"]["selected"]["random"]["score_rate"] >= 0.90
        extra += [
            "",
            f"Declared improvement criterion: **{'passed' if passed else 'failed'}**; largest-opponent win-rate change {gain:+.1%}.",
            "",
        ]
        curve = root / name / "learning_curve.png"
        if curve.exists():
            (destination.parent / "assets").mkdir(exist_ok=True)
            shutil.copyfile(curve, destination.parent / "assets" / f"{name}.png")
            extra += [
                f"![Longer-run validation curve](assets/{name}.png)",
                "",
                "The curve shows validation data; the table above shows held-out tests.",
                "",
            ]
    for name, run in runs:
        if not name.startswith("duo-frontier-training-"):
            continue
        extra += [
            "## Frontier curriculum",
            "",
            f"`{name}` adds {run['training_games']:,} games against frontier, with observation version 2.",
            f"Selected update {run['best_update']}; validation selection balances largest and frontier.",
            "",
            "| Policy | Opponent | W/D/L | Strict-win 95% interval |",
            "|---|---|---|---|",
        ]
        for label in ("initial", "selected"):
            for opponent, result in run["held_out"][label].items():
                lower, upper = result["win_rate_95pct_wilson"]
                extra.append(
                    f"| {label} | {opponent} | {counts(result)} | {lower:.1%}–{upper:.1%} |"
                )
        selected = run["held_out"]["selected"]
        gain = selected["frontier"]["win_rate"] - run["held_out"]["initial"]["frontier"]["win_rate"]
        passed = (
            gain >= 0.05
            and selected["frontier"]["win_rate_95pct_wilson"][0] > 0.5
            and selected["random"]["score_rate"] >= 0.9
        )
        extra += [
            "",
            f"Declared improvement criterion: **{'passed' if passed else 'failed'}**; frontier win-rate change {gain:+.1%}.",
            "",
        ]
    frontier_paths = (
        "duo-frontier-vs-largest.json",
        "duo-margin-vs-frontier.json",
        "duo-longer-vs-frontier.json",
    )
    if any((root / p).exists() for p in frontier_paths):
        extra += [
            "## Held-out geometric baseline",
            "",
            "The frontier baseline maximizes piece area, then own-minus-opponent empty",
            "corner anchors, then breaks ties randomly. It was excluded from training",
            "and checkpoint selection for the runs above. This is one-ply heuristic",
            "evaluation, not a trained policy or a perfect opponent.",
            "",
            "| Match | W/D/L | Strict-win 95% interval |",
            "|---|---|---|",
        ]
        for filename in frontier_paths:
            p = root / filename
            if p.exists():
                result = json.loads(p.read_text())
                lower, upper = result["win_rate_95pct_wilson"]
                extra.append(
                    f"| {filename.removesuffix('.json').removeprefix('duo-')} | {counts(result)} | {lower:.1%}–{upper:.1%} |"
                )
        extra += [""]
    extra += [
        "The first runs used observation version 1. Version 2 appends current scores",
        "and pass count so the completion bonus and exact pass transition are visible.",
        "Old policies load with zero weights on the new inputs; four benchmark",
        "rechecks reproduced identical W/D/L and score margins after migration.",
        "",
    ]
    position = lines.index("## What the evidence supports")
    lines[position:position] = extra
    destination.write_text("\n".join(lines))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blokus-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pilots = [(n, r) for n, r in runs if "pilot" in n]
    followups = [(n, r) for n, r in runs if "curriculum" in n]
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout="constrained")
    x = np.arange(len(pilots))
    for offset, label, color in ((-0.19, "initial", "#919bab"), (0.19, "selected", "#287fa8")):
        values = [r["held_out"][label]["random"]["win_rate"] for _, r in pilots]
        intervals = np.array(
            [r["held_out"][label]["random"]["win_rate_95pct_wilson"] for _, r in pilots]
        )
        axes[0].bar(
            x + offset,
            values,
            width=0.36,
            label=label,
            color=color,
            yerr=np.array([values - intervals[:, 0], intervals[:, 1] - values]),
            capsize=4,
        )
    axes[0].set_xticks(x, [f"Training seed {r['config']['seed']}" for _, r in pilots])
    axes[0].set_title("Cold start: 1,600 games against random play per seed")
    axes[0].set_ylabel("Strict win rate against random")
    axes[0].legend()
    if followups:
        labels = [n.removeprefix("duo-curriculum-") for n, _ in followups]
        values = np.array([r["held_out"]["selected"]["largest"]["win_rate"] for _, r in followups])
        intervals = np.array(
            [r["held_out"]["selected"]["largest"]["win_rate_95pct_wilson"] for _, r in followups]
        )
        axes[1].bar(
            np.arange(len(values)),
            values,
            color="#518b62",
            yerr=np.array([values - intervals[:, 0], intervals[:, 1] - values]),
            capsize=4,
        )
        axes[1].set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    axes[1].set_title("Follow-up: 6,400 additional games per policy")
    axes[1].set_ylabel("Strict win rate against largest-piece")
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    assets = destination.parent / "assets"
    assets.mkdir(exist_ok=True)
    fig.savefig(assets / "blokus-duo-results.png", dpi=150)
    fig.savefig(assets / "blokus-duo-results.svg")
    plt.close(fig)
    print(f"Wrote {destination} from {len(runs)} completed runs.")
