"""Regenerate the compact search/self-play report from saved experiment data."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .duo_crossplay import paired_interval


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def curve(path):
    result = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrent trainer may be writing its last line.
    return result


def percentage(value):
    return f"{100 * value:.2f}%"


def row(label, result):
    interval = result.get("score_rate_95pct_cluster_bootstrap")
    ci = "not estimated" if interval is None else "–".join(percentage(v) for v in interval)
    return (
        f"| {label} | {result['games']} | {result['wins']}/{result['draws']}/{result['losses']} "
        f"| {percentage(result['score_rate'])} | {ci} |"
    )


def build():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blokus-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path("runs")
    plan = {str(b): read(root / f"duo-search-confirm-2542000/budget-{b}.json") for b in (128, 512)}
    external = {"raw": read(root / "duo-pentobi-raw100-2551000/results.json")}
    external.update(
        {str(b): read(root / f"duo-pentobi-search{b}-2551000/results.json") for b in (128, 512)}
    )
    training = {
        name: curve(root / f"duo-az-{name}-{seed}/progress.jsonl")
        for name, seed in [("cold", 642), ("warm", 643)]
    }
    heldout = {
        name: read(root / f"duo-az-{name}-heldout-2549000/results.json") for name in training
    }
    controls = {
        name: read(root / f"duo-az-{name}-heldout-2549000/equal-search-control.json")
        for name in training
    }
    complete = all(
        (root / f"duo-az-{name}-heldout-2549000/completed.json").exists() for name in training
    )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "planning": plan,
        "external": external,
        "training": {},
        "heldout": heldout,
        "equal_search_control": controls,
        "paired_external_gains": {},
        "heldout_pentobi_reference": {},
        "heldout_pentobi_gains": {},
    }
    text = [
        "# Blokus search and self-play results",
        "",
        "Status: **complete**."
        if complete
        else "Status: **preliminary; final self-play evaluation pending**.",
        "",
        "Score = win + half a draw. Each reported randomized-opening case uses 100 independently "
        "generated four-ply openings, each played in both seats. The 95% intervals resample entire "
        "two-game opening pairs. Single training seeds do not establish reproducibility across seeds.",
        "",
        "![Results](assets/search-results.png)",
        "",
        "## Planning with unchanged weights",
        "",
        "| Search budget | Games | W/D/L | Score | Paired 95% interval |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    text.extend(row(f"{budget} simulations", result) for budget, result in plan.items() if result)
    text += [
        "",
        "Opponent is the same frozen frontier-342 network playing raw. Seed2542000.",
        "",
        "## Outside opponent: Pentobi 31.0 level 1",
        "",
        "| Policy | Games | W/D/L | Score | Paired 95% interval |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for label, result in external.items():
        if result and result["games"] == 200:
            text.append(row(label if label == "raw" else f"search {label}", result))
    for budget in ("128", "512"):
        if external[budget] and external[budget]["games"] == 200:
            delta = (
                np.asarray(external[budget]["opening_pair_scores"])
                - external["raw"]["opening_pair_scores"]
            )
            summary["paired_external_gains"][budget] = {
                "gain": float(delta.mean()),
                "interval": paired_interval(delta),
            }
    text += [
        "",
        "All rows share seed2551000 openings and Pentobi seeds, use one Pentobi thread, "
        "no book, no resignation, and default avoidance of symmetric draws. Each terminal score "
        "margin is independently checked against Rust. These settings do not imply a human rating.",
        "",
        "## Self-play budgets and learning",
        "",
        "| Run | Finished games | Search simulations | Neural positions evaluated | Wall minutes* |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, rows in training.items():
        counts = {
            "games": rows[-1]["games"] if rows else 0,
            "simulations": sum(r["metrics"]["simulations"] for r in rows),
            "neural_positions": sum(r["metrics"]["neural_positions"] for r in rows),
            "minutes": sum(r["seconds"] for r in rows) / 60,
        }
        summary["training"][name] = counts
        text.append(
            f"| {name} | {counts['games']:,} | {counts['simulations']:,} | {counts['neural_positions']:,} | {counts['minutes']:.2f} |"
        )
    text += [
        "",
        "*Per-run sum of round wall times; the two runs run concurrently. Both use 128 "
        "simulations/move, 3 CPU workers, 32 rounds × 128 games, 100 gradient steps/round and "
        "20,000 replay positions. Cold starts randomly; warm starts from frontier-342. Neither "
        "uses heuristic or Pentobi training labels. Validation curves have **different "
        "reference opponents**: each run plays its own frozen initial network.",
        "",
        "## Reserved held-out self-play evaluation",
        "",
        "Seed2549000 is never used for checkpoint selection. The selected checkpoint is frozen "
        "after training. Raw and search128 are separate agents.",
        "",
        "| Run / opponent | Games | W/D/L | Score | Paired 95% interval |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for name, results in heldout.items():
        for label, result in (results or {}).items():
            if label != "empty_board_vs_reference":
                text.append(row(f"{name}: {label}", result))
    text += [
        "",
        "### Equal search budget: does training help the searched agent?",
        "",
        "| Run vs original searched reference | Games | W/D/L | Score | Paired 95% interval |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for name, result in controls.items():
        if result:
            text.append(row(name + " search128 vs reference search128", result))
    text += [
        "",
        "Both players get 128 simulations/move. This is an additional diagnostic on "
        "the original held-out openings, with already-frozen checkpoints; no model was "
        "selected or changed from these results. Other reported intervals are nominal "
        "95%, with no correction for the many exploratory comparisons.",
    ]
    text += [
        "",
        "The in-repo reference is the original frontier-342 network. Largest specialist "
        "means longer-margin-243; largest/frontier alone mean hand-written baselines. "
        "Normal-start deterministic games are saved separately as exactly two games without "
        "an uncertainty estimate. See each held-out protocol for hashes and frozen copies. "
        "Warm raw_vs_initial and raw_vs_reference use the same opponent and openings; "
        "their duplicate rows are not independent evidence.",
    ]
    text += [
        "",
        "### Matched Pentobi comparison on reserved openings",
        "",
        "| Policy | Games | W/D/L | Score | Paired 95% interval |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for mode in ("raw", "search128"):
        baseline = read(root / f"duo-pentobi-reference-{mode}-2549000/results.json")
        if not baseline or baseline["games"] != 200:
            continue
        summary["heldout_pentobi_reference"][mode] = baseline
        text.append(row("original frontier342 " + mode, baseline))
        key = "raw_vs_pentobi" if mode == "raw" else "search_vs_pentobi"
        for name, results in heldout.items():
            if results and key in results:
                result = results[key]
                text.append(row(name + " " + mode, result))
                delta = np.asarray(result["opening_pair_scores"]) - baseline["opening_pair_scores"]
                summary["heldout_pentobi_gains"][name + "_" + mode] = {
                    "gain": float(delta.mean()),
                    "interval": paired_interval(delta),
                }
    text += [
        "",
        "More search can improve decisions at fixed weights; that does not establish that "
        "training produced a better raw network. Stronger internal results do not establish "
        "human-level play. Degenerate bootstrap intervals do not prove an exact population score.",
        "",
        "Method, commands and preservation details: [rolling experiment](search-experiment.md). "
        "Run `python -m blokus_rl.duo_search_report` to regenerate this report.",
    ]
    followup = root / "duo-az-warm-search-selection-2559000"
    selection = read(followup / "selection.json")
    newtest = read(followup / "test-results.json")
    if selection and newtest:
        summary["search_selection_followup"] = {"selection": selection, "test": newtest}
        text += [
            "",
            "## Follow-up: select checkpoints using searched validation play",
            "",
            "This adaptive follow-up compared rounds 4/8/12/16/20/24/28/32 of the existing "
            "warm-start run on 40 validation opening pairs, both agents using 128 simulations. "
            "It added **zero training games**. Selection used seed2543000; the test used a "
            "fresh seed 2559000, generated only after freezing the selected checkpoint.",
            "",
            f"Selected round: **{selection['selected_round']}**; validation score "
            f"**{percentage(selection['validation_score'])}**. The declared primary success "
            "criterion required a fresh-test searched-vs-searched 95% lower bound above 50%.",
            "",
            "| Fresh test | Games | W/D/L | Score | Paired 95% interval |",
            "| --- | ---: | --- | ---: | --- |",
        ]
        text.extend(row(label, result) for label, result in newtest.items())
        text += [
            "",
            "**Failed** that criterion: selecting on searched validation did not produce "
            "a confirmed improvement on the new test. Do not promote this round 24 checkpoint "
            "over the original selected round 8 model from these results. This is not a "
            "replication of round 8 on the new seed, because a different checkpoint was tested.",
        ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    matplotlib.rcParams["svg.hashsalt"] = "blokus-search-study"

    def points(ax, items, labels):
        for i, result in enumerate(items):
            ci = result["score_rate_95pct_cluster_bootstrap"]
            y = result["score_rate"] * 100
            err = None if ci is None else np.array([[y - ci[0] * 100], [ci[1] * 100 - y]])
            ax.errorbar(i, y, yerr=err, fmt="o", capsize=4, color="#176B87")
        ax.set_xticks(range(len(items)), labels)
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.2)

    labels = [k for k, v in plan.items() if v]
    points(axes[0, 0], [plan[k] for k in labels], labels)
    axes[0, 0].set(title="Search vs same raw network", xlabel="Simulations per move", ylim=(40, 85))
    axes[0, 0].axhline(50, color="gray", ls="--")
    labels = [k for k, v in external.items() if v and v["games"] == 200]
    points(axes[0, 1], [external[k] for k in labels], labels)
    axes[0, 1].set(
        title="Same weights vs Pentobi level 1", xlabel="Raw or search budget", ylim=(0, 30)
    )
    for name, rows in training.items():
        values = [r for r in rows if "validation" in r]
        label = f"{name} vs " + ("untrained initial" if name == "cold" else "frontier initial")
        axes[1, 0].plot(
            [r["games"] for r in values],
            [100 * r["validation"]["score_rate"] for r in values],
            "o-",
            label=label,
        )
    axes[1, 0].set(
        title="Validation: each run vs its own initial network",
        xlabel="Self-play training games",
        ylabel="Score (%)",
        ylim=(0, 100),
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)
    items, labels = [], []
    for name, result in controls.items():
        if result:
            items.append(result)
            labels.append(name)
    if items:
        points(axes[1, 1], items, labels)
    else:
        axes[1, 1].text(
            0.5,
            0.65,
            "Held-out self-play tests pending",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
    axes[1, 1].set(title="Held-out: both agents use search128", ylim=(0, 100))
    axes[1, 1].axhline(50, color="gray", ls="--")
    fig.suptitle("Blokus Duo: search, self-play and an outside benchmark", fontsize=14)
    asset = Path("docs/assets")
    asset.mkdir(exist_ok=True)
    fig.savefig(asset / "search-results.png", dpi=170)
    svg = asset / "search-results.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    Path("docs/search-results.md").write_text("\n".join(text) + "\n")
    (root / "duo-search-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    build()
