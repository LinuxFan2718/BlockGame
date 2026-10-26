"""Verify a completed level benchmark and render its table and scientific plot."""

import argparse
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._native import DuoGame
from .duo_crossplay import opening_book
from .duo_pentobi_ladder import BEST_SHA256, summarize

APP_IDS = ("ap-8J7F0oIUjPAAYL4nbs591O", "ap-aceNu72e6upXDaPHCMAOsZ")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run, old_manifest):
    run = Path(run)
    completed, hardware, protocol = [
        read(run / name) for name in ("completed.json", "hardware.json", "protocol.json")
    ]
    if completed["games"] != 1800 or hardware["gpu"] is not None:
        raise ValueError("expected completed CPU benchmark")
    expected = {
        "opening_pairs": 100,
        "seed": 2579000,
        "simulations": 128,
        "workers": 8,
        "command_timeout": 600,
        "levels": list(range(1, 10)),
    }
    if protocol["config"] != expected or protocol["openings"] != opening_book(100, seed=2579000):
        raise ValueError("benchmark settings/openings changed")
    if protocol["checkpoint_sha256"] != BEST_SHA256 or digest(run / "model.pt") != BEST_SHA256:
        raise ValueError("frozen model changed")
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    old = read(old_manifest)
    if fingerprint != old["protocol_sha256"] or fingerprint != completed["protocol_sha256"]:
        raise ValueError("original experimental protocol changed")
    preserved = 0
    for relative, sha in old["files"].items():
        if Path(relative).name.startswith("pair-") or relative in ("model.pt", "protocol.json"):
            if digest(run / relative) != sha:
                raise ValueError(f"original immutable artifact changed: {relative}")
            preserved += Path(relative).name.startswith("pair-")
    if preserved != 820:
        raise ValueError("expected all820 original pairs")
    source_results, result, decisions = read(run / "results.json"), {}, 0
    for level in range(1, 10):
        records = [read(path) for path in sorted((run / f"level-{level:02}").glob("pair-*.json"))]
        if [r["index"] for r in records] != list(range(100)):
            raise ValueError(f"missing/duplicate opening pairs at level{level}")
        for row in records:
            index = row["index"]
            if row["protocol_sha256"] != fingerprint or row["level"] != level:
                raise ValueError("pair protocol/routing mismatch")
            if [g["seat"] for g in row["games"]] != [0, 1]:
                raise ValueError("missing swapped seat")
            for detail in row["games"]:
                seat, opening = detail["seat"], protocol["openings"][index]
                if detail["seed"] != 2579000 + index + seat * 1000000:
                    raise ValueError("game seed changed")
                if detail["actions"][: len(opening)] != opening:
                    raise ValueError("wrong opening prefix")
                game, roots = DuoGame(), 0
                for ply, action in enumerate(detail["actions"]):
                    roots += ply >= len(opening) and game.current_player == seat
                    game.step(action)
                if not game.terminated or list(game.scores) != detail["scores"]:
                    raise ValueError("replayed score/rules disagreement")
                if game.scores[seat] - game.scores[1 - seat] != detail["margin"]:
                    raise ValueError("wrong actor-perspective margin")
                if detail["policy_metrics"]["simulations"] != 128 * roots:
                    raise ValueError("actual search budget disagreement")
                decisions += roots
        row = summarize(records, 100)
        if row != source_results[str(level)] or row != read(run / f"level-{level:02}/summary.json"):
            raise ValueError("stored summary differs from recomputed games")
        result[str(level)] = row
    audit = {
        "games_replayed": 1800,
        "opening_pairs": 900,
        "unchanged_prior_pairs": preserved,
        "protocol_sha256": fingerprint,
        "checkpoint_sha256": BEST_SHA256,
        "searched_decisions": decisions,
        "expected_simulations": 128 * decisions,
    }
    return result, audit, protocol, hardware


def plot(results, destination):
    levels = np.arange(1, 10)
    scores = np.array([results[str(i)]["score_rate"] for i in levels]) * 100
    margins = [results[str(i)]["mean_margin"] for i in levels]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
        for level, score in zip(levels, scores, strict=True):
            row = results[str(level)]
            if row["zero_score_one_sided_95_upper"] is None:
                low, high = np.array(row["score_rate_95pct_cluster_bootstrap"]) * 100
                axes[0].errorbar(
                    level,
                    score,
                    yerr=[[score - low], [high - score]],
                    fmt="o",
                    color="#286997",
                    capsize=4,
                )
            else:
                high = row["zero_score_one_sided_95_upper"] * 100
                axes[0].plot([level, level], [0, high], color="#777777", linestyle=":")
                axes[0].plot(level, high, marker="v", color="#777777", markersize=5)
                axes[0].plot(level, 0, marker="o", color="#286997")
        axes[0].axhline(50, color="#999999", linewidth=1, linestyle="--")
        axes[0].set(
            ylabel="Our score (%) · draws count half",
            ylim=(-3, 103),
            title="Score and nominal paired 95% intervals",
        )
        axes[0].text(
            0.04,
            0.90,
            "Zero-score levels: triangle marks\none-sided 95% upper bound (2.95%)",
            transform=axes[0].transAxes,
            fontsize=8,
            va="top",
        )
        axes[1].plot(levels, margins, "o-", color="#bb633c")
        axes[1].axhline(0, color="#999999", linewidth=1, linestyle="--")
        axes[1].set(
            ylabel="Mean score margin (our agent − Pentobi)",
            title="Mean final score margin",
            ylim=(-38, 3),
        )
        for ax in axes:
            ax.set(xlabel="Pentobi 31.0 level", xticks=levels, xlim=(0.5, 9.5))
            ax.grid(axis="y", alpha=0.15)
        fig.suptitle(
            "Frozen small CNN · search 128 · Blokus Duo\n200 games / level · 100 shared, seat-swapped openings",
            fontsize=12,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
        fig.savefig(destination.with_suffix(".svg"))
        plt.close(fig)


def render(run, old_manifest, billing, apps, out):
    run, out = Path(run), Path(out)
    rows, audit, protocol, hardware = verify(run, old_manifest)
    app_rows = {row["app_id"]: row for row in read(apps)}
    if any(app_rows[i]["state"] != "stopped" or int(app_rows[i]["tasks"]) for i in APP_IDS):
        raise ValueError("benchmark apps must be stopped")
    costs = {
        app: sum((Decimal(r["cost"]) for r in read(billing) if r["object_id"] == app), Decimal(0))
        for app in APP_IDS
    }
    lifetimes = {
        app: (
            datetime.fromisoformat(app_rows[app]["stopped_at"])
            - datetime.fromisoformat(app_rows[app]["created_at"])
        ).total_seconds()
        for app in APP_IDS
    }
    figure = out.parent / "assets/pentobi-level-results.png"
    plot(rows, figure)
    table = []
    for level, row in rows.items():
        low, high = row["score_rate_95pct_cluster_bootstrap"]
        interval = (
            f"≤{row['zero_score_one_sided_95_upper'] * 100:.2f}% (one-sided)"
            if row["zero_score_one_sided_95_upper"] is not None
            else f"{low * 100:.2f}%–{high * 100:.2f}%"
        )
        table.append(
            f"| {level} | {row['wins']}/{row['draws']}/{row['losses']} | "
            f"{row['score_rate'] * 100:.2f}% | {interval} | {row['mean_margin']:.2f} | "
            f"{row['sum_game_seconds'] / 200:.2f} |"
        )
    sims = sum(row["policy_metrics"]["simulations"] for row in rows.values())
    neural = sum(row["policy_metrics"]["neural_positions"] for row in rows.values())
    assert sims == audit["expected_simulations"]
    text = f"""# Pentobi levels 1–9 benchmark — 26 October 2026

**The frozen small CNN with 128 search simulations remains below Pentobi level 1
under this protocol.** Its level 1 score is 27.50% (nominal paired 95% interval
22.25%–33.00%). Scores fall with stronger levels; levels 6–9 each won all 200 games.
This benchmarks the earlier selected model, not the new continuation agents
whose training is still running.

![Scores and margins](assets/pentobi-level-results.png)

## Complete held-out results

Score is wins plus half draws, divided by games. Each row uses 100 identical
four-ply random opening positions, with both seats played: 200 games/level and
1,800 games overall. Seed 2579000 was fresh and did not select this checkpoint.

| Pentobi level | Our W/D/L | Our score | Nominal 95% uncertainty | Mean margin | Mean game seconds |
| --- | ---: | ---: | --- | ---: | ---: |
{chr(10).join(table)}

The game seconds include both agents' decisions and protocol overhead, under
eight concurrent workers. They are not per-move times or CPU-second totals.
The margin is our final score minus Pentobi's final score.

Intervals resample entire opening pairs 10,000 times, not 200 independent games.
For zero scores, an ordinary bootstrap degenerates to [0,0]; the table instead
shows a conservative one-sided 95% upper bound of 2.95% based on 100 pairs. It does
not establish a true success probability of exactly zero. These are nominal
intervals without adjustment for nine comparisons, and do not include training
seed uncertainty or systematic precision/backend differences.

## What played

- Our agent: frozen 52,978-parameter shallow CNN, selected round 128 after 32,768
  cold-start self-play games; 128 MCTS simulations per decision, CPU float32,
  one PyTorch thread per worker.
- Model SHA256: `{BEST_SHA256}`.
- Pentobi 31.0, pinned commit d5374ddc20eced902788e733d67ee16bd1836fd3;
  Duo, native levels 1–9, one thread, no opening book or resignation, default
  symmetric-draw setting. Its strength levels are not human Elo ratings.
- This is fixed-agent strength testing, not equal wall time or equal search
  compute. Pentobi's simulations and our neural-leaf simulations differ.

Actual actor totals: **{sims:,} search simulations** over
**{audit["searched_decisions"]:,} decisions**, with **{neural:,} neural positions
evaluated**. Every decision's 128-simulation budget was checked from its replayed
turn count. Actual Pentobi node counts were not collected; native level settings
were retained. This is evaluation only, with no new training games or gradients.

## Recovery and verification

Modal preempted the first CPU container. Its infrastructure restart encountered
our existing-directory guard and stopped. Dennis explicitly approved one
recovery. The recovery reused the exact original installed image, binary,
model, code and experimental protocol, computing only 80 missing whole pairs.
All 820 previously saved pair files are byte-for-byte unchanged. All 900 pairs
have the required unique indices, seat/seed assignments and opening prefixes.
All 1,800 games were independently replayed locally: every move was legal and
all terminal scores and margins agreed with the stored Rust/Pentobi results.
Frozen hashes, summaries and simulation accounting were verified independently.

The recovery host emitted unsupported-NNPACK warnings while PyTorch continued
with a fallback. The original protocol already allowed cross-host floating-point
differences: CPU float32 does not promise identical moves across hardware/backends.
The final 160 games (8.9%) ran in recovery. This remains a mixed-host benchmark
of the same frozen policy/search settings; we did not rerun or replace old games
to make the results look uniform. The final log-tail timeout occurred after the
successful return; completed.json, hardware.json and both stopped apps verify
completion. No further games in this level benchmark are scheduled.

## Time and credits

Original app active lifetime: **{lifetimes[APP_IDS[0]] / 3600:.3f}h**;
recovery app lifetime: **{lifetimes[APP_IDS[1]] / 3600:.3f}h**;
combined **{sum(lifetimes.values()) / 3600:.3f}h**, excluding the approval gap.
App lifetimes include startup/cleanup and are not exact useful compute time.
Recovery function duration alone: **{hardware["seconds_this_call"]:.2f}s**;
it must not be mistaken for the entire benchmark duration.

Provider snapshot: original app **${costs[APP_IDS[0]]:.5f}**, approved recovery
**${costs[APP_IDS[1]]:.5f}**, combined **${sum(costs.values()):.5f} metered**.
Workspace charges were covered by existing credits/allowances in the current
summary. Metering/storage/egress accounting can lag. This is distinct from
**$4.76284** conservative reservations under the original **$5** phase ceiling.
No GPU was used by this benchmark. The separate GPU study continues within its
own budget and is not included in these app costs.

## Interpretation

The current benchmark places this frozen agent below the easiest Pentobi level
on these paired random openings. We cannot assign it an interpolated Pentobi
level or human Elo. It sometimes beats levels 1–3, but its score at every level
is below 50%, with the level 1 interval also below 50%.

The earlier GPU-bfloat16 test reported 31.25% against level 1 on a different book
(seed 2569000). Today's 27.50% uses a fresh book and CPU float32. The difference is
not enough to establish model regression: the weights are unchanged, the intervals overlap,
and openings/precision/grouping differ. This table is the complete consistent
level sweep for the declared mixed-host CPU protocol.

The baseline and stronger-search continuations remain a separate experiment
with new validation/test seeds. Nothing in this benchmark changes their frozen
selection or playing-time search budgets. No AlphaBlokus match was run; these
Duo results do not establish four-player Classic or frontier strength.

## Artifacts and reproduction

Rolling audit: [pentobi-level-status.md](pentobi-level-status.md).
Data: `{run}`. Download manifest/verification files are in its parent directory.
Provider snapshots: `{billing}` and `{apps}`. All earlier certified models
are preserved; no checkpoints were overwritten by this report.

```sh
.venv/bin/python -m blokus_rl.duo_pentobi_report --run {run} --old-manifest {old_manifest} --billing {billing} --apps {apps} --out {out}
```

This command validates existing results and renders the report; it starts no
training, games or paid compute.
"""
    out.write_text(text)
    audit.update(
        costs_usd={key: str(value) for key, value in costs.items()}, app_lifetimes_seconds=lifetimes
    )
    (run.parent / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob("*")) if p.is_file()}
    (run.parent / "download-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "old-manifest", "billing", "apps", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(render(args.run, args.old_manifest, args.billing, args.apps, args.out), indent=2)
    )


if __name__ == "__main__":
    main()
