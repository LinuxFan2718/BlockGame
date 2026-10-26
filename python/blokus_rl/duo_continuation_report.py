"""Audit and report the frozen continuation study; never launch training or games."""

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
import torch

from ._native import DuoGame
from .duo_crossplay import opening_book, paired_interval
from .duo_deep_report import training_summary

APP_ID = "ap-DQEFQZzvb7k0kP8eAjXRWz"
PARENT_SHA = "0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86"
STATE_SHA = "2df38e9b7585c23fe0b8159cc4511db1811c0b745534d8a9223e885dec1db52d"
ARMS = ("standard256", "strong1024")
NAMES = {"parent": "Parent", "standard256": "Standard 256", "strong1024": "Strong 1024"}
COLORS = {"parent": "#718096", "standard256": "#297c9e", "strong1024": "#b35f32"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def check_pairs(row, count):
    pairs = np.asarray(row["opening_pair_scores"])
    if pairs.shape != (count, 2) or not np.isin(pairs, (0, 0.5, 1)).all():
        raise ValueError("invalid opening-pair outcomes")
    expected = {
        "games": 2 * count,
        "opening_pairs": count,
        "wins": int(np.sum(pairs == 1)),
        "draws": int(np.sum(pairs == 0.5)),
        "losses": int(np.sum(pairs == 0)),
        "score_rate": float(pairs.mean()),
        "score_rate_95pct_cluster_bootstrap": paired_interval(pairs),
    }
    if any(row[key] != value for key, value in expected.items()):
        raise ValueError("stored summary disagrees with paired outcomes")
    return pairs


def paired_difference(first, second):
    # Subtract within the same opening and seat, then resample whole openings.
    return {
        "difference": first["score_rate"] - second["score_rate"],
        "interval": paired_interval(
            np.asarray(first["opening_pair_scores"]) - np.asarray(second["opening_pair_scores"])
        ),
    }


def verify(run, parent, parent_state, preserved):
    finished, protocol, heldout = [
        read(run / name)
        for name in ("finished.json", "study-protocol.json", "heldout-protocol.json")
    ]
    if (run / "failure.json").exists():
        raise ValueError("failure marker present")
    if digest(parent) != PARENT_SHA or digest(parent_state) != STATE_SHA:
        raise ValueError("parent model or optimizer/replay changed")
    for path, sha in read(preserved).items():
        if digest(path) != sha:
            raise ValueError(f"certified artifact changed: {path}")
    if heldout != {
        "checkpoint_sha256": finished["checkpoint_sha256"],
        "parent_sha256": PARENT_SHA,
        "seed": 2589000,
        "pairs": 100,
        "simulations": 128,
        "precision": "bfloat16",
    }:
        raise ValueError("reserved held-out protocol changed")
    if protocol["parent_sha256"] != PARENT_SHA or protocol["source_resume_sha256"] != STATE_SHA:
        raise ValueError("wrong lineage")
    configs = protocol["configs"]
    differences = {k for k in configs[ARMS[0]] if configs[ARMS[0]][k] != configs[ARMS[1]][k]}
    if differences != {"full_simulations", "max_seconds"}:
        raise ValueError("unexpected training treatment difference")
    tensors = {"parent": torch.load(parent, map_location="cpu", weights_only=True)["model_state"]}
    training = {}
    original = torch.load(parent_state, map_location="cpu", weights_only=False)
    original_steps = {int(v["step"]) for v in original["optimizer"]["state"].values()}
    if len(original_steps) != 1:
        raise ValueError("inconsistent inherited optimizer steps")
    parent_steps = original_steps.pop()
    del original
    for arm, full, cap in (("standard256", 256, 7200), ("strong1024", 1024, 18000)):
        config, arm_protocol = configs[arm], read(run / arm / "protocol.json")
        if config["full_simulations"] != full or config["max_seconds"] != cap:
            raise ValueError("search budget/cap changed")
        if arm_protocol["config"] != config or arm_protocol["parent_games"] != 32768:
            raise ValueError("branch configuration changed")
        if arm_protocol["inherited_replay_positions"] != 100000:
            raise ValueError("inherited replay changed")
        if (
            arm_protocol["source_resume_sha256"] != STATE_SHA
            or digest(run / arm / "initial.pt") != PARENT_SHA
        ):
            raise ValueError("branch initialization changed")
        frozen = run / f"frozen-{arm}.pt"
        if digest(frozen) != heldout["checkpoint_sha256"][arm] or digest(frozen) != digest(
            run / arm / "best.pt"
        ):
            raise ValueError("selected frozen checkpoint changed")
        summary = training_summary(run, arm)
        if read(run / arm / "completed.json") != finished["training"][arm]:
            raise ValueError("completion markers disagree")
        rows = summary["rows"]
        if (
            summary["rounds"] != 128
            or summary["games"] != 32768
            or summary["reason"] != "round_limit"
        ):
            raise ValueError("planned games did not complete")
        if summary["elapsed_seconds"] > cap:
            raise ValueError("soft training cap exceeded")
        candidates = [(0.5, 0)]
        for row in rows:
            if row["games"] != 256 * row["round"] or row["replay_positions"] != 100000:
                raise ValueError("invalid training counters")
            if not all(np.isfinite(v) for v in row["losses"].values()):
                raise ValueError("nonfinite training")
            if "validation" in row:
                check_pairs(row["validation"], 32)
                candidates.append((row["validation"]["score_rate"], row["round"]))
        selected_score, selected_round = max(candidates, key=lambda pair: pair[0])
        if (
            selected_round != summary["selected_round"]
            or selected_score != summary["best_validation_score"]
        ):
            raise ValueError("selection rule disagrees with frozen checkpoint")
        state = torch.load(run / arm / "resume.pt", map_location="cpu", weights_only=False)
        latest = torch.load(run / arm / "latest.pt", map_location="cpu", weights_only=True)
        if state["round"] != 128 or state["games"] != 32768 or len(state["replay"]) != 100000:
            raise ValueError("atomic round state is incomplete")
        if any(
            not torch.equal(value, latest["model_state"][key])
            for key, value in state["model_state"].items()
        ):
            raise ValueError("latest/atomic state weights disagree")
        if {int(v["step"]) for v in state["optimizer"]["state"].values()} != {
            parent_steps + summary["gradient_steps"]
        }:
            raise ValueError("optimizer steps disagree with inherited plus new updates")
        selected_rows = rows[:selected_round]
        summary["selected_games"] = 256 * selected_round
        summary["selected_new_roots"] = sum(r["new_training_positions"] for r in selected_rows)
        summary["selected_gradient_steps"] = sum(r["gradient_steps"] for r in selected_rows)
        tensors[arm] = torch.load(frozen, map_location="cpu", weights_only=True)["model_state"]
        training[arm] = summary
        del state, latest
    book = opening_book(100, seed=2589000)
    book_sha = hashlib.sha256(json.dumps(book).encode()).hexdigest()
    cases = {}
    replayed, searched_decisions = 0, 0
    for mode, budget in (("raw", 0), ("search128", 128)):
        for case in (f"strong-vs-standard-{mode}", *(f"{a}-vs-parent-{mode}" for a in ARMS)):
            row = read(run / f"heldout-{case}.json")
            check_pairs(row, 100)
            if row["learner_simulations"] != budget or row["opponent_simulations"] != budget:
                raise ValueError("wrong held-out playing budget")
            cases[case] = row
        for agent in ("parent", *ARMS):
            case = f"{agent}-vs-pentobi-{mode}"
            row = read(run / f"heldout-{case}.json")
            pairs = check_pairs(row, 100)
            if row["settings"] != {
                "model_sha256": tensor_digest(tensors[agent]),
                "seed": 2589000,
                "simulations": budget,
                "precision": "bfloat16",
                "openings_sha256": book_sha,
            }:
                raise ValueError("Pentobi evaluation model/book/settings changed")
            partial = read(run / f"heldout-{case}.partial.json")
            if partial != {"settings": row["settings"], "games": row["games_detail"]}:
                raise ValueError("atomic pair records disagree with final results")
            if len(row["games_detail"]) != 200:
                raise ValueError("incomplete external games")
            margins, roots = [], 0
            for j, detail in enumerate(row["games_detail"]):
                index, seat = divmod(j, 2)
                if detail["seat"] != seat or detail["seed"] != 2589000 + index + seat * 1000000:
                    raise ValueError("wrong seat/seed")
                if detail["actions"][:4] != book[index]:
                    raise ValueError("wrong opening")
                game = DuoGame()
                for ply, action in enumerate(detail["actions"]):
                    roots += budget > 0 and ply >= 4 and game.current_player == seat
                    game.step(action)
                margin = game.scores[seat] - game.scores[1 - seat]
                if (
                    not game.terminated
                    or list(game.scores) != detail["scores"]
                    or margin != detail["margin"]
                ):
                    raise ValueError("external saved game replay disagrees")
                if pairs[index, seat] != (1 if margin > 0 else 0.5 if margin == 0 else 0):
                    raise ValueError("saved outcome disagrees with replayed margin")
                margins.append(margin)
                replayed += 1
            if row["mean_margin"] != float(np.mean(margins)):
                raise ValueError("external mean margin disagrees")
            if row["metrics_since_resume"].get("simulations", 0) != budget * roots:
                raise ValueError("external actual search budget disagrees")
            searched_decisions += roots
            cases[case] = row
    diagnostics = {}
    for budget in (512, 2048):
        row = read(run / f"diagnostic-parent-search{budget}-vs-search128.json")
        check_pairs(row, 24)
        if row["learner_simulations"] != budget or row["opponent_simulations"] != 128:
            raise ValueError("diagnostic budget changed")
        diagnostics[str(budget)] = row
    audit = {
        "checkpoint_sha256": heldout["checkpoint_sha256"],
        "parent_sha256": PARENT_SHA,
        "source_resume_sha256": STATE_SHA,
        "preserved_certified_files": len(read(preserved)),
        "parent_optimizer_steps": parent_steps,
        "test_cases": len(cases),
        "test_games": sum(r["games"] for r in cases.values()),
        "external_games_replayed": replayed,
        "external_searched_decisions": searched_decisions,
        "external_expected_simulations": 128 * searched_decisions,
        "openings_sha256": book_sha,
    }
    return cases, training, diagnostics, audit, finished


def draw(cases, training, path):
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "blokus-continuation-20261026",
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.2), layout="constrained")
        for arm in ARMS:
            rows = [r for r in training[arm]["rows"] if "validation" in r]
            axes[0].plot(
                [r["games"] / 1000 for r in rows],
                [r["validation"]["score_rate"] * 100 for r in rows],
                "o-",
                color=COLORS[arm],
                label=NAMES[arm],
                markersize=3,
            )
        axes[0].set(
            title="Validation: selection only",
            xlabel="Additional games (thousands)",
            ylabel="Score vs frozen parent (%)",
            ylim=(40, 100),
        )
        axes[0].legend(fontsize=8)
        for y, mode in enumerate(("raw", "search128")):
            row = cases[f"strong-vs-standard-{mode}"]
            value = row["score_rate"] * 100
            lo, hi = np.array(row["score_rate_95pct_cluster_bootstrap"]) * 100
            axes[1].errorbar(
                value,
                y,
                xerr=[[value - lo], [hi - value]],
                fmt="o",
                capsize=3,
                color=COLORS["strong1024"],
            )
        axes[1].set(
            yticks=[0, 1],
            yticklabels=["Raw", "Search 128\n(primary)"],
            ylim=(-0.6, 1.6),
            xlim=(0, 100),
            title="Held-out: strong vs standard",
            xlabel="Strong score (%)",
        )
        for i, agent in enumerate(("parent", *ARMS)):
            for j, mode in enumerate(("raw", "search128")):
                row = cases[f"{agent}-vs-pentobi-{mode}"]
                value = row["score_rate"] * 100
                lo, hi = np.array(row["score_rate_95pct_cluster_bootstrap"]) * 100
                axes[2].errorbar(
                    value,
                    i + (j - 0.5) * 0.22,
                    xerr=[[value - lo], [hi - value]],
                    fmt="o" if j else "s",
                    capsize=3,
                    color=COLORS[agent],
                    markerfacecolor=COLORS[agent] if j else "white",
                    label=("Search 128" if j else "Raw") if i == 0 else None,
                )
        axes[2].set(
            yticks=range(3),
            yticklabels=[NAMES[a] for a in ("parent", *ARMS)],
            ylim=(-0.5, 2.5),
            xlim=(0, 100),
            title="Held-out: Pentobi 31.0 level 1",
            xlabel="Our score (%)",
        )
        axes[2].legend(fontsize=8)
        for ax in axes:
            (ax.axhline if ax is axes[0] else ax.axvline)(
                50, color="#999999", linestyle="--", linewidth=1
            )
            ax.grid(alpha=0.15)
        fig.suptitle(
            "More experience and stronger self-play search · same small CNN\nHeld-out: 100 fresh, seat-swapped openings · nominal paired 95% intervals",
            fontsize=11,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        fig.savefig(path.with_suffix(".svg"), metadata={"Date": None})
        plt.close(fig)


def table(labelled):
    lines = [
        "| Comparison | W/D/L | Score | Nominal paired 95% interval | Mean margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in labelled:
        lo, hi = row["score_rate_95pct_cluster_bootstrap"]
        lines.append(
            f"| {label} | {row['wins']}/{row['draws']}/{row['losses']} | {row['score_rate']:.2%} | {lo:.2%}–{hi:.2%} | {row['mean_margin']:.2f} |"
        )
    return lines


def render(args):
    run, out = args.run, args.out
    cases, training, diagnostics, audit, finished = verify(
        run, args.parent, args.parent_state, args.preserved
    )
    app = next(r for r in read(args.apps) if r["app_id"] == APP_ID)
    if app["state"] != "stopped" or int(app["tasks"]):
        raise ValueError("study app must be stopped")
    cost = sum(
        (Decimal(r["cost"]) for r in read(args.billing) if r["object_id"] == APP_ID), Decimal(0)
    )
    seconds = (
        datetime.fromisoformat(app["stopped_at"]) - datetime.fromisoformat(app["created_at"])
    ).total_seconds()
    differences = {
        f"{a}-vs-{b}-{m}": paired_difference(
            cases[f"{a}-vs-pentobi-{m}"], cases[f"{b}-vs-pentobi-{m}"]
        )
        for a, b in (
            ("standard256", "parent"),
            ("strong1024", "parent"),
            ("strong1024", "standard256"),
        )
        for m in ("raw", "search128")
    }
    samples = []
    for line in (run / "telemetry.jsonl").read_text().splitlines():
        row = json.loads(line)
        value = row.get(
            "gpu_busy_percent,memory_io_percent,memory_used_mib,memory_total_mib,power_watts"
        )
        if value:
            samples.append([float(v.strip()) for v in value.split(",")])
    telemetry = np.asarray(samples)
    audit |= {
        "provider_metered_usd": str(cost),
        "app_lifetime_seconds": seconds,
        "paired_pentobi_differences": differences,
        "training": {a: {k: v for k, v in t.items() if k != "rows"} for a, t in training.items()},
        "telemetry": {
            "samples": len(samples),
            "mean_gpu_busy_percent": float(telemetry[:, 0].mean()),
            "p95_gpu_busy_percent": float(np.quantile(telemetry[:, 0], 0.95)),
            "peak_vram_mib": float(telemetry[:, 2].max()),
            "total_vram_mib": float(telemetry[:, 3].max()),
        },
    }
    figure = out.parent / "assets/continuation-results.png"
    draw(cases, training, figure)
    primary = cases["strong-vs-standard-search128"]
    lo, hi = primary["score_rate_95pct_cluster_bootstrap"]
    lines = [
        "# Controlled Blokus Duo continuation — results, 26 October 2026",
        "",
        "**Stronger self-play search improved this small CNN, and the resulting search agent beat Pentobi level 1 on the reserved test.** Both continuations improved over their shared parent. This is progress in Duo, not evidence that we have reached the Blokus frontier.",
        "",
        f"The prespecified primary result was **{primary['score_rate']:.2%}** for strong1024 against standard256 at **128 playing simulations each** (nominal paired 95% interval **{lo:.2%}–{hi:.2%}**). Its lower bound exceeds 50%, meeting the declared success criterion for this parent/seed.",
        "",
        "![Controlled continuation results](assets/continuation-results.png)",
        "",
        "## Held-out comparisons",
        "",
        "Each row is 200 games:100 fresh four-ply random openings, replayed in both seats, seed2589000. Score = (wins + half draws)/games. Raw selects directly from the network; search128 uses the same network with 128 tree-search simulations per move. Models were selected on separate validation openings and frozen before this test book was generated. Margins are final score differences from the named first agent's perspective.",
        "",
        *table(
            [
                (f"Strong vs standard · {m}", cases[f"strong-vs-standard-{m}"])
                for m in ("search128", "raw")
            ]
        ),
        "",
        *table(
            [
                (f"{NAMES[a]} vs parent · {m}", cases[f"{a}-vs-parent-{m}"])
                for a in ARMS
                for m in ("raw", "search128")
            ]
        ),
        "",
        "## External opponent: Pentobi 31.0 level 1",
        "",
        *table(
            [
                (f"{NAMES[a]} · {m}", cases[f"{a}-vs-pentobi-{m}"])
                for a in ("parent", *ARMS)
                for m in ("raw", "search128")
            ]
        ),
        "",
        "Pentobi used one thread, no opening book and no resignation. Its final margins were cross-checked by the Rust engine during play; this report independently replayed all 1,200 saved external games. All three agents faced the same fresh openings, seats and per-game Pentobi seeds. The earlier CPU float32 levels 1–9 ladder tested the parent on a different book; those numbers are not a controlled before/after comparison with these CUDA bfloat16 results.",
        "",
        "Same-book score differences below resample the same opening indices together. Units are percentage points; these are secondary comparisons.",
        "",
        "| Difference against Pentobi | Mode | Score difference | Nominal paired 95% interval |",
        "| --- | --- | ---: | ---: |",
    ]
    for key, row in differences.items():
        label, mode = key.rsplit("-", 1)
        low, high = row["interval"]
        lines.append(
            f"| {label} | {mode} | {row['difference'] * 100:+.2f}pp | {low * 100:+.2f} to {high * 100:+.2f}pp |"
        )
    lines += [
        "",
        "## What was trained and selected",
        "",
        "Both branches inherited the same 52,978-parameter four-layer CNN, AdamW optimizer moments and 100,000 replay positions from the parent's 32,768 ancestral games. A declared fresh seed 865 starts additional pure self-play; no Pentobi moves, external demonstrations or heuristic rewards were used for learning.",
        "",
        "On 80% of moves both branches used 32 simulations; these roots were not recorded. On the remaining 20%, standard used 256 simulations and strong used 1,024. Only these full-search roots entered replay. Shared GPU inference served six CPU self-play workers, 32 live games each, batch limit 256, leaf batch 8. Learning retained replay 100,000, approximately three examples/new root, minibatch 256, AdamW lr 0.001, policy CE weight 0.5 plus terminal win/draw/loss value MSE and gradient clipping 1.",
        "",
        "Every 16 rounds, each branch faced the parent on 32 validation opening pairs, seed 2583000, at 128 simulations each. The unchanged parent was also a selectable candidate at 50%. The best validation checkpoint was frozen; validation scores below are selected results, not held-out strength estimates.",
        "",
        "| Actual additional budget | Standard256 | Strong1024 |",
        "| --- | ---: | ---: |",
    ]
    fields = [
        ("Games generated", "games", ","),
        ("Full-search roots recorded", "new_training_positions", ","),
        ("Gradient updates", "gradient_steps", ","),
        ("Gradient training examples", "gradient_examples", ","),
        ("Search simulations (fast + full)", "simulations", ","),
        ("Neural inference positions", "neural_positions", ","),
        ("Mean shared inference batch", "mean_gpu_batch", ".2f"),
        ("Elapsed training seconds", "elapsed_seconds", ".2f"),
        ("Selected additional round", "selected_round", ","),
        ("Games through selected round", "selected_games", ","),
        ("Roots through selected round", "selected_new_roots", ","),
        ("Updates through selected round", "selected_gradient_steps", ","),
        ("Best validation score", "best_validation_score", ".2%"),
    ]
    for label, key, fmt in fields:
        lines.append(
            f"| {label} | {format(training[ARMS[0]][key], fmt)} | {format(training[ARMS[1]][key], fmt)} |"
        )
    lines += [
        "",
        "Both arms finished 128 rounds / 32,768 additional games at the round limit, within their 2-hour / 5-hour soft caps. Each latest lineage has 65,536 ancestral + new games; the selected strong checkpoint is from additional round 112 (28,672 new games; 61,440 including its ancestor), while the selected standard is from round 128. The final 16 strong rounds were generated but not selected. Equal generated-game budgets do not mean equal compute, gradient examples or selected-checkpoint experience. Stronger search also changes trajectories and terminal outcomes, so this tests the whole search-budget treatment rather than target quality alone. Neural inference positions are not gradient training examples.",
        "",
        "## Frozen-parent playing-search diagnostics",
        "",
        "Before learning, the unchanged parent played itself on 24 separate diagnostic pairs (48 games), seed 2581000. These measure extra playing search, not learning gains or strength against Pentobi.",
        "",
        *table([(f"Parent search{k} vs parent search128", r) for k, r in diagnostics.items()]),
        "",
        "## Hardware, costs and execution",
        "",
        f"One L40S, 8 capped CPU cores, 24 GiB capped RAM; bfloat16 inference/training with TF32 permitted for applicable float32 operations. The bounded function took {finished['seconds']:,.2f} seconds ({finished['seconds'] / 3600:.2f} hours); app lifetime including startup/shutdown was{seconds:,.0f} seconds. No training cap mismatch, provider preemption, retry or scientific-protocol change occurred. A separate 16-game CUDA interruption/recovery smoke passed and its weights were discarded. The CLI's final-log-reader timeout happened after finished.json was committed and the app stopped successfully; it was not a failed experiment.",
        "",
        f"Provider metering for this app at the saved billing snapshot is **${cost:.8f}**, versus the conservative **$22.74661 reservation** and **$25 phase ceiling**. Account billing can lag; the reservation is not a charge. The workspace summary at that snapshot is ${read(args.billing_summary)['metered_cost']} metered, ${float(read(args.billing_summary)['billed_cost']):.2f} billed after credits/allowances. Against the original $130 allowance, approximately${130 - float(read(args.billing_summary)['metered_cost']):.2f} remains before later storage/metering adjustments. The$50 reserve stays uncommitted; no new compute was launched.",
        "",
        f"Across {len(samples):,} telemetry samples (all phases, startup included), mean GPU busy was {telemetry[:, 0].mean():.1f}%, 95th percentile {np.quantile(telemetry[:, 0], 0.95):.1f}%; peak used VRAM was {telemetry[:, 2].max():,.0f}/{telemetry[:, 3].max():,.0f}MiB. Compute utilization and occupied GPU memory describe different things. The training table reports actual batching and end-to-end elapsed time, not an extrapolated GPU-only throughput.",
        "",
        "## Verification and practical limits",
        "",
        f"Verified frozen file hashes against both heldout-protocol.json and finished.json, branch initial weights against the unchanged parent, the unchanged parent optimizer/replay hash, final atomic states against latest weights and inherited+additional optimizer steps, all 128 progress rows/arm, checkpoint selection from validation, and all 12 held-out pair summaries ({audit['test_games']:,} games). All1,200 external action traces replayed legally with matching scores and outcomes; external actor search totaled {audit['external_expected_simulations']:,} simulations. Head-to-head evaluations saved paired outcomes and aggregate margins but no action traces, so independent replay is available for the external games only. Preserved certified tic-tac-toe and frontier342 hashes also pass.",
        "",
        "Intervals use 10,000 whole-opening bootstrap resamples, seed 17. They quantify opening sampling for fixed selected models, not training-seed, parent-lineage, scheduling, precision or hardware uncertainty. Secondary comparisons are nominal and not multiplicity adjusted. Both branches share one parent and one new seed; this is not independent replication. Random four-ply openings define a particular test distribution, not an empty-board tournament or all possible play. We beat level 1 here; we have not benchmarked the new checkpoint against levels 2–9. Pentobi simulation counts and neural-leaf search simulations are not interchangeable measures of work. No human Elo is inferred.",
        "",
        "No AlphaBlokus matches were run. Its inspected agent is for four-player Classic; Duo strength does not establish the Classic frontier. Sensible next work would replicate this training treatment and benchmark the frozen strong agent against higher Pentobi levels, while checking whether its now stronger learned policy changes the benefit of deeper playing search. Those are proposals, not automatically launched jobs.",
        "",
        "## Artifacts and reproduction",
        "",
        f"- Downloaded complete private run: `{run}`.",
        f"- Verification and derived budgets: `{run.parent / 'verification.json'}`; all downloaded file hashes: `{run.parent / 'download-sha256.json'}`.",
        f"- App: `{APP_ID}`; source commit, exact payload/runner hashes, lineage hashes and reservation are retained in execution.json and runs/modal-continuation-cost-ledger.jsonl.",
        "- Selected standard SHA256: `" + audit["checkpoint_sha256"]["standard256"] + "`.",
        "- Selected strong SHA256: `" + audit["checkpoint_sha256"]["strong1024"] + "`.",
        "- The rolling protocol, launch command and original source hashes are in [continuation-status.md](continuation-status.md). Training is finished; do not rerun the launch command.",
        "",
        "Regenerate the audit/report/plot locally without training or evaluation:",
        "",
        "```sh",
        "MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m blokus_rl.duo_continuation_report \\",
        f"  --run {run} \\",
        f"  --parent {args.parent} \\",
        f"  --parent-state {args.parent_state} \\",
        f"  --preserved {args.preserved} \\",
        f"  --billing {args.billing} \\",
        f"  --billing-summary {args.billing_summary} \\",
        f"  --apps {args.apps} \\",
        f"  --out {out}",
        "```",
        "",
    ]
    out.write_text("\n".join(lines))
    (run.parent / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob("*")) if p.is_file()}
    (run.parent / "download-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run",
        "parent",
        "parent-state",
        "preserved",
        "billing",
        "billing-summary",
        "apps",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    render(parser.parse_args())
