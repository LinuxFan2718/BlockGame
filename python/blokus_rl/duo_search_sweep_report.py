"""Verify saved frozen-policy games and report a playing-search sweep locally."""

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
from .duo_crossplay import opening_book, paired_interval
from .duo_pentobi_ladder import summarize

BUDGETS = (128, 512, 2048)
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
APP_ID = "ap-UrnuIYPY5ezlfrtdOYyGsy"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run):
    protocol, finished = read(run / "protocol.json"), read(run / "finished.json")
    if (run / "failure.json").exists() or finished["games"] != 600:
        raise ValueError("expected complete successful 600-game sweep")
    if protocol["budgets"] != list(BUDGETS) or finished["budgets"] != list(BUDGETS):
        raise ValueError("fixed playing budgets changed")
    if protocol["model_sha256"] != MODEL_SHA or finished["model_sha256"] != MODEL_SHA:
        raise ValueError("frozen model changed")
    shared = {
        "first_level": 2,
        "last_level": 2,
        "opening_pairs": 100,
        "seed": 2601000,
        "device": "cuda",
        "precision": "bfloat16",
    }
    if protocol["shared_config"] != shared:
        raise ValueError("shared evaluation settings changed")
    execution = protocol["execution"]
    ledger = read("runs/modal-search-sweep-cost-ledger.jsonl")
    if (
        execution != ledger
        or digest("runs/duo-playing-search-sweep-2601000-runner.py") != execution["runner_sha256"]
    ):
        raise ValueError("executed source disagrees with launch ledger")
    if digest("runs/duo-strong-pentobi-steps-2599000-module.py") != protocol["evaluator_sha256"]:
        raise ValueError("immutable evaluator disagrees with preserved source")
    book = opening_book(100, seed=2601000)
    binary_sha = read(
        "runs/duo-pentobi-ladder-final-20261026T1550/duo-pentobi-ladder-2579000/protocol.json"
    )["pentobi_binary_sha256"]
    results, budgets = read(run / "results.json"), {}
    if set(results) != {str(b) for b in BUDGETS}:
        raise ValueError("missing budget results")
    for budget in BUDGETS:
        arm = run / f"search-{budget}"
        p, done = read(arm / "protocol.json"), read(arm / "completed.json")
        if p["config"] != shared | {"simulations": budget} or p["openings"] != book:
            raise ValueError("budgets did not use the same book/settings")
        if p["checkpoint_sha256"] != MODEL_SHA or digest(arm / "model.pt") != MODEL_SHA:
            raise ValueError("arm used a different frozen model")
        if p["pentobi_binary_sha256"] != binary_sha or done["games"] != 200:
            raise ValueError("Pentobi binary changed or arm incomplete")
        fingerprint = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
        if done["protocol_sha256"] != fingerprint or done["levels"] != ["2"]:
            raise ValueError("arm completion protocol disagrees")
        records = [read(path) for path in sorted((arm / "level-02").glob("pair-*.json"))]
        if [r["index"] for r in records] != list(range(100)):
            raise ValueError("missing/duplicate pairs")
        decisions = 0
        for row in records:
            if (
                row["level"] != 2
                or row["protocol_sha256"] != fingerprint
                or [g["seat"] for g in row["games"]] != [0, 1]
            ):
                raise ValueError("pair routing or protocol mismatch")
            for detail in row["games"]:
                index, seat = row["index"], detail["seat"]
                if (
                    detail["seed"] != 2601000 + index + seat * 1000000
                    or detail["actions"][:4] != book[index]
                ):
                    raise ValueError("game seed/opening mismatch")
                game, roots = DuoGame(), 0
                for ply, action in enumerate(detail["actions"]):
                    roots += ply >= 4 and game.current_player == seat
                    game.step(action)
                if (
                    not game.terminated
                    or list(game.scores) != detail["scores"]
                    or game.scores[seat] - game.scores[1 - seat] != detail["margin"]
                ):
                    raise ValueError("independently replayed score disagrees")
                if detail["policy_metrics"]["simulations"] != budget * roots:
                    raise ValueError("actual playing budget disagrees")
                decisions += roots
        summary = summarize(records, 100)
        summary["win_rate"] = summary["wins"] / summary["games"]
        if summary != results[str(budget)] or summary != read(arm / "level-02/summary.json"):
            raise ValueError("recomputed summary disagrees")
        if read(arm / "results.json") != {"2": summary}:
            raise ValueError("arm results disagree")
        budgets[str(budget)] = {
            "searched_decisions": decisions,
            "simulations": budget * decisions,
            "neural_positions": summary["policy_metrics"]["neural_positions"],
            "arm_seconds": read(arm / "timing.json")["seconds"],
        }
    for path, sha in read("runs/search-preserved-checkpoints.json").items():
        if digest(path) != sha:
            raise ValueError("preserved certified model changed")
    source = "runs/duo-continuation-final-20261026T2007/duo-continuation-865/frozen-strong1024.pt"
    if digest(source) != MODEL_SHA:
        raise ValueError("preserved selected model changed")
    differences = {}
    for large, small in ((2048, 128), (512, 128), (2048, 512)):
        a, b = results[str(large)], results[str(small)]
        differences[f"{large}-{small}"] = {
            "score_difference": a["score_rate"] - b["score_rate"],
            "paired_95pct_interval": paired_interval(
                np.asarray(a["opening_pair_scores"]) - np.asarray(b["opening_pair_scores"])
            ),
        }
    return results, {
        "games_replayed": 600,
        "model_sha256": MODEL_SHA,
        "budgets": budgets,
        "paired_differences": differences,
        "certified_checkpoints_unchanged": True,
    }


def plot(results, destination):
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "blokus-playing-sweep-2601000",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), layout="constrained")
        scores = np.array([results[str(b)]["score_rate"] for b in BUDGETS]) * 100
        intervals = (
            np.array([results[str(b)]["score_rate_95pct_cluster_bootstrap"] for b in BUDGETS]) * 100
        )
        x = np.arange(3)
        axes[0].errorbar(
            x,
            scores,
            yerr=[scores - intervals[:, 0], intervals[:, 1] - scores],
            fmt="o-",
            color="#247c86",
            capsize=4,
        )
        axes[0].axhline(50, color="#888888", linestyle="--", linewidth=1)
        axes[0].set(
            title="Score against Pentobi 31.0 level 2",
            ylabel="Score (%) · paired nominal 95% interval",
            ylim=(0, 100),
        )
        times = [results[str(b)]["sum_game_seconds"] / 200 for b in BUDGETS]
        bars = axes[1].bar(x, times, width=0.55, color="#b86a3d")
        axes[1].bar_label(bars, fmt="%.2f", padding=3)
        axes[1].set(
            title="Mean total game time",
            ylabel="Seconds · both agents and protocol overhead",
            ylim=(0, max(times) * 1.18),
        )
        for ax in axes:
            ax.set(
                xticks=x,
                xticklabels=[f"{b:,}" for b in BUDGETS],
                xlabel="Our simulations per move",
                xlim=(-0.4, 2.4),
            )
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
        fig.suptitle(
            "Same frozen small CNN · 100 identical fresh opening pairs per budget", fontsize=11
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
        fig.savefig(destination.with_suffix(".svg"), metadata={"Date": None})
        plt.close(fig)


def render(args):
    results, audit = verify(args.run)
    app = next(r for r in read(args.apps) if r["app_id"] == APP_ID)
    if app["state"] != "stopped" or int(app["tasks"]):
        raise ValueError("app must be stopped with zero tasks")
    cost = sum(
        (Decimal(r["cost"]) for r in read(args.billing) if r["object_id"] == APP_ID), Decimal(0)
    )
    lifetime = (
        datetime.fromisoformat(app["stopped_at"]) - datetime.fromisoformat(app["created_at"])
    ).total_seconds()
    hardware = read(args.run / "hardware.json")
    audit |= {
        "provider_metered_usd": str(cost),
        "app_lifetime_seconds": lifetime,
        "hardware": hardware,
    }
    difference = audit["paired_differences"]["2048-128"]
    low, high = difference["paired_95pct_interval"]
    supported = low > 0
    lines = [
        "## Completed results",
        "",
        f"The prespecified 2,048-minus-128 score difference was **{difference['score_difference'] * 100:+.2f} percentage points**, with a nominal paired95% interval of **{low * 100:+.2f} to {high * 100:+.2f} points**. "
        + (
            "The interval is above zero, supporting a benefit from extra playing search for this frozen agent."
            if supported
            else "The interval does not establish a positive benefit from extra playing search in this test."
        ),
        "",
        "![Score and runtime](assets/playing-search-sweep.png)",
        "",
        "| Simulations/move | W/D/L | Win rate | Score | Nominal paired95% interval | Mean margin | Mean game seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in BUDGETS:
        r = results[str(budget)]
        lo, hi = r["score_rate_95pct_cluster_bootstrap"]
        lines.append(
            f"| {budget:,} | {r['wins']}/{r['draws']}/{r['losses']} | {r['win_rate']:.2%} | {r['score_rate']:.2%} | {lo:.2%}–{hi:.2%} | {r['mean_margin']:.3f} | {r['sum_game_seconds'] / 200:.3f} |"
        )
    lines += [
        "",
        "Score counts draws as half. Each budget completed200 games on the same100 fresh openings. Per-game time includes both agents and protocol overhead; it is not GPU-only or our-agent-only latency. The prior31.5% level2 score used another book, so the new128 baseline is not a controlled change in model strength.",
        "",
        "| Paired score difference | Difference | Nominal95% interval |",
        "| --- | ---: | ---: |",
    ]
    for label, row in audit["paired_differences"].items():
        lo, hi = row["paired_95pct_interval"]
        lines.append(
            f"| {label} simulations | {row['score_difference'] * 100:+.2f}pp | {lo * 100:+.2f} to {hi * 100:+.2f}pp |"
        )
    lines += [
        "",
        "### Actual execution and verification",
        "",
        f"One {hardware['gpu']}, two capped cores,8GiB RAM; PyTorch{hardware['torch']}, CUDA{hardware['cuda']}, bfloat16. Function runtime{hardware['seconds']:.2f} seconds; app lifetime{lifetime:.0f} seconds. Provider metering at the saved snapshot: **${cost:.8f}**, versus the$4.269915 conservative reservation and$5 phase cap. Accounting can lag. All compute is stopped and the$50 reserve stays uncommitted.",
        "",
        "| Budget | Actual searched decisions | Actual simulations | Neural positions | Arm wall seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for budget, row in audit["budgets"].items():
        lines.append(
            f"| {budget} | {row['searched_decisions']:,} | {row['simulations']:,} | {row['neural_positions']:,} | {row['arm_seconds']:.2f} |"
        )
    lines += [
        "",
        "All600 saved games were independently replayed through Rust: legal moves, scores/margins, opening prefixes, seats/seeds and exact actor simulation counts matched. Frozen model, immutable evaluator, Pentobi binary and executed runner hashes passed. Every pair summary and paired interval was recomputed. Certified earlier checkpoints remain unchanged. No new training or model selection occurred. Search can change trajectories and game length; simulation totals therefore need not scale exactly with the nominal budget. Nominal paired intervals exclude training-seed, lineage and hardware uncertainty, and secondary tests are not multiplicity adjusted. This evaluates thinking time for one fixed model against one fixed opponent level, not training improvement or the Blokus frontier.",
        "",
        f"Complete artifacts: `{args.run}`. Derived audit and all downloaded hashes: `{args.run.parent / 'verification.json'}` and `download-sha256.json`. Reproduce locally with `python -m blokus_rl.duo_search_sweep_report --run <run> --apps <stopped-apps.json> --billing <provider-report.json> --out docs/playing-search-sweep.md`; no games are launched by the reporter.",
        "",
    ]
    heading, body = args.out.read_text().split("\n", 1)
    marker = "## Predeclared design"
    if marker in body:
        body = body.split(marker, 1)[1]
    args.out.write_text(
        heading + "\n\n" + "\n".join(lines) + "\n" + marker + "\n\n" + body.strip() + "\n"
    )
    plot(results, args.out.parent / "assets/playing-search-sweep.png")
    (args.run.parent / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.run.parent / "download-sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(args.run)): digest(p)
                for p in sorted(args.run.rglob("*"))
                if p.is_file()
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "apps", "billing", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    render(parser.parse_args())
