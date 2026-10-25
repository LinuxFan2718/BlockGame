"""Matched game-budget experiments, exact audits, and an honest comparison report."""

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from .audit import (
    NeuralPolicy,
    audit_policy,
    game_graph,
    selection_score,
    versus_frozen_policy,
)
from .env import FixedOpponentEnv
from .learning import (
    TrainConfig,
    collect_batch,
    configure_torch,
    load_checkpoint,
    make_ppo_loss,
    optimize_batch,
    save_checkpoint,
)
from .model import ActorCritic
from .search import SearchPolicy, collect_search_batch, optimize_search_batch
from .self_play import OpponentPool

ARMS = ("random", "selfplay", "alphazero")


def audit_metrics(result):
    seats = list(result["by_seat"].values())
    return {
        "certified": result["never_loses_both_seats"],
        "perfect_loss": sum(seat["vs_perfect"]["losses"] for seat in seats) / 2,
        "random_loss": sum(seat["vs_random"]["losses"] for seat in seats) / 2,
        "random_win": sum(seat["vs_random"]["wins"] for seat in seats) / 2,
        "action_errors": sum(seat["avoidable_action_errors"] for seat in seats),
    }


def train_arm(arm, config, out, *, simulations=32, audit_every=25, references=None):
    if arm not in ARMS:
        raise ValueError(f"unknown experiment arm {arm}")
    if audit_every < 1:
        raise ValueError("audit_every must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    configure_torch(config.seed)
    graph = game_graph()
    model = ActorCritic(config.hidden_size).eval()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0001 if arm == "alphazero" else 0,
    )
    pool = OpponentPool(model, config.seed + 100_000) if arm == "selfplay" else None
    env = FixedOpponentEnv(opponent=pool)
    env.reset(seed=config.seed)
    loss_module = make_ppo_loss(model, config) if arm != "alphazero" else None
    search_rng = np.random.default_rng(config.seed + 200_000)
    start = time.monotonic()
    initial_neural = NeuralPolicy(model, graph)
    initial_audit = audit_policy(initial_neural, graph)
    initial_search = (
        audit_policy(SearchPolicy(initial_neural, simulations), graph)
        if arm == "alphazero"
        else None
    )
    del initial_neural
    best_score = selection_score(initial_audit)
    best_update = 0

    def save(name, update):
        save_checkpoint(
            out / name,
            model,
            config,
            metadata={
                "algorithm": arm,
                "update": update,
                "search_simulations": simulations if arm == "alphazero" else 0,
            },
        )

    save("best.pt", 0)
    history = [{"update": 0, "episodes": 0, **audit_metrics(initial_audit)}]
    positions, optimizer_steps, total_simulations = 0, 0, 0
    final_audit = initial_audit
    print(
        f"{arm} seed {config.seed}: started; {config.updates * config.episodes_per_update:,} games",
        flush=True,
    )
    with (out / "metrics.csv").open("w", newline="") as file:
        writer = None
        for update in range(1, config.updates + 1):
            if arm == "alphazero":
                batch, collected = collect_search_batch(
                    model, config, graph, search_rng, simulations
                )
                steps = len(batch["observation"])
                learned = optimize_search_batch(model, optimizer, batch, config)
                optimizer_steps += learned["optimizer_steps"]
                total_simulations += collected["search_simulations"]
            else:
                batch, training_return = collect_batch(
                    model, env, config, before_episode=pool.begin_episode if pool else None
                )
                steps = len(batch)
                learned = optimize_batch(model, loss_module, optimizer, batch, config)
                learned["training_return"] = training_return
                optimizer_steps += config.epochs * math.ceil(steps / config.minibatch_size)
                if pool and update % 10 == 0:
                    pool.add_snapshot(update)
            positions += steps
            row = {
                "update": update,
                "episodes": update * config.episodes_per_update,
                "positions": positions,
                "optimizer_steps": optimizer_steps,
                "search_simulations": total_simulations,
                **learned,
                "certified": None,
                "perfect_loss": None,
                "random_loss": None,
                "random_win": None,
                "action_errors": None,
            }
            if update % audit_every == 0 or update == config.updates:
                final_audit = audit_policy(NeuralPolicy(model, graph), graph)
                row.update(audit_metrics(final_audit))
                score = selection_score(final_audit)
                if score < best_score:
                    best_score, best_update = score, update
                    save("best.pt", update)
                history.append({key: row[key] for key in history[0]})
                print(
                    f"{arm} seed {config.seed} {update}/{config.updates}: "
                    f"perfect-loss {row['perfect_loss']:.2%}, errors {row['action_errors']}, "
                    f"never-lose {row['certified']}",
                    flush=True,
                )
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            file.flush()
    env.close()
    save("final.pt", config.updates)
    best_neural = NeuralPolicy(load_checkpoint(out / "best.pt"), graph)
    best_audit = audit_policy(best_neural, graph)
    best_search_policy = SearchPolicy(best_neural, simulations) if arm == "alphazero" else None
    best_search_audit = audit_policy(best_search_policy, graph) if best_search_policy else None
    frozen_results = {
        name: versus_frozen_policy(best_neural, opponent, graph)
        for name, opponent in (references or {}).items()
    }
    search_frozen_results = (
        {
            name: versus_frozen_policy(best_search_policy, opponent, graph)
            for name, opponent in (references or {}).items()
        }
        if best_search_policy
        else None
    )
    summary = {
        "arm": arm,
        "config": asdict(config),
        "audit_every": audit_every,
        "simulations_per_move": simulations if arm == "alphazero" else 0,
        "training_games": config.updates * config.episodes_per_update,
        "positions": positions,
        "optimizer_steps": optimizer_steps,
        "training_search_simulations": total_simulations,
        "total_seconds_including_audits": time.monotonic() - start,
        "best_update": best_update,
        "initial": initial_audit,
        "initial_with_search": initial_search,
        "final": final_audit,
        "best": best_audit,
        "best_with_search": best_search_audit,
        "frozen_opponents": frozen_results,
        "search_frozen_opponents": search_frozen_results,
        "opponent_episode_counts": pool.episode_counts if pool else None,
        "history": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def study(
    out,
    seeds=(42, 43, 44),
    updates=300,
    episodes_per_update=128,
    simulations=32,
    arms=ARMS,
    audit_every=25,
    workers=1,
):
    if not seeds or not arms or any(arm not in ARMS for arm in arms):
        raise ValueError("supply at least one seed and valid experiment arm")
    if len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms):
        raise ValueError("seeds and arms must be unique")
    base = TrainConfig(updates=updates, episodes_per_update=episodes_per_update)
    if workers < 1:
        raise ValueError("workers must be positive")
    configure_torch(seeds[0])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    graph = game_graph()
    references, reference_files = {}, {}
    for name in ("first-ppo", "ppo-demo"):
        checkpoint = Path("runs") / name / "model.pt"
        if checkpoint.exists():
            references[name] = NeuralPolicy(load_checkpoint(checkpoint), graph)
            reference_files[name] = {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
    manifest = {
        "seeds": list(seeds),
        "arms": list(arms),
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "simulations": simulations,
        "audit_every": audit_every,
        "reference_opponents": reference_files,
        "protocol": "docs/self-play-protocol.md",
        "workers": workers,
    }
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = []
    if workers == 1:
        for seed in seeds:
            for arm in arms:
                summaries.append(
                    train_arm(
                        arm,
                        replace(base, seed=seed),
                        out / f"{arm}-seed-{seed}",
                        simulations=simulations,
                        audit_every=audit_every,
                        references=references,
                    )
                )
    else:
        # Isolate Torch and NumPy RNGs in separate processes. Each seed runs its
        # matched arms sequentially, with one Torch CPU thread in each worker.
        with ProcessPoolExecutor(
            max_workers=min(workers, len(seeds)), mp_context=get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(
                    _run_seed,
                    str(out),
                    seed,
                    base,
                    tuple(arms),
                    simulations,
                    audit_every,
                    reference_files,
                )
                for seed in seeds
            ]
            for future in futures:
                summaries.extend(future.result())
    (out / "comparison.json").write_text(
        json.dumps({"protocol": manifest, "runs": summaries}, indent=2) + "\n"
    )
    write_report(out, summaries)
    plot_comparison(out, summaries)
    print(f"Completed experiment. Report: {out / 'report.md'}", flush=True)
    return summaries


def _run_seed(out, seed, base, arms, simulations, audit_every, reference_files):
    configure_torch(seed)
    graph = game_graph()
    references = {
        name: NeuralPolicy(load_checkpoint(item["path"]), graph)
        for name, item in reference_files.items()
    }
    return [
        train_arm(
            arm,
            replace(base, seed=seed),
            Path(out) / f"{arm}-seed-{seed}",
            simulations=simulations,
            audit_every=audit_every,
            references=references,
        )
        for arm in arms
    ]


def write_report(out, summaries):
    lines = [
        "# Self-play experiment results",
        "",
        "Protocol: `docs/self-play-protocol.md`. Checkpoints are selected using the declared",
        "raw-policy criterion. Probabilities below are computed over all opponent responses,",
        "not estimated from sampled games. Each row averages the two seats.",
        "",
        "| Arm | Seed | Best update | Never loses (X / O) | Loss vs perfect | Loss vs random | Avoidable errors |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for run in summaries:
        for label, key in ((run["arm"], "best"), (run["arm"] + "+search", "best_with_search")):
            result = run[key]
            if result is None:
                continue
            metrics = audit_metrics(result)
            certificate = " / ".join(
                "yes" if result["by_seat"][seat]["never_loses"] else "no" for seat in ("X", "O")
            )
            lines.append(
                f"| {label} | {run['config']['seed']} | {run['best_update']} | {certificate} | "
                f"{metrics['perfect_loss']:.2%} | {metrics['random_loss']:.2%} | {metrics['action_errors']} |"
            )
    lines += ["", "## Interpretation", ""]
    for arm in ARMS:
        runs = [run for run in summaries if run["arm"] == arm]
        if not runs:
            continue
        successes = sum(run["best"]["never_loses_both_seats"] for run in runs)
        lines.append(
            f"- {arm}: raw-network never-lose certificate passed in {successes}/{len(runs)} seeds."
        )
        if arm == "alphazero":
            before = sum(run["initial_with_search"]["never_loses_both_seats"] for run in runs)
            after = sum(run["best_with_search"]["never_loses_both_seats"] for run in runs)
            lines.append(
                f"- Search-assisted AlphaZero-style policy: {before}/{len(runs)} certified before training, "
                f"{after}/{len(runs)} after training."
            )
    lines += [
        "",
        "The primary target is zero losing continuations in BOTH seats. If that fails,",
        "improvements in perfect-opponent loss or tactical errors are partial progress.",
        "A higher random-opponent win rate alone is not a solution certificate.",
        "",
        "## Resource budget",
        "",
        "| Arm | Seed | Games | Training positions | Optimizer steps | Search simulations | Seconds incl. audits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summaries:
        lines.append(
            f"| {run['arm']} | {run['config']['seed']} | {run['training_games']} | "
            f"{run['positions']} | {run['optimizer_steps']} | {run['training_search_simulations']} | "
            f"{run['total_seconds_including_audits']:.1f} |"
        )
    lines += [
        "",
        "The PPO arms have matched game and optimizer settings. The search arm spends more",
        "compute and learns from both players' positions. Its results do not establish equal-compute",
        "superiority. Three seeds are a small reproducibility check.",
        "",
        "See `comparison.json` for per-seat results, final versus selected checkpoints, frozen",
        "reference opponents, untrained search, and explicit losing sequences.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def plot_comparison(out, summaries):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blokus-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    colors = {"random": "#808080", "selfplay": "#0072B2", "alphazero": "#D55E00"}
    for run in summaries:
        history = run["history"]
        for axis, metric in zip(axes, ("perfect_loss", "action_errors"), strict=True):
            axis.plot(
                [row["episodes"] for row in history],
                [row[metric] for row in history],
                color=colors[run["arm"]],
                alpha=0.65,
                label=run["arm"]
                if run["config"]["seed"] == summaries[0]["config"]["seed"]
                else None,
            )
    axes[0].set(
        title="Raw network: loss against perfect play", ylabel="Exact loss probability", ylim=(0, 1)
    )
    axes[1].set(
        title="Raw network: avoidable action errors", ylabel="Positions with a suboptimal move"
    )
    for axis in axes:
        axis.set_xlabel("Training games")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(out / "comparison.png", dpi=160)
    fig.savefig(out / "comparison.svg")
    plt.close(fig)
