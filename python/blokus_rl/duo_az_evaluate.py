"""Freeze completed self-play runs, then evaluate on reserved paired openings."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from .duo_crossplay import evaluate_pair, opening_book, paired_interval
from .duo_learning import load_duo
from .duo_mcts import MctsPolicy
from .duo_pentobi import Pentobi, match
from .learning import configure_torch


def evaluate(run, out, reference, openings=100, seed=2549000, simulations=128):
    run, out = Path(run), Path(out)
    if not (run / "completed.json").exists():
        raise ValueError("finish the bounded training run before held-out evaluation")
    out.mkdir(parents=True, exist_ok=False)
    configure_torch(seed)
    sources = {
        "selected": run / "best.pt",
        "reference": Path(reference),
        "initial": run / "initial.pt",
    }
    paths = {}
    for name, path in sources.items():
        paths[name] = out / f"{name}.pt"
        shutil.copy2(path, paths[name])
    book = opening_book(openings, seed=seed)
    protocol = {
        "run": str(run.resolve()),
        "seed": seed,
        "openings": book,
        "simulations": simulations,
        "purpose": "held-out evaluation, never checkpoint selection",
        "checkpoint_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
        "pentobi": "31.0, level1, threads1, no book/resign; same opening book, seat swaps",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    models = {name: load_duo(path) for name, path in paths.items()}
    results = {}
    searched = MctsPolicy(models["selected"], simulations)
    cases = [
        ("raw_vs_initial", models["selected"], models["initial"]),
        ("raw_vs_reference", models["selected"], models["reference"]),
        ("search_vs_reference", searched, models["reference"]),
        ("raw_vs_largest", models["selected"], "largest"),
        ("raw_vs_frontier", models["selected"], "frontier"),
    ]
    for label, policy, opponent in cases:
        start = time.monotonic()
        result = evaluate_pair(policy, opponent, book, seed)
        result["seconds"] = time.monotonic() - start
        results[label] = result
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            label,
            json.dumps(
                {k: result[k] for k in ("wins", "draws", "losses", "score_rate", "seconds")}
            ),
            flush=True,
        )
    # A deterministic normal opening gives exactly two games, with no fake CI.
    results["empty_board_vs_reference"] = evaluate_pair(
        models["selected"], models["reference"], [[]], seed, bootstrap=False
    )
    for label, policy in [("raw_vs_pentobi", models["selected"]), ("search_vs_pentobi", searched)]:
        pairs, details = [], []
        start = time.monotonic()
        with Pentobi(seed=seed) as pentobi:
            for index, opening in enumerate(book):
                pair = []
                for seat in (0, 1):
                    result = match(pentobi, policy, seat, seed + index + seat * 1000000, opening)
                    details.append(result)
                    pair.append(
                        1.0 if result["margin"] > 0 else 0.5 if result["margin"] == 0 else 0.0
                    )
                pairs.append(pair)
        results[label] = {
            "games": len(details),
            "wins": sum(r["margin"] > 0 for r in details),
            "draws": sum(r["margin"] == 0 for r in details),
            "losses": sum(r["margin"] < 0 for r in details),
            "score_rate": sum(sum(p) for p in pairs) / (2 * len(pairs)),
            "opening_pair_scores": pairs,
            "score_rate_95pct_cluster_bootstrap": paired_interval(pairs),
            "games_detail": details,
            "seconds": time.monotonic() - start,
        }
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            label,
            json.dumps(
                {
                    k: v
                    for k, v in results[label].items()
                    if k not in ("games_detail", "opening_pair_scores")
                }
            ),
            flush=True,
        )
    (out / "completed.json").write_text(
        json.dumps({"run": str(run), "cases": list(results)}) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, default=Path("runs/duo-frontier-training-342/best.pt")
    )
    parser.add_argument("--openings", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2549000)
    parser.add_argument("--simulations", type=int, default=128)
    args = parser.parse_args()
    evaluate(args.run, args.out, args.reference, args.openings, args.seed, args.simulations)


if __name__ == "__main__":
    main()
