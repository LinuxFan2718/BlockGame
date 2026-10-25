"""Matched search-budget probe. Frozen weights; common, seat-swapped openings."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .duo_crossplay import evaluate_pair, opening_book
from .duo_learning import load_duo
from .duo_mcts import MctsPolicy
from .learning import configure_torch


def trial(task):
    checkpoint, out, budget, book, seed = task
    configure_torch(seed)
    model = load_duo(checkpoint)
    policy = MctsPolicy(model, simulations=budget)
    start = time.monotonic()
    result = evaluate_pair(policy, model, book, seed)
    result.update(seconds=time.monotonic() - start, budget=budget, metrics=policy.metrics)
    Path(out, f"budget-{budget}.json").write_text(json.dumps(result, indent=2) + "\n")
    return {k: result[k] for k in ("budget", "wins", "draws", "losses", "score_rate", "seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--openings", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2541000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    book = opening_book(args.openings, seed=args.seed)
    protocol = vars(args) | {
        "checkpoint": str(args.checkpoint.resolve()),
        "out": str(args.out.resolve()),
        "openings": book,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "purpose": "planning vs same frozen raw policy; evaluation only, no training noise",
        "interval_unit": "two seat-swapped games per opening",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tasks = [(args.checkpoint, args.out, b, book, args.seed) for b in args.budgets]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(trial, tasks):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
