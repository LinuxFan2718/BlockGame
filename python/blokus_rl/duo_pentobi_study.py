"""Seat-swapped held-out games against a pinned outside Blokus program."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .duo_crossplay import opening_book, paired_interval
from .duo_learning import load_duo
from .duo_mcts import MctsPolicy
from .duo_pentobi import Pentobi, match
from .learning import configure_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--openings", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2551000)
    args = parser.parse_args()
    configure_torch(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    book = opening_book(args.openings, seed=args.seed)
    model = load_duo(args.checkpoint)
    policy = MctsPolicy(model, args.simulations) if args.simulations else model
    protocol = vars(args) | {
        "checkpoint": str(args.checkpoint.resolve()),
        "out": str(args.out),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "openings": book,
        "pentobi_version": "31.0",
        "pentobi_commit": "d5374ddc20eced902788e733d67ee16bd1836fd3",
        "threads": 1,
        "book": False,
        "resign": False,
        "avoid_symmetric_draw": True,
        "setup": "rotate 180 degrees when our first player chooses J5; Pentobi fixes E10",
        "passes": "forced passes omitted in GTP; explicit colors; scores cross-checked every game",
        "interval_unit": "whole two-game opening pair",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    pairs, results = [], []
    start = time.monotonic()
    with Pentobi(level=args.level, seed=args.seed) as pentobi:
        for index, opening in enumerate(book):
            values = []
            for seat in (0, 1):
                result = match(pentobi, policy, seat, args.seed + index + seat * 1000000, opening)
                results.append(result)
                values.append(
                    1.0 if result["margin"] > 0 else 0.5 if result["margin"] == 0 else 0.0
                )
            pairs.append(values)
            summary = {
                "games": len(results),
                "wins": sum(r["margin"] > 0 for r in results),
                "draws": sum(r["margin"] == 0 for r in results),
                "losses": sum(r["margin"] < 0 for r in results),
                "score_rate": float(np.mean(pairs)),
                "mean_margin": float(np.mean([r["margin"] for r in results])),
                "score_rate_95pct_cluster_bootstrap": paired_interval(pairs),
                "opening_pair_scores": pairs,
                "seconds": time.monotonic() - start,
                "games_detail": results,
            }
            (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in summary.items()
                        if k not in ("games_detail", "opening_pair_scores")
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
