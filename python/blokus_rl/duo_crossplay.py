"""Cross-play with actual greedy policies and paired randomized openings.

Statistical units are openings. Each is replayed with the two policies in both
seats; the bootstrap resamples whole pairs, preserving this dependence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ._native import DuoGame
from .duo_learning import choose, load_duo


def opening_book(count=100, plies=4, seed=1790000):
    if count < 1 or plies < 0:
        raise ValueError("positive opening count and nonnegative plies required")
    result = []
    for index in range(count):
        rng = np.random.default_rng(seed + index)
        game = DuoGame()
        moves = []
        for _ in range(plies):
            if game.terminated:
                raise ValueError("opening ends the game; use fewer plies")
            action = int(rng.choice(game.legal_actions()))
            game.step(action)
            moves.append(action)
        if game.terminated:
            raise ValueError("terminal opening")
        result.append(moves)
    return result


def paired_interval(pair_values, seed=17, resamples=10000):
    values = np.asarray(pair_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError("expected one row per opening, two seat-swapped games per row")
    units = values.mean(axis=1)
    if len(units) == 1:
        return None  # One exact opening is descriptive, not a population estimate.
    rng = np.random.default_rng(seed)
    sampled = units[rng.integers(len(units), size=(resamples, len(units)))].mean(axis=1)
    return np.quantile(sampled, [0.025, 0.975]).tolist()


@torch.no_grad()
def evaluate_pair(policy, opponent, openings, seed=1790000, *, bootstrap=True):
    if not openings:
        raise ValueError("at least one opening required")
    wins, points = [], []
    seats = [{"wins": 0, "draws": 0, "losses": 0, "margins": []} for _ in range(2)]
    example = None
    for index, opening in enumerate(openings):
        pair_wins, pair_points = [], []
        initial = DuoGame()
        for action in opening:
            initial.step(action)
        if initial.terminated:
            raise ValueError("cannot evaluate a terminal opening")
        for seat in (0, 1):
            game = initial.copy()
            moves = list(opening)
            rng = np.random.default_rng(seed + index + seat * 1_000_000)
            while not game.terminated:
                action = choose(game, policy if game.current_player == seat else opponent, rng)
                game.step(action)
                moves.append(action)
            score = game.scores
            margin = score[seat] - score[1 - seat]
            pair_wins.append(float(margin > 0))
            pair_points.append(1.0 if margin > 0 else (0.5 if margin == 0 else 0.0))
            seats[seat]["wins" if margin > 0 else ("draws" if margin == 0 else "losses")] += 1
            seats[seat]["margins"].append(margin)
            if margin < 0 and example is None:
                example = {
                    "learner_seat": seat,
                    "opening_index": index,
                    "opening": list(opening),
                    "actions": moves,
                    "scores": score,
                }
        wins.append(pair_wins)
        points.append(pair_points)
    for seat in seats:
        seat["mean_margin"] = float(np.mean(seat.pop("margins")))
    return {
        "mode": "search_crossplay"
        if any(hasattr(p, "select_action") for p in (policy, opponent))
        else "greedy_vs_greedy",
        "learner_simulations": getattr(policy, "simulations", 0),
        "opponent_simulations": getattr(opponent, "simulations", 0),
        "opening_pairs": len(openings),
        "games": 2 * len(openings),
        "win_rate": float(np.mean(wins)),
        "score_rate": float(np.mean(points)),
        "wins": sum(s["wins"] for s in seats),
        "draws": sum(s["draws"] for s in seats),
        "losses": sum(s["losses"] for s in seats),
        "by_seat": seats,
        "win_rate_95pct_cluster_bootstrap": paired_interval(wins) if bootstrap else None,
        "score_rate_95pct_cluster_bootstrap": paired_interval(points) if bootstrap else None,
        "loss_example": example,
        "opening_pair_scores": points,
        "opening_pair_wins": wins,
    }


def crossplay(checkpoints, out, openings=100, seed=1790000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    book = opening_book(openings, seed=seed)
    models = {name: load_duo(path) for name, path in checkpoints.items()}
    protocol = {
        "seed": seed,
        "opening_plies": 4,
        "openings": book,
        "opening_sha256": hashlib.sha256(json.dumps(book).encode()).hexdigest(),
        "checkpoint_sha256": {
            name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for name, p in checkpoints.items()
        },
        "checkpoints": {name: str(Path(p).resolve()) for name, p in checkpoints.items()},
        "interval_unit": "whole two-game opening pair",
        "bootstrap_resamples": 10000,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = {}
    names = list(models)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            key = f"{left}_vs_{right}"
            results[key] = {
                "learner": left,
                "opponent": right,
                "empty_board": evaluate_pair(models[left], models[right], [[]], bootstrap=False),
            }
            if left.startswith("frontier") and right.startswith("frontier"):
                results[key]["random_openings"] = evaluate_pair(
                    models[left], models[right], book, seed
                )
            (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                key,
                json.dumps(
                    {k: v["score_rate"] for k, v in results[key].items() if isinstance(v, dict)}
                ),
                flush=True,
            )
    return results


def main():
    from .learning import configure_torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--openings", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1790000)
    args = parser.parse_args()
    models = {p.parent.name.removeprefix("duo-"): p for p in args.checkpoints}
    if len(models) != len(args.checkpoints):
        parser.error("checkpoints must come from distinct named run directories")
    configure_torch(args.seed)
    crossplay(models, args.out, args.openings, args.seed)


if __name__ == "__main__":
    main()
