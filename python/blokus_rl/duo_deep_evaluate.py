"""Batched paired-opening evaluation for expensive spatial policies."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ._native import DuoGame
from .duo_crossplay import paired_interval
from .duo_env import decision
from .duo_learning import choose
from .duo_mcts import evaluate_records, search_games
from .duo_shared import evaluate_indexed, indexed


class IndexedSearchPolicy:
    def __init__(self, model, simulations=128, precision="float32"):
        self.model, self.simulations = model, simulations
        self.evaluator, self.metrics = evaluator_for(model, precision), {}

    def select_action(self, game, rng=None):
        if self.simulations:
            return search_games(
                self.model,
                [game],
                self.simulations,
                8,
                evaluator=self.evaluator,
                metrics=self.metrics,
            )[0].action()
        observation, actions, candidates = decision(game)
        probabilities, _ = self.evaluator([{"observation": observation, "candidates": candidates}])
        return int(actions[int(np.argmax(probabilities[0, : len(actions)]))])


def evaluate_pentobi(
    model,
    binary,
    openings,
    out,
    *,
    simulations=128,
    seed=2569000,
    precision="float32",
    commit=None,
):
    """Resume at whole opening-pair boundaries; every final score is cross-checked."""
    from .duo_deep import atomic_json
    from .duo_pentobi import Pentobi, match

    out = Path(out)
    partial = out.with_suffix(".partial.json")
    fingerprint = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        fingerprint.update(key.encode())
        fingerprint.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    settings = {
        "model_sha256": fingerprint.hexdigest(),
        "seed": seed,
        "simulations": simulations,
        "precision": precision,
        "openings_sha256": hashlib.sha256(json.dumps(openings).encode()).hexdigest(),
    }
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved["settings"] != settings:
            raise ValueError("partial Pentobi evaluation configuration/model changed")
        details = saved["games"]
    else:
        details = []
    if len(details) % 2:
        raise ValueError("partial evaluation must end at a whole opening pair")
    policy = IndexedSearchPolicy(model, simulations, precision)
    start = time.monotonic()
    with Pentobi(binary=binary, level=1, seed=seed) as pentobi:
        for index in range(len(details) // 2, len(openings)):
            for seat in (0, 1):
                details.append(
                    match(pentobi, policy, seat, seed + index + seat * 1000000, openings[index])
                )
            atomic_json(partial, {"settings": settings, "games": details})
            if commit and (index + 1) % 10 == 0:
                commit()
                print(f"Pentobi evaluation: {index + 1}/{len(openings)} pairs", flush=True)
    pairs = np.asarray(
        [1 if r["margin"] > 0 else 0.5 if r["margin"] == 0 else 0 for r in details]
    ).reshape(-1, 2)
    result = {
        "pentobi": "31.0 level1, one thread, no book/resignation; every margin checked",
        "games": len(details),
        "opening_pairs": len(openings),
        "wins": int(np.sum(pairs == 1)),
        "draws": int(np.sum(pairs == 0.5)),
        "losses": int(np.sum(pairs == 0)),
        "score_rate": float(pairs.mean()),
        "score_rate_95pct_cluster_bootstrap": paired_interval(pairs),
        "opening_pair_scores": pairs.tolist(),
        "games_detail": details,
        "mean_margin": float(np.mean([r["margin"] for r in details])),
        "simulations": simulations,
        "metrics_since_resume": policy.metrics,
        "seconds_since_resume": time.monotonic() - start,
        "settings": settings,
    }
    atomic_json(out, result)
    if commit:
        commit()
    return result


def evaluator_for(model, precision="float32"):
    def evaluate(rows):
        if getattr(model, "architecture", "additive") in ("residual", "conv"):
            return evaluate_indexed(model, indexed(rows), precision)
        return evaluate_records(model, rows)

    return evaluate


@torch.no_grad()
def evaluate_many(
    policy,
    opponent,
    openings,
    *,
    simulations=128,
    opponent_simulations=128,
    live_games=32,
    seed=2563000,
    precision="float32",
):
    started = time.monotonic()
    outcomes = np.zeros((len(openings), 2))
    margins, metrics = [], {}
    evaluators = [evaluator_for(policy, precision)]
    evaluators.append(None if isinstance(opponent, str) else evaluator_for(opponent, precision))
    jobs = [(i, seat) for i in range(len(openings)) for seat in (0, 1)]
    for offset in range(0, len(jobs), live_games):
        subset = jobs[offset : offset + live_games]
        games, rngs = [], []
        for index, seat in subset:
            game = DuoGame()
            for action in openings[index]:
                game.step(action)
            if game.terminated:
                raise ValueError("terminal evaluation opening")
            games.append(game)
            rngs.append(np.random.default_rng(seed + index + seat * 1000000))
        active = list(range(len(games)))
        while active:
            for agent, model in enumerate((policy, opponent)):
                ids = [i for i in active if int(games[i].current_player != subset[i][1]) == agent]
                if not ids:
                    continue
                budget = opponent_simulations if agent else simulations
                if isinstance(model, str):
                    actions = [choose(games[i], model, rngs[i]) for i in ids]
                elif budget:
                    results = search_games(
                        model,
                        [games[i] for i in ids],
                        budget,
                        8,
                        128,
                        evaluator=evaluators[agent],
                        metrics=metrics,
                    )
                    actions = [r.action() for r in results]
                else:
                    observations = [decision(games[i]) for i in ids]
                    probabilities, _ = evaluators[agent](
                        [{"observation": obs, "candidates": cand} for obs, _, cand in observations]
                    )
                    actions = [
                        int(obs[1][np.argmax(probabilities[row, : len(obs[1])])])
                        for row, obs in enumerate(observations)
                    ]
                for i, action in zip(ids, actions, strict=True):
                    games[i].step(action)
                active = [i for i in active if not games[i].terminated]
        for game, (index, seat) in zip(games, subset, strict=True):
            margin = game.scores[seat] - game.scores[1 - seat]
            margins.append(margin)
            outcomes[index, seat] = 1 if margin > 0 else (0.5 if margin == 0 else 0)
    return {
        "games": len(jobs),
        "opening_pairs": len(openings),
        "wins": int(np.sum(outcomes == 1)),
        "draws": int(np.sum(outcomes == 0.5)),
        "losses": int(np.sum(outcomes == 0)),
        "score_rate": float(outcomes.mean()),
        "score_rate_95pct_cluster_bootstrap": paired_interval(outcomes),
        "opening_pair_scores": outcomes.tolist(),
        "mean_margin": float(np.mean(margins)),
        "learner_simulations": simulations,
        "opponent_simulations": 0 if isinstance(opponent, str) else opponent_simulations,
        "metrics": metrics,
        "seconds": time.monotonic() - started,
    }
