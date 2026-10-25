"""Batched neural PUCT over live Rust games, with no position-table dependency."""

import time
from dataclasses import dataclass

import numpy as np
import torch

from ._native import DuoSearch
from .duo_env import decode_decision, legacy_decision
from .duo_learning import pack


@dataclass
class SearchResult:
    observation: np.ndarray
    actions: np.ndarray
    candidates: np.ndarray
    policy: np.ndarray
    visits: np.ndarray
    priors: np.ndarray
    means: np.ndarray
    simulations: int
    nodes: int

    def action(self):
        # Visit counts decide, neural priors resolve equal counts deterministically.
        best = np.flatnonzero(self.visits == self.visits.max())
        index = int(best[np.argmax(self.priors[best])])
        return int(self.actions[index])


@torch.no_grad()
def search_games(
    model,
    games,
    simulations=128,
    leaf_batch=8,
    inference_batch=128,
    exploration=1.5,
    *,
    rngs=None,
    root_noise=0.25,
    noise_mass=10.0,
    metrics=None,
    native_features=True,
):
    if simulations < 1 or leaf_batch < 1 or inference_batch < 1:
        raise ValueError("positive simulation and batch sizes required")
    if not games:
        return []
    if rngs is not None and len(rngs) != len(games):
        raise ValueError("one root-noise RNG per game required")
    if not 0 <= root_noise <= 1 or noise_mass <= 0:
        raise ValueError("invalid root noise")
    device = next(model.parameters()).device
    trees = [DuoSearch(game, exploration, 1.0) for game in games]
    roots = {}
    start = time.monotonic()
    evaluated = batches = 0
    while any(tree.completed < simulations for tree in trees):
        pending = []
        for index, tree in enumerate(trees):
            remaining = simulations - tree.completed
            if remaining <= 0:
                continue
            budget = min(leaf_batch, remaining)
            entries = (
                [(token, decode_decision(data)) for token, data in tree.request_decisions(budget)]
                if native_features
                else [(token, legacy_decision(game)) for token, game in tree.request(budget)]
            )
            for token, (observation, actions, candidates) in entries:
                record = {"observation": observation, "candidates": candidates}
                pending.append((index, token, record, len(actions)))
                if token == 0:
                    roots[index] = (observation, actions, candidates)
        for offset in range(0, len(pending), inference_batch):
            subset = pending[offset : offset + inference_batch]
            batch = pack([entry[2] for entry in subset]).to(device)
            distribution, values = model.distribution(
                batch["observation"], batch["candidates"], batch["action_mask"]
            )
            # NumPy cannot represent bfloat16; the Rust boundary always uses f32.
            probabilities = distribution.probs.float().cpu().numpy()
            values = values.flatten().clamp(-1, 1).float().cpu().numpy()
            for row, (index, token, _, count) in enumerate(subset):
                priors = probabilities[row, :count].astype(np.float64)
                if token == 0 and rngs is not None and root_noise:
                    noise = rngs[index].dirichlet(np.full(count, noise_mass / count))
                    priors = (1 - root_noise) * priors + root_noise * noise
                trees[index].expand(token, priors.tolist(), float(values[row]))
            evaluated += len(subset)
            batches += 1
    results = []
    for index, tree in enumerate(trees):
        stats = tree.stats()
        observation, actions, candidates = roots[index]
        if not np.array_equal(actions, stats["actions"]):
            raise RuntimeError("search/network legal action ordering mismatch")
        visits = np.asarray(stats["visits"], dtype=np.int64)
        if visits.sum() != simulations or stats["simulations"] != simulations:
            raise RuntimeError("search simulation accounting mismatch")
        results.append(
            SearchResult(
                observation,
                actions,
                candidates,
                (visits / visits.sum()).astype(np.float32),
                visits,
                np.asarray(stats["priors"]),
                np.asarray(stats["means"]),
                simulations,
                stats["nodes"],
            )
        )
    if metrics is not None:
        metrics["neural_positions"] = metrics.get("neural_positions", 0) + evaluated
        metrics["inference_batches"] = metrics.get("inference_batches", 0) + batches
        metrics["simulations"] = metrics.get("simulations", 0) + simulations * len(games)
        metrics["search_seconds"] = metrics.get("search_seconds", 0.0) + time.monotonic() - start
    return results


class MctsPolicy:
    """A frozen policy + planning budget; usable by existing live-game evaluators."""

    def __init__(self, model, simulations=128, leaf_batch=8, exploration=1.5):
        self.model = model
        self.simulations, self.leaf_batch, self.exploration = simulations, leaf_batch, exploration
        self.metrics = {}

    def select_action(self, game, rng=None):
        return search_games(
            self.model,
            [game],
            self.simulations,
            self.leaf_batch,
            exploration=self.exploration,
            metrics=self.metrics,
        )[0].action()
