"""A teaching-scale AlphaZero-style search and self-play training objective.

Search uses Rust-generated transitions, network priors, predicted leaf values,
and actual terminal rewards. It NEVER uses the evaluator's minimax answers.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .audit import EMPTY, NeuralPolicy


@dataclass
class SearchNode:
    priors: np.ndarray
    counts: list[int]
    total_values: list[float]


class SearchPolicy:
    def __init__(self, neural: NeuralPolicy, simulations=32, exploration=1.5):
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.neural = neural
        self.graph = neural.graph
        self.simulations = simulations
        self.exploration = exploration
        self.total_simulations = 0
        self._actions = {}

    def distribution(self, root, *, rng=None):
        graph = self.graph
        if not graph.positions[root].children:
            raise ValueError("cannot search a terminal position")
        tree = {}

        def expand(board):
            priors, value = self.neural.predictions[board]
            tree[board] = SearchNode(priors.copy(), [0] * 9, [0.0] * 9)
            return float(np.clip(value, -1.0, 1.0))

        expand(root)
        if rng is not None:
            # Training-only root noise. Evaluation is deterministic and noise-free.
            actions = list(graph.positions[root].children)
            noise = rng.dirichlet(np.full(len(actions), 0.3))
            tree[root].priors[actions] = 0.75 * tree[root].priors[actions] + 0.25 * noise

        def simulate(board):
            if board not in tree:
                return expand(board)
            state = graph.positions[board]
            node = tree[board]
            scale = math.sqrt(sum(node.counts) + 1)

            def priority(action):
                count = node.counts[action]
                mean = node.total_values[action] / count if count else 0.0
                return mean + self.exploration * node.priors[action] * scale / (1 + count)

            action = max(state.children, key=priority)
            child = state.children[action]
            if not graph.positions[child].children:
                value = graph.reward(child, state.player)
            else:
                value = -simulate(child)  # Opponent's value has the opposite sign.
            node.counts[action] += 1
            node.total_values[action] += value
            return value

        for _ in range(self.simulations):
            simulate(root)
        self.total_simulations += self.simulations
        counts = np.asarray(tree[root].counts, dtype=np.float64)
        return counts / counts.sum()

    def action(self, board):
        if board not in self._actions:
            self._actions[board] = int(self.distribution(board).argmax())
        return self._actions[board]


def collect_search_batch(model, config, graph, rng, simulations):
    neural = NeuralPolicy(model, graph)
    search = SearchPolicy(neural, simulations)
    observations, masks, targets, outcomes = [], [], [], []
    game_lengths = []
    for _ in range(config.episodes_per_update):
        board = EMPTY
        players = []
        while graph.positions[board].children:
            state = graph.positions[board]
            distribution = search.distribution(board, rng=rng)
            sign = 1 if state.player == "X" else -1
            observations.append([cell * sign for cell in board])
            masks.append([action in state.children for action in range(9)])
            targets.append(distribution)
            players.append(state.player)
            action = int(rng.choice(9, p=distribution))
            board = state.children[action]
        outcomes.extend(graph.reward(board, player) for player in players)
        game_lengths.append(len(players))
    return {
        "observation": torch.tensor(observations, dtype=torch.float32),
        "action_mask": torch.tensor(masks, dtype=torch.bool),
        "policy_target": torch.tensor(np.asarray(targets), dtype=torch.float32),
        "value_target": torch.tensor(outcomes, dtype=torch.float32).unsqueeze(-1),
    }, {
        "search_simulations": search.total_simulations,
        "mean_game_length": float(np.mean(game_lengths)),
    }


def optimize_search_batch(model, optimizer, batch, config):
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "optimizer_steps": 0}
    for _ in range(config.epochs):
        for indices in torch.randperm(len(batch["observation"])).split(config.minibatch_size):
            logits, values = model(batch["observation"][indices])
            mask = batch["action_mask"][indices]
            # A finite masking constant avoids the undefined product 0 * -inf.
            log_probabilities = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
            policy_loss = -(batch["policy_target"][indices] * log_probabilities).sum(-1).mean()
            value_loss = F.mse_loss(values, batch["value_target"][indices])
            loss = policy_loss + value_loss
            if not torch.isfinite(loss):
                raise RuntimeError("search training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["optimizer_steps"] += 1
    for key in ("policy_loss", "value_loss"):
        totals[key] /= totals["optimizer_steps"]
    return totals
