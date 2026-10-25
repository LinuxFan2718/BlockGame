"""Exact evaluation on a game graph generated ONLY by the Rust game engine.

Minimax is an evaluation oracle. Its answers are never training targets or moves
for training opponents. The never-lose certificate applies to a fixed, greedy
policy starting from an empty board, with either seat and any legal opponent.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from ._native import Game
from .model import ActorCritic

Board = tuple[int, ...]
EMPTY: Board = (0,) * 9


@dataclass(frozen=True)
class Position:
    player: str | None
    outcome: str
    children: dict[int, Board]


class GameGraph:
    def __init__(self):
        self.positions: dict[Board, Position] = {}

        def visit(game, history):
            board = tuple(game.observe("X")[0])
            if board in self.positions:
                return board
            children = {}
            self.positions[board] = Position(game.current_player, game.outcome, children)
            for action in game.legal_actions():
                # Replay at most nine moves to make an independent Rust state.
                # This avoids a second implementation of legality or win detection.
                child = Game()
                for move in (*history, action):
                    child.step(move)
                children[action] = visit(child, (*history, action))
            return board

        visit(Game(), ())

    def reward(self, board: Board, perspective: str) -> int:
        outcome = self.positions[board].outcome
        if outcome in ("draw", "in_progress"):
            return 0
        return 1 if outcome == perspective else -1

    @cache
    def minimax(self, board: Board, perspective: str) -> int:
        node = self.positions[board]
        if not node.children:
            return self.reward(board, perspective)
        values = [self.minimax(child, perspective) for child in node.children.values()]
        return max(values) if node.player == perspective else min(values)

    def optimal_actions(self, board: Board) -> tuple[int, ...]:
        node = self.positions[board]
        if not node.children:
            return ()
        best = self.minimax(board, node.player)
        return tuple(
            action
            for action, child in node.children.items()
            if self.minimax(child, node.player) == best
        )


@cache
def game_graph() -> GameGraph:
    return GameGraph()


class NeuralPolicy:
    """Frozen predictions, computed with the same single-board inference as play.

    Also supplies priors and leaf values to MCTS. It never consults minimax.
    """

    def __init__(self, model: ActorCritic, graph: GameGraph | None = None):
        self.graph = graph or game_graph()
        self.predictions = {}
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for board, node in self.graph.positions.items():
                    if not node.children:
                        continue
                    sign = 1 if node.player == "X" else -1
                    cells = torch.tensor(board, dtype=torch.float32) * sign
                    mask = torch.tensor([action in node.children for action in range(9)])
                    distribution, value = model.distribution(cells, mask)
                    self.predictions[board] = (
                        distribution.probs.numpy().astype(np.float64),
                        float(value.item()),
                    )
        finally:
            model.train(was_training)

    def action(self, board: Board) -> int:
        return int(self.predictions[board][0].argmax())


def audit_policy(policy, graph: GameGraph | None = None):
    """Enumerate all opponent continuations; probabilities are computed, not sampled."""
    graph = graph or game_graph()
    actions = {
        board: policy.action(board) for board, node in graph.positions.items() if node.children
    }
    for board, action in actions.items():
        if action not in graph.positions[board].children:
            raise ValueError("evaluated policy selected an illegal action")

    by_seat = {}
    for seat in ("X", "O"):

        @cache
        def probabilities(board, opponent):
            node = graph.positions[board]
            if not node.children:
                reward = graph.reward(board, seat)
                return (float(reward == 1), float(reward == 0), float(reward == -1))
            if node.player == seat:
                return probabilities(node.children[actions[board]], opponent)
            choices = (
                graph.optimal_actions(board) if opponent == "perfect" else tuple(node.children)
            )
            outcomes = [probabilities(node.children[action], opponent) for action in choices]
            return tuple(sum(outcome[i] for outcome in outcomes) / len(outcomes) for i in range(3))

        @cache
        def continuations(board):
            node = graph.positions[board]
            if not node.children:
                reward = graph.reward(board, seat)
                return (int(reward == 1), int(reward == 0), int(reward == -1))
            choices = (actions[board],) if node.player == seat else tuple(node.children)
            outcomes = [continuations(node.children[action]) for action in choices]
            return tuple(sum(outcome[i] for outcome in outcomes) for i in range(3))

        def witness(board):
            node = graph.positions[board]
            if not node.children:
                return []
            choices = (actions[board],) if node.player == seat else tuple(node.children)
            for action in choices:
                child = node.children[action]
                if continuations(child)[2] > 0:
                    return [action + 1, *witness(child)]
            return []

        wins, draws, losses = continuations(EMPTY)
        decisions = [board for board, node in graph.positions.items() if node.player == seat]
        mistakes = sum(actions[board] not in graph.optimal_actions(board) for board in decisions)
        by_seat[seat] = {
            "never_loses": losses == 0,
            "opponent_sequence_counts": {"wins": wins, "draws": draws, "losses": losses},
            "losing_example_squares": witness(EMPTY) if losses else None,
            "avoidable_action_errors": mistakes,
            "decision_positions": len(decisions),
            "vs_random": dict(zip(("wins", "draws", "losses"), probabilities(EMPTY, "random"))),
            "vs_perfect": dict(zip(("wins", "draws", "losses"), probabilities(EMPTY, "perfect"))),
        }
    return {
        "policy_mode": "deterministic greedy; no root exploration noise",
        "reachable_game_positions": len(graph.positions),
        "never_loses_both_seats": all(result["never_loses"] for result in by_seat.values()),
        "by_seat": by_seat,
    }


def selection_score(result):
    """Declared checkpoint criterion, shared by all experiment arms."""
    seats = list(result["by_seat"].values())
    return (
        int(not result["never_loses_both_seats"]),
        sum(seat["vs_perfect"]["losses"] for seat in seats),
        sum(seat["avoidable_action_errors"] for seat in seats),
        sum(seat["vs_random"]["losses"] for seat in seats),
        -sum(seat["vs_random"]["wins"] for seat in seats),
    )


def print_audit(result):
    for seat, numbers in result["by_seat"].items():
        print(
            f"  {seat}: never loses = {numbers['never_loses']}; "
            f"loss vs random {numbers['vs_random']['losses']:.2%}; "
            f"draw vs perfect {numbers['vs_perfect']['draws']:.2%}; "
            f"action errors {numbers['avoidable_action_errors']}/{numbers['decision_positions']}",
            flush=True,
        )
        if numbers["losing_example_squares"]:
            print(
                f"     A losing game (squares 1-9): {numbers['losing_example_squares']}", flush=True
            )


def versus_frozen_policy(policy, opponent: NeuralPolicy, graph=None):
    """Exact results: greedy evaluated agent versus a frozen, sampling neural opponent."""
    graph = graph or game_graph()
    results = {}
    for seat in ("X", "O"):

        @cache
        def outcomes(board):
            state = graph.positions[board]
            if not state.children:
                reward = graph.reward(board, seat)
                return (float(reward == 1), float(reward == 0), float(reward == -1))
            if state.player == seat:
                return outcomes(state.children[policy.action(board)])
            probabilities = opponent.predictions[board][0]
            weighted = [
                (probabilities[action], outcomes(child)) for action, child in state.children.items()
            ]
            return tuple(sum(weight * result[i] for weight, result in weighted) for i in range(3))

        results[seat] = dict(zip(("wins", "draws", "losses"), outcomes(EMPTY)))
    return results
