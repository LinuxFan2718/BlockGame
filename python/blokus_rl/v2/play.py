"""Choose one move for a single position with a v2 network (used by the app).

The Rust engine replays the move history as an opening, searches for the side
to move (exact endgame solving included) and stops as soon as that side has
moved: the other seat is external, so the engine waits for it.
"""

import numpy as np
import torch

from .._native import duo_v2_encode
from . import engine as v2
from .arena import replay_moves


class Cancelled(Exception):
    pass


@torch.no_grad()
def raw_policy_move(evaluator, moves):
    """Network-only move: the highest-probability legal placement."""
    planes, scalars, legal, actions = duo_v2_encode([int(m) for m in moves])
    batch = (
        planes,
        scalars,
        np.asarray(legal, np.uint16).tobytes(),
        np.asarray([len(legal)], np.uint32).tobytes(),
        np.zeros(1, np.uint8).tobytes(),
    )
    probabilities, _ = evaluator.collect(evaluator.submit(batch))
    return int(actions[int(np.argmax(probabilities))])


def choose_move(
    evaluator,
    moves,
    simulations,
    *,
    solver_max_moves=30,
    solver_root_nodes=2_000_000,
    score_weight=0.05,
    leaf_batch=16,
    cancel=None,
    seed=0,
):
    moves = [int(m) for m in moves]
    game = replay_moves(moves)
    if game.terminated:
        raise ValueError("the game is over")
    if simulations <= 0:
        return raw_policy_move(evaluator, moves)
    bot = v2.play_agent(
        simulations,
        leaf_batch=leaf_batch,
        solver_max_moves=solver_max_moves,
        solver_root_nodes=solver_root_nodes,
        score_weight=score_weight,
    )
    other = {**bot, "external": True}
    agents = [bot, other] if game.current_player == 0 else [other, bot]
    engine = v2.make_engine(
        agents, 1, 1, groups=1, openings=[moves], shared_tree=False, seed=seed
    )
    while True:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        batch = engine.prepare(0)
        waiting = engine.waiting_external()
        if waiting:
            return int(waiting[0][1][len(moves)])
        finished = engine.drain()
        if finished:
            return int(finished[0]["moves"][len(moves)])
        handle = evaluator.submit(batch)
        if handle is None:
            raise RuntimeError("engine produced no work and no move")
        engine.apply(0, *evaluator.collect(handle))
