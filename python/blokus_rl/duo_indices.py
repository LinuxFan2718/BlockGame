"""Exact, immutable engine-action to spatial-policy lookup for either observer."""

from functools import lru_cache

import numpy as np

from ._native import DuoGame
from .duo_env import CELLS, PASS


@lru_cache(maxsize=1)
def action_index_table():
    game = DuoGame()
    placements = {}
    for action in range(PASS):
        try:
            _, _, _, cells = game.describe(action)
        except ValueError:
            continue
        placements[action] = frozenset(cells)
    by_cells = {cells: action for action, cells in placements.items()}
    if len(by_cells) != len(placements):
        raise RuntimeError("ambiguous native placement geometry")
    table = np.full((2, PASS + 1), np.iinfo(np.uint16).max, dtype=np.uint16)
    for action, cells in placements.items():
        table[0, action] = action
        table[1, action] = by_cells[frozenset(CELLS - 1 - c for c in cells)]
    table[:, PASS] = PASS
    table.setflags(write=False)
    return table


def policy_indices(actions, player):
    if player not in (0, 1):
        raise ValueError("expected player 0 or 1")
    indices = action_index_table()[player, actions]
    if np.any(indices == np.iinfo(np.uint16).max):
        raise ValueError("unknown native placement")
    return indices
