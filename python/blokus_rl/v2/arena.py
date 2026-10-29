"""Matches between a v2 network (searching in the Rust engine) and an outside
player: a baseline policy, Pentobi, or anything that can choose a move from a
move history. Openings are played from both seats; scores count a draw as half.
"""

import numpy as np

from .._native import DuoGame
from ..duo_crossplay import opening_book, paired_interval
from . import engine as v2

PASS = 17836


def replay_moves(moves):
    game = DuoGame()
    for m in moves:
        game.step(int(m))
    return game


def random_mover(seed=0):
    rng = np.random.default_rng(seed)

    def choose(game):
        return int(rng.choice(game.legal_actions()))

    return choose


def largest_mover(seed=0):
    """Largest piece available, ties broken at random (v1's baseline)."""
    rng = np.random.default_rng(seed)

    def choose(game):
        actions = game.legal_actions()
        if actions == [PASS]:
            return PASS
        sizes = np.array([len(game.describe(a)[3]) for a in actions])
        return int(rng.choice(np.asarray(actions)[sizes == sizes.max()]))

    return choose


def summarize(games, agent=0):
    """Score of `agent` (engine agent index) by opening pair."""
    pairs = {}
    wins = draws = losses = 0
    for g in games:
        seat = g["seats"].index(agent)
        margin = g["scores"][seat] - g["scores"][1 - seat]
        score = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        wins += margin > 0
        draws += margin == 0
        losses += margin < 0
        pairs.setdefault(g["index"] // 2, []).append(score)
    complete = [v for v in pairs.values() if len(v) == 2]
    return {
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": float(np.mean([s for v in complete for s in v])) if complete else None,
        "interval": paired_interval(complete) if len(complete) > 1 else None,
    }


def play(
    evaluator,
    choose,
    *,
    pairs,
    simulations,
    slots=128,
    seed=1,
    opening_plies=4,
    agent_overrides=None,
):
    """Network (agent 0) versus `choose(game) -> action` (agent 1)."""
    book = opening_book(pairs, plies=opening_plies, seed=seed)
    ours = v2.play_agent(simulations, **(agent_overrides or {}))
    engine = v2.make_engine(
        [ours, {**ours, "external": True}],
        min(slots, 2 * pairs),
        2 * pairs,
        alternate_seats=True,
        openings=book,
        shared_tree=False,
        seed=seed,
    )

    def external(engine):
        for slot, moves in engine.waiting_external():
            engine.external_move(slot, choose(replay_moves(moves)))

    games = []
    v2.run(engine, evaluator, on_finished=games.extend, external=external)
    return summarize(games)
