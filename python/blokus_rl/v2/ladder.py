"""Pentobi ladder for v2 networks: many concurrent games, one Pentobi process
per game slot, Pentobi moves generated in a thread pool while the Rust engine
searches the other games.

Every game is cross-checked: Pentobi's final score must equal ours, and every
Pentobi move must be legal in our engine. Openings (random plies) are played
from both seats; the reported score counts a draw as half a win.
"""

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import torch

from .._native import DuoGame
from ..duo_crossplay import opening_book
from ..duo_pentobi import DEFAULT_BINARY, Pentobi
from . import arena
from . import engine as v2
from .model import load

PASS = 17836


class PentobiSeat:
    """Keeps one Pentobi process in sync with one engine slot."""

    def __init__(self, binary, level, seed):
        self.pentobi = Pentobi(binary, level=level, seed=seed, timeout=600)
        self.seed = seed
        self.game = None
        self.moves = []

    def sync(self, moves):
        moves = [int(m) for m in moves]
        if self.game is None or moves[: len(self.moves)] != self.moves:
            self.seed += 1
            self.pentobi.reset(self.seed)
            self.game, self.moves = DuoGame(), []
        for m in moves[len(self.moves) :]:
            self.pentobi.play(self.game, m)
            self.game.step(m)
            self.moves.append(m)

    def generate(self, moves):
        self.sync(moves)
        action = self.pentobi.generate(self.game)
        # Pentobi has already applied its own move; keep our copy in step.
        self.game.step(action)
        self.moves.append(action)
        return action

    def check_final(self, moves, margin):
        self.sync(moves)
        theirs = self.pentobi.final_margin()
        if theirs != margin:
            raise RuntimeError(f"final score disagreement: ours {margin}, Pentobi {theirs}")
        self.game = None

    def close(self):
        self.pentobi.close()


class PentobiOpponent:
    """`external` callback for v2.run: asynchronous Pentobi moves per slot."""

    def __init__(self, binary, level, slots, threads, seed):
        self.seats = {}
        self.binary, self.level, self.seed = binary, level, seed
        self.pool = ThreadPoolExecutor(threads)
        self.pending = {}
        self.pentobi_seconds = 0.0
        self.pentobi_moves = 0

    def seat(self, slot):
        if slot not in self.seats:
            self.seats[slot] = PentobiSeat(self.binary, self.level, self.seed + 1000 * slot)
        return self.seats[slot]

    def _generate(self, slot, moves):
        started = time.perf_counter()
        action = self.seat(slot).generate(moves)
        return action, time.perf_counter() - started

    def __call__(self, engine):
        waiting = engine.waiting_external()
        for slot, moves in waiting:
            if slot not in self.pending:
                self.pending[slot] = self.pool.submit(self._generate, slot, list(moves))
        done = [s for s, f in self.pending.items() if f.done()]
        if not done and self.pending and len(waiting) >= engine.active_games():
            # Nothing for the network to do: block until Pentobi answers.
            wait(list(self.pending.values()), return_when=FIRST_COMPLETED)
            done = [s for s, f in self.pending.items() if f.done()]
        for slot in done:
            action, seconds = self.pending.pop(slot).result()
            self.pentobi_seconds += seconds
            self.pentobi_moves += 1
            engine.external_move(slot, action)

    def close(self):
        self.pool.shutdown(wait=True)
        for seat in self.seats.values():
            seat.close()


def play_level(evaluator, level, *, pairs, simulations, seed, slots, threads, binary, overrides):
    book = opening_book(pairs, plies=4, seed=seed)
    ours = v2.play_agent(simulations, **overrides)
    engine = v2.make_engine(
        [ours, {**ours, "external": True}],
        min(slots, 2 * pairs),
        2 * pairs,
        groups=2,
        alternate_seats=True,
        openings=book,
        shared_tree=False,
        seed=seed,
    )
    opponent = PentobiOpponent(binary, level, slots, threads, seed)
    games = []
    started = time.monotonic()
    try:
        v2.run(engine, evaluator, on_finished=games.extend, external=opponent)
        # Cross-check final scores with an independent Pentobi replay.
        checker = PentobiSeat(binary, 1, seed)
        try:
            for g in games:
                checker.game = None
                checker.check_final(g["moves"], g["scores"][0] - g["scores"][1])
        finally:
            checker.close()
    finally:
        opponent.close()
    result = arena.summarize(games)
    result.update(
        level=level,
        simulations=simulations,
        seconds=round(time.monotonic() - started, 1),
        pentobi_seconds_per_move=round(opponent.pentobi_seconds / max(opponent.pentobi_moves, 1), 3),
        counters=engine.counters(),
    )
    return result, games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7_000_000)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default=v2.default_device())
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--solver-max-moves", type=int, default=30)
    parser.add_argument("--solver-root-nodes", type=int, default=2_000_000)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--symmetries", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--stop-below", type=float, default=None,
                        help="stop climbing once a level scores below this")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    model, data = load(args.checkpoint, args.device)
    torch.set_grad_enabled(False)
    evaluator = v2.Evaluator(
        model, args.device, score_weight=args.score_weight, symmetries=args.symmetries
    )
    overrides = dict(
        solver_max_moves=args.solver_max_moves,
        solver_root_nodes=args.solver_root_nodes,
        score_weight=args.score_weight,
    )
    results = []
    for level in args.levels:
        result, games = play_level(
            evaluator, level, pairs=args.pairs, simulations=args.simulations, seed=args.seed,
            slots=args.slots, threads=args.threads, binary=args.binary, overrides=overrides,
        )
        result["checkpoint"] = str(args.checkpoint)
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"level-{level}.json").write_text(json.dumps(result, indent=2) + "\n")
            (args.out / f"level-{level}-games.json").write_text(json.dumps(games) + "\n")
        if args.stop_below is not None and result["score"] < args.stop_below:
            break
    return results


if __name__ == "__main__":
    main()
