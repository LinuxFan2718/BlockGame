"""Pentobi ladder for v2 networks: many concurrent games, one Pentobi process
per game slot, Pentobi moves generated in a thread pool while the Rust engine
searches the other games.

Every game is cross-checked: Pentobi's final score must equal ours, and every
Pentobi move must be legal in our engine. Openings (random plies) are played
from both seats; the reported score counts a draw as half a win.

Memory: each Pentobi process fills search trees sized by its level (about
2 GB at level 9), so the number of concurrent games is capped to fit a memory
budget, and a watchdog kills every Pentobi process if the machine runs short.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
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

GB = 1e9

# Pentobi 31.0 sizes its two search trees from the level it was started with
# (get_memory in libpentobi_mcts/Player.cpp): 2 GB at level 9, scaled down by
# the Trigon simulation counts below that, never more than a quarter of RAM.
# Long searches fill them: a level-9 process reaches 2.05 GB within 4 moves.
TRIGON_COUNTS = (100, 246, 457, 876, 1882, 5506, 19819, 99092, 495465)
PENTOBI_OVERHEAD = 50_000_000


def physical_memory():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def pentobi_bytes(level, total=None):
    """Memory one Pentobi process at `level` grows to."""
    total = physical_memory() if total is None else total
    wanted = 2_000_000_000
    if level < 9:
        wanted /= (TRIGON_COUNTS[8] / TRIGON_COUNTS[max(level, 5) - 1]) ** 0.8
    return int(min(wanted, total // 4)) + PENTOBI_OVERHEAD


def available_memory():
    """Bytes that can be used without swapping: free plus reclaimable pages."""
    if sys.platform == "darwin":
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        page = int(out.split("page size of ")[1].split()[0])
        pages = {k.strip(): int(v.strip(" .")) for k, v in
                 (line.split(":") for line in out.splitlines()[1:] if ":" in line)}
        reclaimable = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        return page * sum(pages[k] for k in reclaimable)
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("cannot read available memory")


def memory_pressure():
    """macOS pressure level: 1 normal, 2 warning, 4 critical (1 elsewhere)."""
    if sys.platform != "darwin":
        return 1
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or 1)


def resident_bytes(pids):
    if not pids:
        return 0
    out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, pids))],
                         capture_output=True, text=True).stdout
    return 1024 * sum(int(x) for x in out.split())


def concurrent_games(level, slots, pairs, budget):
    """Games to run at once so their Pentobi processes fit in `budget` bytes."""
    each = pentobi_bytes(level)
    fit = int(budget // each)
    if fit < 1:
        raise MemoryError(
            f"one level-{level} Pentobi needs {each / GB:.2f} GB; budget is {budget / GB:.2f} GB")
    return min(slots, 2 * pairs, fit)


class MemoryGuard:
    """Watches memory during a level. Kills every Pentobi process (ending the
    level with an error) if available memory falls below `floor` bytes or macOS
    reports memory pressure for three checks in a row; logs progress meanwhile."""

    def __init__(self, opponent, floor, progress, interval=5.0, log_every=600.0):
        self.opponent, self.floor, self.progress = opponent, floor, progress
        self.interval, self.log_every = interval, log_every
        self.lowest_available = available_memory()
        self.peak_pentobi = 0
        self.tripped = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self):
        pressured, last_log = 0, time.monotonic()
        while not self._stop.wait(self.interval):
            free = available_memory()
            used = resident_bytes(self.opponent.pids())
            self.lowest_available = min(self.lowest_available, free)
            self.peak_pentobi = max(self.peak_pentobi, used)
            pressured = pressured + 1 if memory_pressure() > 1 else 0
            if free < self.floor or pressured >= 3:
                self.tripped = (f"available memory {free / GB:.2f} GB, "
                                f"pressure level {memory_pressure()}")
                self.opponent.kill()
                return
            if time.monotonic() - last_log >= self.log_every:
                last_log = time.monotonic()
                print(f"[ladder] {self.progress()} games done, Pentobi using {used / GB:.1f} GB, "
                      f"{free / GB:.1f} GB available", file=sys.stderr, flush=True)

    def stop(self):
        self._stop.set()
        self._thread.join()


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

    def __init__(self, binary, level, threads, seed, floor=0):
        self.seats = {}
        self.binary, self.level, self.seed, self.floor = binary, level, seed, floor
        self.pool = ThreadPoolExecutor(threads)
        self.pending = {}
        self.pentobi_seconds = 0.0
        self.pentobi_moves = 0
        self.lock = threading.Lock()
        self.killed = False

    def seat(self, slot):
        with self.lock:
            if slot not in self.seats:
                if self.killed:
                    raise MemoryError("Pentobi processes were killed by the memory guard")
                # Started processes grow to their full size during their first
                # searches, so count that growth as already spoken for.
                each = pentobi_bytes(self.level)
                growth = max(len(self.seats) * each - resident_bytes(self.pids()), 0)
                free = available_memory() - growth
                if free < each + self.floor:
                    raise MemoryError(
                        f"not starting Pentobi #{len(self.seats) + 1}: {free / GB:.2f} GB "
                        f"free after growth, need {(each + self.floor) / GB:.2f} GB")
                self.seats[slot] = PentobiSeat(self.binary, self.level, self.seed + 1000 * slot)
            return self.seats[slot]

    def pids(self):
        return [s.pentobi.process.pid for s in list(self.seats.values())]

    def kill(self):
        with self.lock:
            self.killed = True
            for seat in self.seats.values():
                seat.pentobi.process.kill()

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


def play_level(evaluator, level, *, pairs, simulations, seed, slots, threads, binary, overrides,
               budget, floor):
    book = opening_book(pairs, plies=4, seed=seed)
    ours = v2.play_agent(simulations, **overrides)
    at_once = concurrent_games(level, slots, pairs, budget)
    print(f"[ladder] level {level}: each Pentobi grows to {pentobi_bytes(level) / GB:.2f} GB; "
          f"{at_once} games at once (budget {budget / GB:.1f} GB, "
          f"{available_memory() / GB:.1f} GB available)", file=sys.stderr, flush=True)
    engine = v2.make_engine(
        [ours, {**ours, "external": True}],
        at_once,
        2 * pairs,
        groups=2,
        alternate_seats=True,
        openings=book,
        shared_tree=False,
        seed=seed,
    )
    opponent = PentobiOpponent(binary, level, min(threads, at_once), seed, floor)
    games = []
    guard = MemoryGuard(opponent, floor, lambda: f"{len(games)}/{2 * pairs}")
    started = time.monotonic()
    try:
        try:
            v2.run(engine, evaluator, on_finished=games.extend, external=opponent)
        except Exception as error:
            if guard.tripped:
                raise MemoryError(f"memory guard stopped level {level}: {guard.tripped}") from error
            raise
        guard.stop()
        # Cross-check final scores with an independent Pentobi replay.
        checker = PentobiSeat(binary, 1, seed)
        try:
            for g in games:
                checker.game = None
                checker.check_final(g["moves"], g["scores"][0] - g["scores"][1])
        finally:
            checker.close()
    finally:
        guard.stop()
        opponent.close()
    result = arena.summarize(games)
    result.update(
        level=level,
        simulations=simulations,
        seconds=round(time.monotonic() - started, 1),
        pentobi_seconds_per_move=round(opponent.pentobi_seconds / max(opponent.pentobi_moves, 1), 3),
        games_at_once=at_once,
        peak_pentobi_gb=round(guard.peak_pentobi / GB, 2),
        lowest_available_gb=round(guard.lowest_available / GB, 2),
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
    parser.add_argument("--memory-gb", type=float, default=None,
                        help="RAM the Pentobi processes may use "
                             "(default: 3/4 of what is available when each level starts)")
    parser.add_argument("--min-free-gb", type=float, default=1.0,
                        help="kill Pentobi if available memory falls below this")
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
        budget = args.memory_gb * GB if args.memory_gb else 0.75 * available_memory()
        result, games = play_level(
            evaluator, level, pairs=args.pairs, simulations=args.simulations, seed=args.seed,
            slots=args.slots, threads=args.threads, binary=args.binary, overrides=overrides,
            budget=budget, floor=args.min_free_gb * GB,
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
