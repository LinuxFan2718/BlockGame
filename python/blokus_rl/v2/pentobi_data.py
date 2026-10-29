"""Generate Pentobi-vs-Pentobi games for imitation learning.

Each worker thread drives its own single-threaded Pentobi process, which plays
both colors. A few random opening plies (synchronized into Pentobi) and a
fresh random seed per game diversify the games. Every game's final score is
checked against Pentobi's. Output is JSON lines and resumable: rerunning
appends until the requested game count is reached.

Build an imitation replay with `--replay out.bin`: every non-forced move
becomes a one-hot policy target, with the game's real outcome, margin and
final ownership as value and auxiliary targets.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .._native import DuoGame, DuoReplay
from ..duo_pentobi import DEFAULT_BINARY, Pentobi


def play_game(pentobi, level, seed, random_plies):
    rng = np.random.default_rng(seed)
    pentobi.reset(seed)
    game, moves = DuoGame(), []
    started = time.perf_counter()
    while not game.terminated:
        if len(moves) < random_plies:
            action = int(rng.choice(game.legal_actions()))
            pentobi.play(game, action)
        else:
            action = pentobi.generate(game)
        game.step(action)
        moves.append(int(action))
    margin = game.scores[0] - game.scores[1]
    if pentobi.final_margin() != margin:
        raise RuntimeError("final score disagreement with Pentobi")
    return {
        "level": level,
        "seed": seed,
        "random_plies": random_plies,
        "moves": moves,
        "scores": list(game.scores),
        "seconds": round(time.perf_counter() - started, 3),
    }


def generate(out, levels, games, workers, seed, binary, random_plies, max_seconds=None):
    out = Path(out)
    done = out.read_text().count("\n") if out.exists() else 0
    lock = threading.Lock()
    counter = {"next": done}
    deadline = time.monotonic() + max_seconds if max_seconds else None

    def worker(index):
        pentobi, current_level = None, None
        try:
            while True:
                with lock:
                    k = counter["next"]
                    if k >= games or (deadline and time.monotonic() > deadline):
                        return
                    counter["next"] += 1
                level = levels[k % len(levels)]
                if level != current_level:
                    if pentobi is not None:
                        pentobi.close()
                    pentobi = Pentobi(binary, level=level, seed=seed + k, timeout=3600)
                    current_level = level
                plies = random_plies[k % len(random_plies)]
                row = play_game(pentobi, level, seed + 7919 * k, plies)
                with lock, out.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
        finally:
            if pentobi is not None:
                pentobi.close()

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(worker, i) for i in range(workers)]:
            future.result()


def build_replay(paths, capacity=50_000_000):
    replay = DuoReplay(capacity)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            replay.add_moves(json.loads(line)["moves"], 0)
    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[6])
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=31_000_000)
    parser.add_argument("--random-plies", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--replay", type=Path, help="write an imitation replay from --inputs")
    parser.add_argument("--inputs", type=Path, nargs="*")
    args = parser.parse_args()
    if args.replay:
        replay = build_replay(args.inputs)
        args.replay.write_bytes(bytes(replay.to_bytes()))
        print(json.dumps({"games": replay.games, "positions": len(replay)}))
        return
    generate(
        args.out, args.levels, args.games, args.workers, args.seed, args.binary,
        args.random_plies, args.max_seconds,
    )


if __name__ == "__main__":
    main()
