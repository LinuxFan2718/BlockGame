"""Independent engine checks and matches via Pentobi's documented GTP interface."""

import os
import select
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ._native import DuoGame
from .duo_env import PASS
from .duo_learning import choose

DEFAULT_BINARY = Path(".local/pentobi-build/pentobi_gtp/pentobi-gtp")


def move_cells(move, rotate=False):
    if not move or move.lower() == "pass":
        return frozenset()
    cells = [(14 - int(s[1:])) * 14 + ord(s[0].lower()) - ord("a") for s in move.split(",")]
    if any(not 0 <= c < 196 for c in cells):
        raise ValueError(f"invalid Pentobi move: {move}")
    return frozenset(195 - c if rotate else c for c in cells)


def move_string(cells, rotate=False):
    if not cells:
        return "pass"
    cells = sorted(195 - c if rotate else c for c in cells)
    return ",".join(f"{chr(97 + c % 14)}{14 - c // 14}" for c in cells)


def action_cells(game, action):
    return frozenset() if action == PASS else frozenset(game.describe(action)[3])


class Pentobi:
    def __init__(self, binary=DEFAULT_BINARY, level=1, seed=42, timeout=60):
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            [
                str(Path(binary).resolve()),
                "--game",
                "duo",
                "--level",
                str(level),
                "--threads",
                "1",
                "--seed",
                str(seed),
                "--nobook",
                "--noresign",
                "--quiet",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
        )
        self.buffer = b""
        self.timeout = timeout
        self.rotate = None
        if self.command("version") != "31.0":
            self.close()
            raise RuntimeError("benchmark requires pinned Pentobi 31.0")

    def command(self, command):
        self.process.stdin.write((command + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while b"\n\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"Pentobi command timed out: {command}")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"Pentobi exited during: {command}")
            self.buffer += chunk
        response, self.buffer = self.buffer.split(b"\n\n", 1)
        response = response.decode().strip()
        if not response.startswith("="):
            raise RuntimeError(f"Pentobi rejected {command!r}: {response}")
        return response[1:].strip()

    def reset(self, seed):
        self.command("clear_board")
        self.command(f"set_random_seed {seed}")
        self.rotate = None

    def play(self, game, action):
        if action == PASS:
            if game.legal_actions() != [PASS]:
                raise ValueError("pass must be forced")
            return  # Pentobi omits forced passes; explicit colors advance play.
        cells = action_cells(game, action)
        if self.rotate is None:
            self.rotate = 135 in cells  # J5 -> canonical first-player E10.
        color = "b" if game.current_player == 0 else "w"
        self.command(f"play {color} {move_string(cells, self.rotate)}")

    def generate(self, game):
        if game.legal_actions() == [PASS]:
            return PASS
        if self.rotate is None:
            self.rotate = False
        color = "b" if game.current_player == 0 else "w"
        cells = move_cells(self.command(f"genmove {color}"), self.rotate)
        mapping = {action_cells(game, a): a for a in game.legal_actions()}
        if cells not in mapping:
            raise RuntimeError(f"Pentobi generated a move our engine rejects: {cells}")
        return mapping[cells]

    def check_legal(self, game):
        color = "b" if game.current_player == 0 else "w"
        reply = self.command(f"all_legal {color}")
        external = {move_cells(s, bool(self.rotate)) for s in reply.splitlines()} or {frozenset()}
        ours = {action_cells(game, a) for a in game.legal_actions()}
        if self.rotate is None:
            ours = {s for s in ours if 60 in s}  # E10; other half is rotation-equivalent.
        if external != ours:
            raise RuntimeError(
                f"legal-set mismatch: external-only {external - ours}, ours-only {ours - external}"
            )
        return len(ours)

    def final_margin(self):
        score = self.command("final_score")
        if score == "0":
            return 0.0
        return float(score[2:]) * (1 if score.startswith("B+") else -1)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def match(pentobi, policy, seat, seed, opening=(), check_legal=False):
    pentobi.reset(seed)
    game, moves = DuoGame(), []
    rng = np.random.default_rng(seed)
    while not game.terminated:
        if check_legal:
            pentobi.check_legal(game)
        if len(moves) < len(opening):
            action = opening[len(moves)]
            pentobi.play(game, action)
        elif game.current_player == seat:
            action = choose(game, policy, rng)
            pentobi.play(game, action)
        else:
            action = pentobi.generate(game)
        game.step(action)
        moves.append(action)
    margin = game.scores[0] - game.scores[1]
    external_margin = pentobi.final_margin()
    if margin != external_margin:
        raise RuntimeError(f"final score disagreement: Rust {margin}, Pentobi {external_margin}")
    return {
        "seat": seat,
        "seed": seed,
        "actions": moves,
        "scores": game.scores,
        "margin": margin if seat == 0 else -margin,
    }
