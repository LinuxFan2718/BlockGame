"""Loopback-only human play. Rust owns rules; one worker owns neural inference."""

import errno
import hashlib
import json
import secrets
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import torch

from ._native import DuoGame, duo_pieces
from .duo_device import available_devices, inference_device
from .duo_env import PASS
from .duo_learning import choose, load_duo
from .duo_mcts import evaluate_records, search_games

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).with_name("web")
SIMULATIONS = (0, 128, 512, 2048)
MODEL_PATHS = {
    "replicated": (
        "Control · seed 1071",
        "runs/control-replication-final-20261028T171624Z/duo-control-replication-1071/frozen-control1024.pt",
    ),
    "control": (
        "Control · seed 1070",
        "runs/integrated-study-final-20261028T152508Z/duo-integrated-teacher-1070/frozen-control1024.pt",
    ),
    "teacher": (
        "Teacher · seed 1070",
        "runs/integrated-study-final-20261028T152508Z/duo-integrated-teacher-1070/frozen-teacher2048.pt",
    ),
    "parent": (
        "Original parent",
        "runs/duo-level3-final-20261027T0049/duo-strong-search2048-level3-2603000/search-2048/model.pt",
    ),
}


class StaleGame(ValueError):
    pass


class CancelledSearch(Exception):
    pass


def model_catalog(checkpoint=None):
    if checkpoint is not None:
        path = Path(checkpoint).resolve(strict=True)
        return {"custom": (path.stem, path)}
    return {key: (label, ROOT / path) for key, (label, path) in MODEL_PATHS.items() if (ROOT / path).is_file()}


class LocalGame:
    def __init__(self, catalog, device="auto", *, policy_loader=load_duo):
        if not catalog:
            raise ValueError("No local checkpoint found. Supply a checkpoint to duo-web.")
        self.catalog, self.policy_loader = catalog, policy_loader
        self.hashes = {
            key: hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None
            for key, (_, path) in catalog.items()
        }
        self.default_device = device
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="blokus-bot")
        self.models = {}
        self.cancel = threading.Event()
        self.new({"human": 0, "simulations": 2048, "model": next(iter(catalog)), "device": device})

    def close(self):
        self.cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def config(self):
        return {
            "application": "blokus-duo-local",
            "models": [{"id": key, "label": label} for key, (label, _) in self.catalog.items()],
            "devices": available_devices(),
            "default_device": self.default_device,
            "pieces": duo_pieces(),
            "simulations": SIMULATIONS,
        }

    def new(self, options):
        human = options.get("human", 0)
        simulations = options.get("simulations", 2048)
        key = options.get("model", next(iter(self.catalog)))
        if type(human) is not int or human not in (0, 1):
            raise ValueError("Choose first or second seat.")
        if type(simulations) is not int or simulations not in SIMULATIONS:
            raise ValueError("Unsupported search budget.")
        if key not in self.catalog:
            raise ValueError("Unknown model.")
        device = inference_device(options.get("device", self.default_device))
        with self.lock:
            self.cancel.set()
            self.cancel = threading.Event()
            self.game, self.actions = DuoGame(), []
            self.id, self.version = uuid.uuid4().hex, 0
            self.human, self.simulations, self.model_key = human, simulations, key
            self.device = str(device)
            self.error, self.last_search, self.thinking_since = None, None, None
            self._advance()
            return self.state()

    def _check(self, data):
        if data.get("id") != self.id or data.get("version") != self.version:
            raise StaleGame("The board changed. Please try your move again.")

    def _apply(self, action):
        player = self.game.current_player
        cells = [] if action == PASS else self.game.describe(action)[3]
        piece = None if action == PASS else self.game.describe(action)[0]
        self.game.step(action)
        self.actions.append({"player": player, "action": action, "cells": cells, "piece": piece})
        self.version += 1

    def _advance(self):
        while not self.game.terminated and self.game.legal_actions() == [PASS]:
            self._apply(PASS)
        self.status = "finished" if self.game.terminated else "your_turn"
        if not self.game.terminated and self.game.current_player != self.human:
            self.status, self.thinking_since = "thinking", time.monotonic()
            self.executor.submit(
                self._bot, self.game.copy(), self.id, self.version, self.cancel,
                self.model_key, self.device, self.simulations,
            )

    def move(self, data):
        with self.lock:
            self._check(data)
            if self.status != "your_turn":
                raise ValueError("Wait for your turn.")
            action = data.get("action")
            if type(action) is not int or action not in self.game.legal_actions():
                raise ValueError("That placement is not legal.")
            self._apply(action)
            self.last_search = None
            self._advance()
            return self.state()

    def undo(self, data):
        with self.lock:
            self._check(data)
            human_moves = [i for i, row in enumerate(self.actions)
                           if row["player"] == self.human and row["action"] != PASS]
            if not human_moves:
                raise ValueError("No human move to undo yet.")
            prefix = self.actions[: human_moves[-1]]
            self.cancel.set()
            self.cancel = threading.Event()
            self.game, self.actions = DuoGame(), []
            for row in prefix:
                self._apply(row["action"])
            # Monotonic version even when returning to an earlier board.
            self.version = data["version"] + 1
            self.error, self.last_search, self.thinking_since = None, None, None
            self._advance()
            return self.state()

    def _bot(self, game, game_id, version, cancel, model_key, device, simulations):
        if cancel.is_set():
            return
        start = time.monotonic()
        try:
            model_id = (model_key, device)
            if model_id not in self.models:
                self.models[model_id] = self.policy_loader(self.catalog[model_key][1]).to(device).eval()
            model = self.models[model_id]

            def evaluate(records):
                if cancel.is_set():
                    raise CancelledSearch()
                return evaluate_records(model, records)

            metrics = {}
            with torch.inference_mode():
                if simulations:
                    action = search_games(
                        model, [game], simulations, evaluator=evaluate, metrics=metrics,
                    )[0].action()
                else:
                    action = choose(game, model, np.random.default_rng(42))
            elapsed = time.monotonic() - start
            with self.lock:
                if cancel.is_set() or self.id != game_id or self.version != version:
                    return
                self._apply(action)
                self.last_search = {"seconds": elapsed, "simulations": simulations, **metrics}
                self.thinking_since = None
                self._advance()
        except CancelledSearch:
            return
        except Exception as exc:
            with self.lock:
                if self.id == game_id and self.version == version and not cancel.is_set():
                    self.status, self.error = "error", f"Bot inference failed: {exc}"
                    self.thinking_since = None

    def state(self):
        with self.lock:
            moves = []
            if self.status == "your_turn":
                for action in self.game.legal_actions():
                    if action != PASS:
                        piece, _, _, cells = self.game.describe(action)
                        moves.append({"id": action, "piece": piece, "cells": cells})
            winner = None
            if self.game.terminated:
                a, b = self.game.scores
                winner = "draw" if a == b else ("human" if (a > b) == (self.human == 0) else "bot")
            return {
                "id": self.id, "version": self.version, "board": list(self.game.board),
                "human": self.human, "current_player": self.game.current_player,
                "scores": self.game.scores,
                "remaining": [self.game.remaining(0), self.game.remaining(1)],
                "starts": self.game.required_starts(), "legal_moves": moves,
                "status": self.status, "winner": winner, "error": self.error,
                "actions": self.actions.copy(), "device": self.device, "model": self.model_key,
                "simulations": self.simulations, "last_search": self.last_search,
                "checkpoint_sha256": self.hashes[self.model_key],
                "can_undo": any(row["player"] == self.human and row["action"] != PASS
                                for row in self.actions),
                "thinking_seconds": time.monotonic() - self.thinking_since if self.thinking_since else 0,
            }


class PlayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, session):
        self.session, self.token = session, secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), PlayHandler)


class PlayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, status, body, kind="application/json"):
        if kind == "application/json":
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Closing/reloading a tab does not interrupt a bot turn.

    def _local_request(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in allowed and (
            not origin or origin in {f"http://{host}" for host in allowed}
        )

    def do_GET(self):
        if not self._local_request():
            return self._reply(403, {"error": "Local requests only."})
        path = urlsplit(self.path).path
        if path == "/api/config":
            return self._reply(200, {**self.server.session.config(), "token": self.server.token})
        if path == "/api/state":
            return self._reply(200, self.server.session.state())
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/rules": ("rules.html", "text/html; charset=utf-8")}
        if path not in files:
            return self._reply(404, {"error": "Not found."})
        filename, kind = files[path]
        return self._reply(200, (ASSETS / filename).read_bytes(), kind)

    def do_POST(self):
        if not self._local_request() or self.headers.get("X-Blokus-Token") != self.server.token:
            return self._reply(403, {"error": "Reload the local game page and try again."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a request object.")
            routes = {"/api/new": self.server.session.new, "/api/move": self.server.session.move,
                      "/api/undo": self.server.session.undo}
            path = urlsplit(self.path).path
            if path not in routes:
                return self._reply(404, {"error": "Not found."})
            return self._reply(200, routes[path](data))
        except StaleGame as exc:
            return self._reply(409, {"error": str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            return self._reply(400, {"error": str(exc)})


def serve(checkpoint=None, device="auto", port=8765, open_browser=True):
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535.")
    session = LocalGame(model_catalog(checkpoint), device)
    try:
        server = PlayServer(port, session)
    except OSError as exc:
        session.close()
        if exc.errno == errno.EADDRINUSE and port:
            url = f"http://127.0.0.1:{port}"
            try:
                with urlopen(url + "/api/config", timeout=2) as response:
                    existing = json.load(response)
                if existing.get("application") == "blokus-duo-local":
                    print(f"Duo is already running: {url}\nReusing its current game.", flush=True)
                    if open_browser:
                        webbrowser.open(url)
                    return
            except (OSError, ValueError):
                pass
        raise
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Blokus Duo: {url}\nInference: {session.device}. Ctrl-C stops the local app.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    from .cli import main

    main()
