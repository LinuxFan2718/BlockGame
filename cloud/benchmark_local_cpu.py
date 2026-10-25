"""Match the Modal search workload locally, with no cloud or model updates."""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from blokus_rl._native import DuoGame
from blokus_rl.duo_conv import DuoConvActorCritic
from blokus_rl.duo_learning import load_duo
from blokus_rl.duo_mcts import search_games
from blokus_rl.learning import configure_torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    checkpoint = ROOT / "runs/duo-frontier-training-342/best.pt"
    destination = ROOT / "runs/modal-matched-mac-cpu-260642.json"
    if destination.exists():
        raise ValueError("refusing to overwrite existing measurements")
    configure_torch(260642)
    base = load_duo(checkpoint)
    result = {
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "search": [],
        "caveat": "One CPU thread/coordinator. Timing only; no model updates.",
    }
    for architecture in ("additive", "conv"):
        configure_torch(260642)
        if architecture == "additive":
            model = type(base)(base.hidden_size)
            model.load_state_dict(base.state_dict())
        else:
            model = DuoConvActorCritic()
        model.eval()
        for repetition, live in enumerate((8, 32, 32)):
            games = [DuoGame() for _ in range(live)]
            for index, game in enumerate(games):
                opening_rng = np.random.default_rng(260642 + index)
                for _ in range(4):
                    game.step(int(opening_rng.choice(game.legal_actions())))
            metrics = {}
            start = time.monotonic()
            for _ in range(4):
                searched = search_games(model, games, 128, 8, metrics=metrics)
                for game, search in zip(games, searched, strict=True):
                    game.step(search.action())
            seconds = time.monotonic() - start
            result["search"].append(
                {
                    "architecture": architecture,
                    "mode": "cpu",
                    "repetition": repetition,
                    "live_games": live,
                    "plies": 4,
                    "seconds": seconds,
                    "decisions_per_second": 4 * live / seconds,
                    **metrics,
                }
            )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
