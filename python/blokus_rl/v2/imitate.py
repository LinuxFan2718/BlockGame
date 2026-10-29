"""Supervised pretraining of the v2 network on Pentobi games.

Targets per position: Pentobi's chosen move (one-hot policy), the game's
win/draw/loss and score margin, and the final owner of every cell. Positions
are sampled with random board symmetries. A held-out set of games measures
how often the network's top move matches Pentobi's and its value loss.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .._native import DuoReplay
from . import engine as v2
from .model import build, save
from .train import Learner, TrainConfig, to_tensors


def read_games(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A file still being appended may end mid-line.
    return rows


def split_games(paths, validation_games, seed):
    rows = read_games(paths)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    held = set(order[:validation_games].tolist())
    train, validation = DuoReplay(100_000_000), DuoReplay(100_000_000)
    for i, row in enumerate(rows):
        (validation if i in held else train).add_moves(row["moves"], 0)
    return train, validation, len(rows)


@torch.no_grad()
def evaluate(model, replay, device, batches=20, size=1024):
    model.eval()
    top1 = value = n = 0
    for k in range(batches):
        batch = to_tensors(replay.sample(size, len(replay), 99_000 + k, False), device)
        policy, outcome, _, _ = model(batch["planes"], batch["scalars"])
        rows, cols = batch["rows"], batch["cols"]
        padded = torch.full((batch["n"], batch["width"]), -1e9, device=device)
        padded[rows, cols] = policy.float()[rows, batch["legal"]]
        target = torch.zeros_like(padded)
        target[rows, cols] = batch["policy"]
        top1 += float((padded.argmax(1) == target.argmax(1)).float().sum())
        value += float(
            torch.nn.functional.cross_entropy(outcome.float(), batch["outcome"], reduction="sum")
        )
        n += batch["n"]
    model.train()
    return {"top1": round(top1 / n, 4), "value_ce": round(value / n, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-games", type=int, default=300)
    parser.add_argument("--device", default=v2.default_device())
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    train, validation, games = split_games(args.inputs, args.validation_games, args.seed)
    print(json.dumps({"games": games, "train_positions": len(train), "validation_positions": len(validation)}), flush=True)
    config = TrainConfig(
        blocks=args.blocks, channels=args.channels, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device, precision=args.precision,
        ema_decay=0.999, weight_decay=args.weight_decay,
    )
    learner = Learner(config, build({"blocks": args.blocks, "channels": args.channels, "context": 64}))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        learner.optimizer, max_lr=args.learning_rate, total_steps=args.steps, pct_start=0.05
    )
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    best = float("inf")
    log = (args.out / "progress.jsonl").open("a")
    for step in range(1, args.steps + 1):
        learner.step(train.sample(args.batch_size, len(train), int(rng.integers(2**62)), True))
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = {
                "step": step,
                "seconds": round(time.monotonic() - started),
                "epochs": round(step * args.batch_size / len(train), 2),
                **{k: round(v, 4) for k, v in learner.drain_metrics().items()},
                **{f"val_{k}": v for k, v in evaluate(learner.ema, validation, learner.device).items()},
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            save(args.out / "imitation.pt", learner.ema, {"step": step, "games": games})
            # Value heads overfit correlated positions first; keep the best one.
            if row["val_value_ce"] < best:
                best = row["val_value_ce"]
                save(args.out / "best.pt", learner.ema, {"step": step, "games": games, **row})
    log.close()


if __name__ == "__main__":
    main()
