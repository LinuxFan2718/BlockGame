"""Warm-start a larger v2 network from an existing run: train it on that run's
self-play replay plus Pentobi games, with the run's network as teacher.

Targets are the usual ones (search visit counts, win/draw/loss, margin,
ownership), blended with the teacher's predictions on the same position. The
teacher's smooth win/draw/loss estimate keeps a larger network from memorizing
the results of the comparatively few games in the replay, so the student
starts near the teacher's strength and self-play can continue from there.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .._native import DuoReplay
from . import engine as v2
from .model import MARGIN_SCALE, build, load, save
from .train import Learner, TrainConfig, concat_samples


def teacher_targets(teacher, policy_weight, value_weight):
    @torch.no_grad()
    def blend(batch):
        policy, outcome, margin, _ = teacher(batch["planes"], batch["scalars"])
        rows, cols = batch["rows"], batch["cols"]
        padded = torch.full((batch["n"], batch["width"]), -1e9, device=policy.device)
        padded[rows, cols] = policy.float()[rows, batch["legal"]]
        moves = torch.softmax(padded, 1)[rows, cols]
        batch["policy"] = (1 - policy_weight) * batch["policy"] + policy_weight * moves
        results = F.one_hot(batch["outcome"], 3).float()
        batch["outcome"] = (1 - value_weight) * results + value_weight * torch.softmax(outcome.float(), 1)
        batch["margin"] = (1 - value_weight) * batch["margin"] + value_weight * margin.float() * MARGIN_SCALE

    return blend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--imitation", type=Path)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--imitation-fraction", type=float, default=0.1)
    parser.add_argument("--teacher-policy", type=float, default=0.25)
    parser.add_argument("--teacher-value", type=float, default=0.5)
    parser.add_argument("--device", default=v2.default_device())
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--snapshots", type=int, default=4, help="evenly spaced checkpoints to keep")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    replay = DuoReplay.from_bytes(args.replay.read_bytes())
    imitation = DuoReplay.from_bytes(args.imitation.read_bytes()) if args.imitation else None
    teacher, _ = load(args.teacher, args.device)
    config = TrainConfig(
        blocks=args.blocks, channels=args.channels, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        device=args.device, precision=args.precision, ema_decay=0.999,
    )
    learner = Learner(config, build({"blocks": args.blocks, "channels": args.channels, "context": 64}))
    params = sum(p.numel() for p in learner.model.parameters())
    print(json.dumps({"replay_positions": len(replay), "imitation_positions": len(imitation or []),
                      "parameters": params}), flush=True)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        learner.optimizer, max_lr=args.learning_rate, total_steps=args.steps, pct_start=0.05
    )
    blend = teacher_targets(teacher, args.teacher_policy, args.teacher_value)
    imitation_n = int(round(args.imitation_fraction * args.batch_size)) if imitation else 0
    snapshot_every = max(1, args.steps // args.snapshots)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    log = (args.out / "progress.jsonl").open("a")
    for step in range(1, args.steps + 1):
        parts = [replay.sample(args.batch_size - imitation_n, len(replay), int(rng.integers(2**62)), True)]
        if imitation_n:
            parts.append(imitation.sample(imitation_n, len(imitation), int(rng.integers(2**62)), True))
        learner.step(concat_samples(parts), targets=blend)
        schedule.step()
        if step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "seconds": round(time.monotonic() - started),
                "epochs": round(step * (args.batch_size - imitation_n) / len(replay), 2),
                "lr": schedule.get_last_lr()[0],
                **{k: round(v, 4) for k, v in learner.drain_metrics().items()},
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        if step % snapshot_every == 0 or step == args.steps:
            meta = {"step": step, "teacher": str(args.teacher), "replay": str(args.replay)}
            save(args.out / f"student-{step:06d}.pt", learner.ema, meta)
            save(args.out / "student.pt", learner.ema, meta)
    log.close()


if __name__ == "__main__":
    main()
