"""v2 self-play training: Rust engine + replay, PyTorch losses, EMA weights.

One process does everything. The engine plays continuously with the EMA
network; between engine batches the trainer takes gradient steps so that each
recorded position is sampled about `train_ratio` times. Checkpoints hold the
weights, optimizer and the compact replay (a few bytes per move), so a run can
resume after interruption.

Targets per recorded position (all rebuilt by Rust under a random symmetry):
search visit distribution, win/draw/loss, final score margin, final owner of
each cell.
"""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .._native import DuoReplay
from . import engine as v2
from .model import MARGIN_SCALE, PLANES, SIZE, build, load, save


@dataclass
class TrainConfig:
    blocks: int = 6
    channels: int = 64
    # Self-play
    slots: int = 512
    games: int = 10_000_000
    full_simulations: int = 1024
    fast_simulations: int = 128
    full_probability: float = 0.25
    temperature_plies: int = 12
    leaf_batch: int = 4
    c_puct: float = 1.5
    fpu_reduction: float = 0.25
    noise_weight: float = 0.25
    solver_max_moves: int = 20
    solver_root_nodes: int = 100_000
    score_weight: float = 0.05
    # Learning
    replay_capacity: int = 2_000_000
    replay_min: int = 20_000
    window_fraction: float = 0.5
    window_min: int = 100_000
    train_ratio: float = 4.0
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 3e-5
    ema_decay: float = 0.995
    margin_weight: float = 0.5
    ownership_weight: float = 0.5
    imitation_fraction: float = 0.0
    # Run control
    device: str = field(default_factory=v2.default_device)
    precision: str = "float32"
    seed: int = 2026
    max_seconds: float = 3600.0
    checkpoint_seconds: float = 600.0
    log_seconds: float = 60.0


def window(replay, config):
    return max(config.window_min, int(config.window_fraction * len(replay)))


def concat_samples(parts):
    return {key: b"".join(bytes(p[key]) for p in parts) for key in parts[0]}


def to_tensors(sample, device):
    counts = np.frombuffer(sample["counts"], np.uint32).astype(np.int64)
    n = len(counts)
    rows = np.repeat(np.arange(n), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    cols = np.arange(len(rows)) - np.repeat(starts, counts)

    def t(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    return {
        "n": n,
        "width": int(counts.max()),
        "rows": t(rows),
        "cols": t(cols),
        "planes": t(np.frombuffer(sample["planes"], np.uint8).reshape(n, PLANES, SIZE, SIZE)).float(),
        "scalars": t(np.frombuffer(sample["scalars"], np.float32).reshape(n, -1)),
        "legal": t(np.frombuffer(sample["legal"], np.uint16).astype(np.int64)),
        "policy": t(np.frombuffer(sample["policy"], np.float32)),
        "outcome": t(np.frombuffer(sample["outcome"], np.uint8).astype(np.int64)),
        "margin": t(np.frombuffer(sample["margin"], np.float32)),
        "ownership": t(np.frombuffer(sample["ownership"], np.uint8).astype(np.int64).reshape(n, -1)),
    }


def losses(model, batch, config):
    policy, outcome, margin, owner = model(batch["planes"], batch["scalars"], ownership=True)
    n, rows, cols = batch["n"], batch["rows"], batch["cols"]
    padded = torch.full((n, batch["width"]), -1e9, device=policy.device)
    padded[rows, cols] = policy.float()[rows, batch["legal"]]
    log_p = torch.log_softmax(padded, 1)[rows, cols]
    target = batch["policy"]
    policy_loss = -(target * log_p).sum() / n
    entropy = -(target * torch.log(target.clamp_min(1e-12))).sum() / n
    value_loss = F.cross_entropy(outcome.float(), batch["outcome"])
    margin_loss = F.smooth_l1_loss(margin.float(), batch["margin"] / MARGIN_SCALE)
    ownership_loss = F.cross_entropy(owner.float().flatten(2), batch["ownership"])
    total = (
        policy_loss
        + value_loss
        + config.margin_weight * margin_loss
        + config.ownership_weight * ownership_loss
    )
    return total, {
        "policy": float(policy_loss.detach()),
        "policy_kl": float((policy_loss - entropy).detach()),
        "value": float(value_loss.detach()),
        "margin": float(margin_loss.detach()),
        "ownership": float(ownership_loss.detach()),
    }


class Learner:
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device).train()
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.steps = 0
        self.samples = 0
        self.metrics = []

    def autocast(self):
        if self.config.precision == "bfloat16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    def step(self, sample):
        batch = to_tensors(sample, self.device)
        with self.autocast():
            total, parts = losses(self.model, batch, self.config)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        self.update_ema()
        self.steps += 1
        self.samples += batch["n"]
        parts["grad_norm"] = float(norm)
        self.metrics.append(parts)

    @torch.no_grad()
    def update_ema(self):
        decay = min(self.config.ema_decay, (1 + self.steps) / (10 + self.steps))
        for e, p in zip(self.ema.parameters(), self.model.parameters()):
            e.lerp_(p.detach(), 1 - decay)
        for e, b in zip(self.ema.buffers(), self.model.buffers()):
            e.copy_(b)

    def drain_metrics(self):
        if not self.metrics:
            return {}
        keys = self.metrics[0].keys()
        out = {k: float(np.mean([m[k] for m in self.metrics])) for k in keys}
        self.metrics = []
        return out

    def state(self):
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "samples": self.samples,
        }

    def restore(self, state):
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps, self.samples = state["steps"], state["samples"]


def atomic_write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(data, (bytes, bytearray)):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data)
    os.replace(tmp, path)


def agent_config(config):
    return v2.self_play_agent(
        full_simulations=config.full_simulations,
        fast_simulations=config.fast_simulations,
        full_probability=config.full_probability,
        temperature_plies=config.temperature_plies,
        leaf_batch=config.leaf_batch,
        c_puct=config.c_puct,
        fpu_reduction=config.fpu_reduction,
        noise_weight=config.noise_weight,
        solver_max_moves=config.solver_max_moves,
        solver_root_nodes=config.solver_root_nodes,
        score_weight=config.score_weight,
    )


def save_checkpoint(out, learner, replay, stats, config):
    out = Path(out)
    atomic_write(out / "replay.bin", bytes(replay.to_bytes()))
    tmp = out / "state.pt.tmp"
    torch.save(
        {"config": asdict(config), "learner": learner.state(), "stats": stats}, tmp
    )
    os.replace(tmp, out / "state.pt")
    save(out / "latest.pt", learner.ema, {"steps": learner.steps, "games": stats["games"]})


def train(config, out, *, init=None, imitation=None, resume=False, commit=None):
    out = Path(out)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if resume:
        state = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        saved = TrainConfig(**state["config"])
        changed = {k for k, v in asdict(config).items() if getattr(saved, k) != v}
        # Schedules may change between resumed chunks; the game itself may not.
        adjustable = {
            "max_seconds", "checkpoint_seconds", "log_seconds", "games", "learning_rate",
            "imitation_fraction", "replay_capacity", "window_fraction", "window_min",
            "train_ratio", "temperature_plies", "full_simulations", "fast_simulations",
            "full_probability", "slots",
        }
        if changed - adjustable:
            raise ValueError(f"resume config changed: {sorted(changed - adjustable)}")
        if changed:
            print(json.dumps({"resume_changes": sorted(changed)}), flush=True)
        model = build({"blocks": config.blocks, "channels": config.channels, "context": 64})
        learner = Learner(config, model)
        learner.restore(state["learner"])
        for group in learner.optimizer.param_groups:
            group["lr"] = config.learning_rate
        replay = DuoReplay.from_bytes((out / "replay.bin").read_bytes())
        replay.capacity = config.replay_capacity
        stats = state["stats"]
    else:
        out.mkdir(parents=True, exist_ok=False)
        if init:
            model, _ = load(init)
            if model.config["blocks"] != config.blocks or model.config["channels"] != config.channels:
                raise ValueError("init checkpoint shape does not match config")
        else:
            model = build({"blocks": config.blocks, "channels": config.channels, "context": 64})
        learner = Learner(config, model)
        replay = DuoReplay(config.replay_capacity)
        stats = {"games": 0, "elapsed": 0.0}
        atomic_write(out / "config.json", json.dumps(asdict(config), indent=2))
    imitation_replay = DuoReplay.from_bytes(Path(imitation).read_bytes()) if imitation else None
    if config.imitation_fraction and imitation_replay is None:
        raise ValueError("imitation_fraction requires --imitation")
    learner.model.to(device)
    learner.ema.to(device)
    for state in learner.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    agent = agent_config(config)
    engine = v2.make_engine(
        [agent, agent], config.slots, config.games, seed=config.seed + stats["games"]
    )
    evaluator = v2.Evaluator(
        learner.ema, config.device, precision=config.precision, score_weight=config.score_weight
    )
    started = time.monotonic()
    base_elapsed = stats["elapsed"]
    last_checkpoint = last_log = started
    window_stats = {"games": 0, "length": 0, "margin": 0.0, "first_wins": 0, "solved": 0}
    rng = np.random.default_rng(config.seed + learner.steps)
    log = (out / "progress.jsonl").open("a")

    def on_finished(games):
        for g in games:
            stats["games"] += 1
            window_stats["games"] += 1
            window_stats["length"] += len(g["moves"])
            margin = g["scores"][0] - g["scores"][1]
            window_stats["margin"] += abs(margin)
            window_stats["first_wins"] += margin > 0
            window_stats["solved"] += g["solved_records"]

    def poll():
        nonlocal last_checkpoint, last_log
        now = time.monotonic()
        target = config.train_ratio * replay.added_positions
        imitation_n = (
            int(round(config.imitation_fraction * config.batch_size)) if imitation_replay else 0
        )
        own_n = config.batch_size - imitation_n
        while len(replay) >= config.replay_min and learner.samples * own_n / config.batch_size < target:
            parts = [replay.sample(own_n, window(replay, config), int(rng.integers(2**62)), True)]
            if imitation_n:
                parts.append(
                    imitation_replay.sample(
                        imitation_n, len(imitation_replay), int(rng.integers(2**62)), True
                    )
                )
            learner.step(concat_samples(parts))
        elapsed = base_elapsed + now - started
        if now - last_log >= config.log_seconds:
            g = max(window_stats["games"], 1)
            row = {
                "elapsed": round(elapsed),
                "games": stats["games"],
                "replay": len(replay),
                "positions": replay.added_positions,
                "steps": learner.steps,
                "games_per_second": round(window_stats["games"] / (now - last_log), 3),
                "mean_length": round(window_stats["length"] / g, 2),
                "mean_abs_margin": round(window_stats["margin"] / g, 2),
                "first_player_win_rate": round(window_stats["first_wins"] / g, 3),
                "solved_records_per_game": round(window_stats["solved"] / g, 2),
                "evaluations_per_second": round(evaluator.positions / max(now - started, 1e-9)),
                **{k: round(v, 4) for k, v in learner.drain_metrics().items()},
                **engine.counters(),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            for k in window_stats:
                window_stats[k] = 0
            last_log = now
        if now - last_checkpoint >= config.checkpoint_seconds:
            stats["elapsed"] = elapsed
            save_checkpoint(out, learner, replay, stats, config)
            save(out / f"ema-{stats['games']:08d}.pt", learner.ema, {"steps": learner.steps})
            last_checkpoint = now
            if commit:
                commit()
        return elapsed >= config.max_seconds

    runner = v2.run_threaded if device.type == "cuda" else v2.run
    runner(engine, evaluator, on_finished=on_finished, replay=replay, poll=poll)
    stats["elapsed"] = base_elapsed + time.monotonic() - started
    save_checkpoint(out, learner, replay, stats, config)
    save(out / f"ema-{stats['games']:08d}.pt", learner.ema, {"steps": learner.steps})
    if commit:
        commit()
    log.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--imitation", type=Path)
    parser.add_argument("--resume", action="store_true")
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    config = TrainConfig(**{k: getattr(args, k) for k in asdict(defaults)})
    stats = train(config, args.out, init=args.init, imitation=args.imitation, resume=args.resume)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
