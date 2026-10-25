"""AlphaZero-style Blokus Duo: live search, self-play, visit targets and final results.

This is a small local implementation of the learning loop, not a reproduction
of DeepMind's network, search scale or results. No heuristic labels are used.
"""

import argparse
import hashlib
import json
import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ._native import DuoGame
from .duo_crossplay import evaluate_pair, opening_book
from .duo_learning import load_duo, pack, save_duo
from .duo_mcts import search_games
from .duo_model import DuoActorCritic
from .learning import configure_torch


@dataclass(frozen=True)
class AzConfig:
    seed: int = 642
    rounds: int = 20
    games_per_round: int = 64
    simulations: int = 128
    workers: int = 3
    live_games: int = 8
    leaf_batch: int = 8
    replay_positions: int = 20000
    gradient_steps: int = 100
    minibatch_size: int = 128
    learning_rate: float = 0.0003
    eval_every: int = 5
    validation_openings: int = 24
    validation_seed: int = 2543000
    test_seed: int = 2549000
    device: str = "cpu"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in ("device",) and value <= 0:
                raise ValueError(f"{key} must be positive")
        if self.device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")


def collect_worker(task):
    checkpoint, config, seed, count = task
    configure_torch(seed)
    model = load_duo(checkpoint)
    records, scores, metrics = [], [], {}
    for offset in range(0, count, config.live_games):
        n = min(config.live_games, count - offset)
        games = [DuoGame() for _ in range(n)]
        rngs = [np.random.default_rng(seed + offset + i) for i in range(n)]
        histories = [[] for _ in range(n)]
        plies = [0] * n
        active = list(range(n))
        while active:
            results = search_games(
                model,
                [games[i] for i in active],
                config.simulations,
                config.leaf_batch,
                rngs=[rngs[i] for i in active],
                metrics=metrics,
            )
            for i, result in zip(active, results, strict=True):
                player = games[i].current_player
                histories[i].append(
                    {
                        "observation": result.observation,
                        "candidates": result.candidates,
                        "policy_target": result.policy,
                        "player": player,
                    }
                )
                if plies[i] < 10:
                    action = int(
                        rngs[i].choice(
                            result.actions, p=result.policy.astype(float) / result.policy.sum()
                        )
                    )
                else:
                    action = result.action()
                games[i].step(action)
                plies[i] += 1
                if games[i].terminated:
                    score = games[i].scores
                    winner = int(np.sign(score[0] - score[1]))
                    for record in histories[i]:
                        record["value_target"] = float(
                            winner if record.pop("player") == 0 else -winner
                        )
                    records.extend(histories[i])
                    scores.append(score)
            active = [i for i in active if not games[i].terminated]
    return records, scores, metrics


def fit(model, optimizer, replay, rng, config):
    model.train()
    device = next(model.parameters()).device
    totals = np.zeros(3)
    for _ in range(config.gradient_steps):
        indices = rng.choice(len(replay), min(config.minibatch_size, len(replay)), replace=False)
        rows = [replay[int(i)] for i in indices]
        batch = pack(rows).to(device)
        target = torch.zeros(batch["action_mask"].shape, device=device)
        for i, row in enumerate(rows):
            target[i, : len(row["policy_target"])] = torch.from_numpy(row["policy_target"]).to(
                device
            )
        logits = model.logits(batch["observation"], batch["candidates"])
        logs = F.log_softmax(logits.masked_fill(~batch["action_mask"], -1e9), dim=-1)
        policy_loss = -(target * logs).sum(-1).mean()
        values = model(batch["observation"])[1]
        value_loss = F.mse_loss(values, batch["value_target"])
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite self-play training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        totals += [float(policy_loss.detach()), float(value_loss.detach()), float(norm)]
    model.eval()
    return dict(
        zip(
            ("policy_loss", "value_loss", "gradient_norm"),
            (totals / config.gradient_steps).tolist(),
            strict=True,
        )
    )


def train(config, out, initial=None, resume=False):
    out = Path(out)
    configure_torch(config.seed)
    if resume:
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["config"] != asdict(config):
            raise ValueError("resume config must match exactly")
        model = load_duo(out / "initial.pt")
        model.load_state_dict(saved["model_state"])
        replay = deque(saved["replay"], maxlen=config.replay_positions)
        start_round, best = saved["round"] + 1, saved["best"]
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
    else:
        out.mkdir(parents=True, exist_ok=False)
        model = load_duo(initial) if initial else DuoActorCritic()
        replay = deque(maxlen=config.replay_positions)
        rng = np.random.default_rng(config.seed)
        start_round, best = 1, 0.5
        protocol = {
            "config": asdict(config),
            "initial": str(initial) if initial else None,
            "initial_sha256": hashlib.sha256(Path(initial).read_bytes()).hexdigest()
            if initial
            else None,
            "training": "current network plays both seats with noisy PUCT; visit cross entropy + terminal WDL MSE",
            "selection": "raw greedy vs frozen initial network on validation openings",
            "test": "reserved seed; never used for selection",
        }
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        save_duo(out / "initial.pt", model, config, 0, metadata={"algorithm": "alphazero_style"})
        save_duo(out / "best.pt", model, config, 0)
    reference = load_duo(out / "initial.pt")
    book = opening_book(config.validation_openings, seed=config.validation_seed)
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
    # CPU workers each use one core; independent deterministic seeds per round.
    with ProcessPoolExecutor(
        max_workers=config.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for round_ in range(start_round, config.rounds + 1):
            start = time.monotonic()
            model.cpu()
            save_duo(out / "latest.pt", model, config, round_ - 1)
            counts = [
                config.games_per_round // config.workers
                + int(i < config.games_per_round % config.workers)
                for i in range(config.workers)
            ]
            tasks = [
                (out / "latest.pt", config, config.seed + round_ * 100000 + i * 1000, n)
                for i, n in enumerate(counts)
                if n
            ]
            metrics, scores = {}, []
            for rows, results, counters in pool.map(collect_worker, tasks):
                replay.extend(rows)
                scores.extend(results)
                for key, value in counters.items():
                    metrics[key] = metrics.get(key, 0) + value
            collect_seconds = time.monotonic() - start
            model.to(config.device)
            losses = fit(model, optimizer, list(replay), rng, config)
            model.cpu()
            row = {
                "round": round_,
                "games": round_ * config.games_per_round,
                "replay_positions": len(replay),
                "collect_seconds": collect_seconds,
                "metrics": metrics,
                "losses": losses,
                "mean_remaining_score": float(np.mean(scores)),
            }
            if round_ % config.eval_every == 0 or round_ == config.rounds:
                result = evaluate_pair(model, reference, book, config.validation_seed)
                row["validation"] = result
                if result["score_rate"] > best:
                    best = result["score_rate"]
                    save_duo(
                        out / "best.pt",
                        model,
                        config,
                        round_,
                        metadata={"algorithm": "alphazero_style"},
                    )
            save_duo(
                out / "latest.pt", model, config, round_, metadata={"algorithm": "alphazero_style"}
            )
            save_duo(out / f"round-{round_:03d}.pt", model, config, round_)
            # Atomic complete round checkpoint includes optimizer, replay and RNG.
            torch.save(
                {
                    "config": asdict(config),
                    "round": round_,
                    "best": best,
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "replay": list(replay),
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                },
                out / "resume.tmp",
            )
            (out / "resume.tmp").replace(out / "resume.pt")
            row["seconds"] = time.monotonic() - start
            with (out / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("validation", "metrics")}
                    | {"validation_score": row.get("validation", {}).get("score_rate")}
                ),
                flush=True,
            )
            model.to(config.device)
    model.cpu()
    save_duo(
        out / "final.pt", model, config, config.rounds, metadata={"algorithm": "alphazero_style"}
    )
    (out / "completed.json").write_text(
        json.dumps(
            {
                "rounds": config.rounds,
                "games": config.rounds * config.games_per_round,
                "next": "evaluate best.pt using reserved test openings, raw and with search",
            }
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--resume", action="store_true")
    defaults = asdict(AzConfig())
    for name, value in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    config = AzConfig(**{key: getattr(args, key) for key in defaults})
    train(config, args.out, args.init, args.resume)


if __name__ == "__main__":
    main()
