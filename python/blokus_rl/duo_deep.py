"""Bounded cold-start residual-CNN self-play with shared batched inference."""

import argparse
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .duo_conv import DuoConvActorCritic
from .duo_crossplay import opening_book
from .duo_deep_evaluate import evaluate_many
from .duo_learning import load_duo, save_duo
from .duo_replay import pack_replay, replay_records
from .duo_residual import DuoResidualActorCritic
from .duo_shared import SharedSearchPool, pack_indexed, precision_context
from .learning import configure_torch


@dataclass(frozen=True)
class DeepConfig:
    seed: int = 764
    architecture: str = "residual"
    channels: int = 64
    blocks: int = 10
    hidden_size: int = 128
    rounds: int = 128
    games_per_round: int = 256
    workers: int = 6
    live_games: int = 32
    fast_probability: float = 0.8
    fast_simulations: int = 32
    full_simulations: int = 256
    temperature_plies: int = 12
    leaf_batch: int = 8
    inference_batch: int = 256
    batch_wait_ms: float = 1.0
    replay_positions: int = 100000
    training_ratio: float = 3.0
    minibatch_size: int = 256
    learning_rate: float = 0.001
    policy_loss_weight: float = 0.5
    eval_every: int = 8
    validation_openings: int = 24
    validation_seed: int = 2563000
    test_seed: int = 2569000
    validation_simulations: int = 128
    max_seconds: float = 9000
    round_timeout_seconds: float = 1200
    device: str = "cuda"
    precision: str = "bfloat16"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and key not in ("fast_probability", "batch_wait_ms"):
                if value <= 0:
                    raise ValueError(f"{key} must be positive")
        if not 0 <= self.fast_probability < 1 or self.batch_wait_ms < 0:
            raise ValueError("invalid fast-search probability or batching wait")
        if self.full_simulations > 65535 or self.fast_simulations > 65535:
            raise ValueError("visit counts must fit unsigned16 replay storage")
        if self.architecture not in ("residual", "conv"):
            raise ValueError("expected residual or conv architecture")
        if self.device not in ("cpu", "cuda") or self.precision not in ("float32", "bfloat16"):
            raise ValueError("unsupported device/precision")
        if self.channels % 8 or self.channels < 8:
            raise ValueError("channels must be divisible by eight")


def fit_indexed(model, optimizer, replay, rng, config, steps):
    model.train()
    device = next(model.parameters()).device
    totals = np.zeros(3)
    for _ in range(steps):
        selected = rng.choice(len(replay), min(config.minibatch_size, len(replay)), replace=False)
        rows = [replay[int(i)] for i in selected]
        observation, indices, mask = pack_indexed(rows, device)
        targets = np.zeros(tuple(mask.shape), dtype=np.float32)
        for i, row in enumerate(rows):
            visits = row["visits"].astype(np.float32)
            if visits.sum() <= 0:
                raise ValueError("empty replay visit target")
            targets[i, : len(visits)] = visits / visits.sum()
        target_policy = torch.from_numpy(targets).to(device)
        target_value = torch.tensor([r["value_target"] for r in rows], device=device)
        with precision_context(model, config.precision):
            scores, values = model(observation)
            logits = scores.gather(-1, indices).float().masked_fill(~mask, -1e9)
            policy_loss = -(target_policy * F.log_softmax(logits, -1)).sum(-1).mean()
            value_loss = F.mse_loss(values.flatten().float(), target_value)
            loss = config.policy_loss_weight * policy_loss + value_loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite residual self-play loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        totals += [float(policy_loss.detach()), float(value_loss.detach()), float(norm)]
    model.eval()
    return dict(zip(("policy_loss", "value_loss", "gradient_norm"), (totals / steps).tolist()))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def train(
    config,
    out,
    reference,
    *,
    resume=False,
    commit=None,
    stop_after_round=None,
    inference_transport="shared_memory",
    inference_cache_entries=0,
):
    out, reference = Path(out), Path(reference)
    runtime = {
        "inference_transport": inference_transport,
        "inference_cache_entries": inference_cache_entries,
    }
    configure_torch(config.seed)
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_float32_matmul_precision("high")
    if resume:
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["config"] != asdict(config):
            raise ValueError("resume configuration must match exactly")
        if saved.get("runtime", runtime) != runtime:
            raise ValueError("resume inference runtime must match exactly")
        if saved["reference_sha256"] != hashlib.sha256(reference.read_bytes()).hexdigest():
            raise ValueError("resume reference changed")
        model = load_duo(out / "latest.pt").to(config.device)
        model.load_state_dict(saved["model_state"])
        replay = deque(replay_records(saved["replay"]), maxlen=config.replay_positions)
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["numpy_rng"]
        start_round, best, games = saved["round"] + 1, saved["best"], saved["games"]
        previous_seconds = saved["elapsed_seconds"]
        torch.save(saved["best_checkpoint"], out / "best.tmp")
        (out / "best.tmp").replace(out / "best.pt")
    else:
        out.mkdir(parents=True, exist_ok=False)
        model = (
            DuoResidualActorCritic(config.hidden_size, config.channels, config.blocks)
            if config.architecture == "residual"
            else DuoConvActorCritic(config.hidden_size)
        ).to(config.device)
        rng = np.random.default_rng(config.seed)
        replay = deque(maxlen=config.replay_positions)
        start_round, best, games, previous_seconds = 1, -1.0, 0, 0.0
        atomic_json(
            out / "protocol.json",
            {
                "config": asdict(config),
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "training": "random neural initialization; current network plays both seats; no expert labels; full-search visit CE + terminal WDL MSE",
                "selection": "searched validation versus frozen original frontier342, equal simulations; fresh held-out openings never used for selection",
                "reproducibility": "independent game RNGs, fixed worker-order replay; async floating point batches may vary across scheduling/hardware",
                "parameters": sum(p.numel() for p in model.parameters()),
            },
        )
        save_duo(out / "initial.pt", model, config, 0)
        save_duo(out / "latest.pt", model, config, 0)
        save_duo(out / "best.pt", model, config, 0)
    opponent = load_duo(reference).to(config.device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved  # The deque owns retained records; do not pin a stale full replay as it evolves.
    book = opening_book(config.validation_openings, seed=config.validation_seed)
    started = time.monotonic()
    last_round_seconds, last_round = 0.0, start_round - 1
    reason = "round_limit"
    with SharedSearchPool(
        config, transport=inference_transport, cache_entries=inference_cache_entries
    ) as pool:
        for round_ in range(start_round, config.rounds + 1):
            elapsed = previous_seconds + time.monotonic() - started
            if round_ > 1 and elapsed + last_round_seconds >= config.max_seconds:
                reason = "time_budget"
                break
            round_start = time.monotonic()
            records, scores, metrics = pool.collect(model, round_, config.games_per_round)
            if not records:
                raise RuntimeError("no full-search training examples in the round")
            replay.extend(records)
            games += len(scores)
            steps = max(1, math.ceil(config.training_ratio * len(records) / config.minibatch_size))
            fit_start = time.monotonic()
            losses = fit_indexed(model, optimizer, list(replay), rng, config, steps)
            row = {
                "round": round_,
                "games": games,
                "new_training_positions": len(records),
                "replay_positions": len(replay),
                "gradient_steps": steps,
                "gradient_examples": steps * min(config.minibatch_size, len(replay)),
                "training_seconds": time.monotonic() - fit_start,
                "losses": losses,
                "metrics": metrics,
                "mean_terminal_score": float(np.mean(scores)),
            }
            if round_ % config.eval_every == 0 or round_ == config.rounds:
                result = evaluate_many(
                    model,
                    opponent,
                    book,
                    simulations=config.validation_simulations,
                    opponent_simulations=config.validation_simulations,
                    seed=config.validation_seed,
                    precision=config.precision,
                )
                row["validation"] = result
                if result["score_rate"] > best:
                    best = result["score_rate"]
                    save_duo(out / "best.pt", model, config, round_)
            save_duo(out / "latest.pt", model, config, round_)
            if round_ % config.eval_every == 0 or round_ == 1:
                save_duo(out / f"round-{round_:03d}.pt", model, config, round_)
            elapsed = previous_seconds + time.monotonic() - started
            torch.save(
                {
                    "config": asdict(config),
                    "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                    "round": round_,
                    "games": games,
                    "best": best,
                    "elapsed_seconds": elapsed,
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "replay": pack_replay(replay),
                    "resume_format_version": 2,
                    "runtime": runtime,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
                    "best_checkpoint": torch.load(
                        out / "best.pt", map_location="cpu", weights_only=True
                    ),
                },
                out / "resume.tmp",
            )
            (out / "resume.tmp").replace(out / "resume.pt")
            last_round_seconds = time.monotonic() - round_start
            row["round_seconds"], row["elapsed_seconds"] = last_round_seconds, elapsed
            with (out / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            atomic_json(out / "status.json", row)
            if commit:
                commit()
            last_round = round_
            print(
                json.dumps(
                    {
                        "round": round_,
                        "games": games,
                        "seconds": last_round_seconds,
                        "positions": len(records),
                        "mean_batch": metrics["mean_gpu_batch"],
                        "validation": row.get("validation", {}).get("score_rate"),
                        **losses,
                    }
                ),
                flush=True,
            )
            if stop_after_round and round_ >= stop_after_round:
                # Test/failure-injection hook: leave a resumable, incomplete run.
                return
    if best < 0 or (reason == "time_budget" and last_round % config.eval_every):
        # A short time-limited run may finish before its first scheduled validation.
        result = evaluate_many(
            model,
            opponent,
            book,
            simulations=config.validation_simulations,
            opponent_simulations=config.validation_simulations,
            seed=config.validation_seed,
            precision=config.precision,
        )
        if result["score_rate"] > best:
            best = result["score_rate"]
            save_duo(out / "best.pt", model, config, last_round)
        atomic_json(out / "final-validation.json", result)
    save_duo(out / "final.pt", model, config, last_round)
    atomic_json(
        out / "completed.json",
        {
            "rounds": last_round,
            "games": games,
            "reason": reason,
            "elapsed_seconds": previous_seconds + time.monotonic() - started,
            "best_validation_score": best,
        },
    )
    if commit:
        commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--inference-transport", choices=("queue", "shared_memory"), default="shared_memory"
    )
    parser.add_argument("--inference-cache-entries", type=int, default=0)
    defaults = asdict(DeepConfig())
    for name, value in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    config = DeepConfig(**{key: getattr(args, key) for key in defaults})
    train(
        config,
        args.out,
        args.reference,
        resume=args.resume,
        inference_transport=args.inference_transport,
        inference_cache_entries=args.inference_cache_entries,
    )


if __name__ == "__main__":
    main()
