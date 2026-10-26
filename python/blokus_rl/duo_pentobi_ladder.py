"""Frozen-policy benchmark against all Pentobi levels; parallel, atomic whole pairs."""

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .duo_crossplay import opening_book, paired_interval
from .duo_deep import atomic_json
from .duo_deep_evaluate import IndexedSearchPolicy
from .duo_learning import load_duo
from .duo_pentobi import DEFAULT_BINARY, Pentobi, match
from .learning import configure_torch

WORKER = {}
BEST_SHA256 = "0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86"


@dataclass(frozen=True)
class LadderConfig:
    opening_pairs: int = 100
    seed: int = 2579000
    simulations: int = 128
    workers: int = 3
    command_timeout: int = 600
    levels: tuple[int, ...] = tuple(range(1, 10))

    def __post_init__(self):
        if (
            min(self.opening_pairs, self.seed, self.simulations, self.workers, self.command_timeout)
            < 1
        ):
            raise ValueError("positive benchmark sizes/budgets required")
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("unique Pentobi levels required")
        if any(level not in range(1, 10) for level in self.levels):
            raise ValueError("Pentobi supports levels 1 through 9")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize_worker(checkpoint, binary, configuration):
    config = LadderConfig(**configuration)
    configure_torch(config.seed)
    WORKER.update(model=load_duo(checkpoint), binary=binary, config=config)


def play_pair(job):
    level, index, opening = job
    config = WORKER["config"]
    games = []
    with Pentobi(
        binary=WORKER["binary"], level=level, seed=config.seed, timeout=config.command_timeout
    ) as external:
        for seat in (0, 1):
            policy = IndexedSearchPolicy(WORKER["model"], config.simulations, "float32")
            started = time.monotonic()
            row = match(external, policy, seat, config.seed + index + seat * 1000000, opening)
            row |= {"seconds": time.monotonic() - started, "policy_metrics": policy.metrics}
            games.append(row)
    return {"level": level, "index": index, "games": games}


def summarize(records, expected_pairs):
    records = sorted(records, key=lambda r: r["index"])
    details = [game for row in records for game in row["games"]]
    pairs = np.asarray(
        [
            [1.0 if g["margin"] > 0 else 0.5 if g["margin"] == 0 else 0.0 for g in row["games"]]
            for row in records
        ]
    )
    metrics = {}
    for row in details:
        for key, value in row["policy_metrics"].items():
            metrics[key] = metrics.get(key, 0) + value
    return {
        "complete": len(records) == expected_pairs,
        "opening_pairs": len(records),
        "games": len(details),
        "wins": int(np.sum(pairs == 1)),
        "draws": int(np.sum(pairs == 0.5)),
        "losses": int(np.sum(pairs == 0)),
        "score_rate": float(pairs.mean()),
        "score_rate_95pct_cluster_bootstrap": paired_interval(pairs),
        "zero_score_one_sided_95_upper": -math.expm1(math.log(0.05) / len(pairs))
        if not pairs.any()
        else None,
        "opening_pair_indices": [r["index"] for r in records],
        "opening_pair_scores": pairs.tolist(),
        "mean_margin": float(np.mean([r["margin"] for r in details])),
        "sum_game_seconds": sum(r["seconds"] for r in details),
        "policy_metrics": metrics,
    }


def run(config, checkpoint, binary, out, *, resume=False, commit=None, lock_path=None):
    checkpoint, binary, out = Path(checkpoint), Path(binary), Path(out)
    if out.exists() and not resume:
        raise ValueError("use an unused output directory or explicitly resume")
    book = opening_book(config.opening_pairs, seed=config.seed)
    protocol = {
        "config": asdict(config),
        "checkpoint_sha256": digest(checkpoint),
        "pentobi_binary_sha256": digest(binary),
        "pentobi_version": "31.0",
        "pentobi_commit": "d5374ddc20eced902788e733d67ee16bd1836fd3",
        "openings": book,
        "purpose": "fixed-policy benchmark; no training, checkpoint selection or early stopping on scores",
        "precision": "CPU float32, one PyTorch thread per worker",
        "opponent": "Duo; native level settings; threads1; no opening book/resignation; default symmetric-draw setting",
        "uncertainty": "whole opening-pair bootstrap, nominal 95%; all-zero scores also have a conservative one-sided paired upper bound",
        "reproducibility": "seed per game and independent engine per pair; cross-host floating point identity not promised",
        "torch_version": str(torch.__version__),
    }
    protocol = json.loads(json.dumps(protocol))
    out.mkdir(parents=True, exist_ok=True)
    with Path(lock_path or out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "protocol.json").exists():
            if json.loads((out / "protocol.json").read_text()) != protocol:
                raise ValueError("benchmark protocol/model/binary changed")
        else:
            (out / "model.pt").write_bytes(checkpoint.read_bytes())
            atomic_json(out / "protocol.json", protocol)
        if digest(out / "model.pt") != protocol["checkpoint_sha256"]:
            raise ValueError("frozen benchmark model changed")
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        records = {level: {} for level in config.levels}
        for level in config.levels:
            directory = out / f"level-{level:02}"
            directory.mkdir(exist_ok=True)
            for path in directory.glob("pair-*.json"):
                row = json.loads(path.read_text())
                index = row["index"]
                if (
                    row["protocol_sha256"] != fingerprint
                    or row["level"] != level
                    or not 0 <= index < len(book)
                    or index in records[level]
                    or [g["seat"] for g in row["games"]] != [0, 1]
                    or [g["seed"] for g in row["games"]]
                    != [config.seed + index, config.seed + index + 1000000]
                ):
                    raise ValueError("invalid cached opening pair")
                records[level][index] = row
        # Round-robin levels: each gets coverage before a whole level completes.
        jobs = [
            (level, index, opening)
            for index, opening in enumerate(book)
            for level in config.levels
            if index not in records[level]
        ]
        iterator = iter(jobs)
        started = last_commit = time.monotonic()
        if commit:
            commit()
        if jobs:
            with ProcessPoolExecutor(
                max_workers=config.workers,
                mp_context=mp.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(str((out / "model.pt").resolve()), str(binary.resolve()), asdict(config)),
            ) as pool:
                pending = {}

                def fill():
                    while len(pending) < config.workers:
                        job = next(iterator, None)
                        if job is None:
                            break
                        pending[pool.submit(play_pair, job)] = job[:2]

                fill()
                while pending:
                    ready, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in ready:
                        level, index = pending.pop(future)
                        row = future.result()
                        if (row["level"], row["index"]) != (level, index):
                            raise RuntimeError("worker result routing mismatch")
                        row["protocol_sha256"] = fingerprint
                        atomic_json(out / f"level-{level:02}/pair-{index:03}.json", row)
                        records[level][index] = row
                        count = len(records[level])
                        if count == 1 or count % 10 == 0 or count == config.opening_pairs:
                            atomic_json(
                                out / f"level-{level:02}/summary.json",
                                summarize(list(records[level].values()), config.opening_pairs),
                            )
                            print(
                                json.dumps(
                                    {"level": level, "pairs": count, "total": config.opening_pairs}
                                ),
                                flush=True,
                            )
                    status = {
                        "pairs_done": {str(level): len(rows) for level, rows in records.items()},
                        "pairs_total": config.opening_pairs,
                        "seconds_this_call": time.monotonic() - started,
                    }
                    atomic_json(out / "status.json", status)
                    if commit and time.monotonic() - last_commit >= 45:
                        commit()
                        last_commit = time.monotonic()
                    fill()
        results = {
            str(level): summarize(list(rows.values()), config.opening_pairs)
            for level, rows in records.items()
        }
        atomic_json(out / "results.json", results)
        atomic_json(
            out / "completed.json",
            {
                "protocol_sha256": fingerprint,
                "levels": list(results),
                "games": sum(r["games"] for r in results.values()),
            },
        )
        if commit:
            commit()
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/duo-deep-local-764/conv.pt"))
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--openings", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2579000)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        LadderConfig(args.openings, args.seed, args.simulations, args.workers),
        args.checkpoint,
        args.binary,
        args.out,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
