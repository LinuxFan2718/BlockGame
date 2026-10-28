"""Frozen-policy collection throughput only: no optimizer updates or strength test."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path


def run(args):
    import blokus_rl.duo_mcts as search_implementation
    import blokus_rl.duo_shared as implementation
    import numpy as np
    import torch
    from blokus_rl.duo_deep import DeepConfig
    from blokus_rl.duo_learning import load_duo
    from blokus_rl.duo_shared import SharedSearchPool
    from blokus_rl.learning import configure_torch

    configure_torch(1067)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    config = DeepConfig(
        architecture="conv",
        seed=1067,
        rounds=1,
        games_per_round=args.games,
        workers=args.workers,
        live_games=args.live_games,
        full_simulations=args.full_simulations,
        fast_simulations=32,
        device=args.device,
        precision="bfloat16" if args.device == "cuda" else "float32",
    )
    model = load_duo(args.model).to(args.device).eval()
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    options = (
        {}
        if args.variant == "baseline"
        else {
            "transport": "queue" if args.variant == "lookup-queue" else "shared_memory",
            "cache_entries": 8192 if args.variant == "shared-cache" else 0,
        }
    )
    started = time.monotonic()
    with SharedSearchPool(config, **options) as pool:
        # Same frozen weights; warm worker imports, native catalog and CUDA first.
        warmup_games = max(args.workers, min(args.games, 12))
        _, _, warmup_metrics = pool.collect(model, 0, warmup_games)
        records, scores, metrics = pool.collect(model, args.round, args.games)
    content = hashlib.sha256()
    for row in records:
        for key, dtype in (
            ("observation", np.float32),
            ("indices", np.uint16),
            ("visits", np.uint16),
        ):
            content.update(np.asarray(row[key], dtype=dtype).tobytes())
        content.update(np.asarray([row["value_target"]], dtype=np.float64).tobytes())
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != before:
        raise RuntimeError("benchmark modified frozen checkpoint")
    if len(scores) != args.games or any(
        int(row["visits"].sum()) != config.full_simulations for row in records
    ):
        raise RuntimeError("benchmark game/search accounting disagrees")
    result = {
        "variant": args.variant,
        "repeat_round": args.round,
        "config": asdict(config),
        "model_sha256": before,
        "parameters": sum(p.numel() for p in model.parameters()),
        "games": len(scores),
        "recorded_roots": len(records),
        "metrics": metrics,
        "collection_seconds": metrics["collect_seconds"],
        "total_seconds_including_warmup": time.monotonic() - started,
        "record_sha256": content.hexdigest(),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "gradient_updates": 0,
        "warmup_games": warmup_games,
        "warmup_metrics": warmup_metrics,
        "implementation_sha256": {
            "duo_shared": hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
            "duo_mcts": hashlib.sha256(
                Path(search_implementation.__file__).read_bytes()
            ).hexdigest(),
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "variant",
                    "repeat_round",
                    "games",
                    "collection_seconds",
                    "gradient_updates",
                )
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("baseline", "lookup-queue", "shared", "shared-cache"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--games", type=int, default=192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--live-games", type=int, default=32)
    parser.add_argument("--full-simulations", type=int, default=1024)
    parser.add_argument("--round", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
