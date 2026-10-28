"""Fixed-work replay fitting and self-play round timing; all learned copies discarded."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .duo_deep import DeepConfig, atomic_json, fit_indexed
from .duo_learning import load_duo
from .duo_replay import pack_replay, replay_records
from .duo_shared import SharedSearchPool
from .duo_training_device import (
    metal_rng_state,
    prepare_training_device,
    synchronize_device,
)
from .learning import configure_torch

MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
STATE_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def hardware(device):
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "torch": str(torch.__version__), "numpy": np.__version__,
              "python": platform.python_version(), "device": device,
              "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
              "cpu_threads": torch.get_num_threads()}
    if device == "mps":
        result.update(
            processor=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        )
    if device == "cuda":
        result.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
    return result


def benchmark(args):
    if args.out.exists():
        raise ValueError("benchmark output exists; refuse accidental repeat")
    if args.variant == "production" and args.device != "cuda":
        raise ValueError("production CUDA-Graph variant requires CUDA")
    if any(n <= 0 for n in (args.games, args.repeats, args.optimizer_steps, args.workers)):
        raise ValueError("positive fixed workload required")
    if sha(args.model) != args.model_sha or sha(args.source_state) != args.state_sha:
        raise ValueError("benchmark input hash mismatch")
    configure_torch(1090)
    precision = "bfloat16" if args.variant == "production" else "float32"
    backend = "cuda_graphs" if args.variant == "production" else "eager"
    prepare_training_device(args.device, precision, backend)
    if args.device == "cuda":
        # The matched path forbids TF32; the production path retains it.
        torch.set_float32_matmul_precision("high" if args.variant == "production" else "highest")
        torch.backends.cudnn.allow_tf32 = args.variant == "production"
        torch.backends.cuda.matmul.allow_tf32 = args.variant == "production"
    config = DeepConfig(
        architecture="conv", device=args.device, precision=precision, seed=1090,
        rounds=1, games_per_round=args.games, workers=args.workers, live_games=64,
        full_simulations=args.full_simulations, fast_simulations=32,
        batch_wait_ms=0, replay_positions=args.replay_positions,
        round_timeout_seconds=min(args.deadline, 600),
    )
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > args.deadline:
            raise TimeoutError("fixed benchmark deadline reached; no automatic retry")

    try:
        saved = torch.load(args.source_state, map_location="cpu", weights_only=False)
        replay = replay_records(saved["replay"])
        if len(replay) != args.replay_positions:
            raise ValueError("benchmark must use exactly the declared replay window")
        del saved
        model = load_duo(args.model).to(args.device).eval()
        base = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        base_sha = tensor_sha(base)
        result = {
            "hardware": hardware(args.device), "variant": args.variant, "config": asdict(config),
            "model_sha256": args.model_sha, "source_state_sha256": args.state_sha,
            "initial_tensor_sha256": base_sha, "initial_replay_positions": len(replay),
            "parameters": sum(p.numel() for p in model.parameters()),
            "deterministic_algorithms": "warn_only" if args.device == "mps" else "strict",
            "backend": backend, "fit": [], "rounds": [],
            "source_sha256": {
                name: sha(importlib.util.find_spec("blokus_rl." + name).origin)
                for name in ("duo_deep", "duo_shared", "duo_mcts", "duo_conv", "duo_replay",
                             "duo_training_device", "duo_training_benchmark", "_native")
            },
            "exclusions": "round cycle excludes validation, Volume commits, initial load and worker startup",
        }
        atomic_json(args.out / "protocol.json", result)

        def reset():
            model.load_state_dict(base)
            model.eval()
            return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)

        # Discarded optimizer warm-up: same real replay, separate copy generation.
        optimizer = reset()
        fit_indexed(model, optimizer, replay, np.random.default_rng(1091), config, 2)
        synchronize_device(args.device)
        for repeat in range(args.repeats):
            deadline()
            optimizer = reset()
            rng = np.random.default_rng(1092)  # Identical row selections each repetition/backend.
            synchronize_device(args.device)
            tick = time.monotonic()
            losses = fit_indexed(model, optimizer, replay, rng, config, args.optimizer_steps)
            synchronize_device(args.device)
            seconds = time.monotonic() - tick
            steps = {int(value["step"].item()) for value in optimizer.state.values()}
            if steps != {args.optimizer_steps} or tensor_sha(model.state_dict()) == base_sha:
                raise RuntimeError("optimizer update accounting failed")
            row = {"repeat": repeat+1, "seconds": seconds, "steps": args.optimizer_steps,
                   "examples": args.optimizer_steps*min(config.minibatch_size, len(replay)), "losses": losses,
                   "final_tensor_sha256": tensor_sha(model.state_dict())}
            result["fit"].append(row)
            atomic_json(args.out / "progress.json", result)
            print(json.dumps({"stage":"fit", **row}), flush=True)

        reset()
        with SharedSearchPool(config, inference_backend=backend) as pool:
            _, warm_scores, warm_metrics = pool.collect(model, 0, max(12, args.workers))
            result["warmup"] = {"games": len(warm_scores), "metrics": warm_metrics,
                                "optimizer_steps": 2}
            for repeat in range(args.repeats):
                deadline()
                optimizer = reset()
                rng = np.random.default_rng(1093)
                synchronize_device(args.device)
                tick = time.monotonic()
                records, scores, metrics = pool.collect(model, repeat+1, args.games)
                full = len(records)
                sims = int(metrics["simulations"])
                residual = sims - full*config.full_simulations
                if len(scores) != args.games or residual < 0 or residual % 32:
                    raise RuntimeError("collection work accounting failed")
                if any(int(row["visits"].sum()) != config.full_simulations for row in records):
                    raise RuntimeError("recorded root search budget changed")
                window = deque(replay, maxlen=config.replay_positions)
                window.extend(records)
                steps = max(1, math.ceil(config.training_ratio * full / config.minibatch_size))
                synchronize_device(args.device)
                fit_tick = time.monotonic()
                losses = fit_indexed(model, optimizer, list(window), rng, config, steps)
                synchronize_device(args.device)
                fit_seconds = time.monotonic() - fit_tick
                with tempfile.TemporaryDirectory(prefix="blokus-bench-") as temporary:
                    checkpoint = Path(temporary) / "round.pt"
                    save_tick = time.monotonic()
                    bundle = {"model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "replay": pack_replay(window), "numpy_rng": rng.bit_generator.state,
                              "torch_rng": torch.get_rng_state(),
                              "mps_rng": metal_rng_state(args.device)}
                    torch.save(bundle, checkpoint)
                    save_seconds = time.monotonic() - save_tick
                    cycle_seconds = time.monotonic() - tick
                    size = checkpoint.stat().st_size
                    # Readback verification is excluded from the timed round cycle.
                    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    checkpoint_ok = (
                        tensor_sha(restored["model_state"]) == tensor_sha(model.state_dict())
                        and len(replay_records(restored["replay"])) == config.replay_positions
                        and {int(v["step"].item()) for v in restored["optimizer"]["state"].values()} == {steps}
                    )
                    if not checkpoint_ok:
                        raise RuntimeError("checkpoint round-trip failed")
                    del restored, bundle
                row = {"repeat":repeat+1, "games":len(scores), "recorded_roots":full,
                       "fast_roots":residual//32, "simulations":sims,
                       "collection_seconds":metrics["collect_seconds"], "metrics":metrics,
                       "fit_seconds":fit_seconds, "gradient_steps":steps,
                       "gradient_examples":steps*min(config.minibatch_size, len(window)), "losses":losses,
                       "checkpoint_seconds":save_seconds, "checkpoint_bytes":size,
                       "round_cycle_seconds":cycle_seconds, "checkpoint_readback_verified":True}
                result["rounds"].append(row)
                atomic_json(args.out / "progress.json", result)
                print(json.dumps({"stage":"round", **row}), flush=True)
        if sha(args.model) != args.model_sha or sha(args.source_state) != args.state_sha:
            raise RuntimeError("preserved benchmark inputs changed")
        result["elapsed_seconds"] = time.monotonic() - started
        result["process_peak_rss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["rss_units"] = "bytes" if platform.system() == "Darwin" else "KiB"
        if args.device == "mps":
            result["metal_memory_bytes"] = {
                "allocated": torch.mps.current_allocated_memory(),
                "driver": torch.mps.driver_allocated_memory(),
            }
        atomic_json(args.out / "finished.json", result)
        return result
    except BaseException:
        import traceback

        atomic_json(args.out / "failure.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--variant", choices=("eager", "production"), default="eager")
    parser.add_argument("--model-sha", default=MODEL_SHA)
    parser.add_argument("--state-sha", default=STATE_SHA)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--optimizer-steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--full-simulations", type=int, default=1024)
    parser.add_argument("--replay-positions", type=int, default=100000)
    parser.add_argument("--deadline", type=int, default=900)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
