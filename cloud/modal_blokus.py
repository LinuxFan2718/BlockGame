"""Small, explicitly bounded Modal benchmark; no persistent deployments.

Uses only an allowlisted source archive and one checkpoint argument. Authentication
stays in Modal's local SDK configuration. No home files or entire runs/ upload.
"""

import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP = "blokus-rl-hardware-benchmark"
MAX_SECONDS = 300
CPU_CORES = 4
MEMORY_GIB = 8
# Public list prices checked 2026-10-25; CPU/memory are added separately.
RATES = {"l4": 0.000222, "h100": 0.001097}


def source_archive():
    paths = [
        ROOT / name
        for name in (
            "Cargo.toml",
            "Cargo.lock",
            "pyproject.toml",
            "README.md",
            "bindings/python/Cargo.toml",
            "bindings/python/Cargo.lock",
        )
    ]
    paths += sorted((ROOT / "src").glob("*.rs"))
    paths += sorted((ROOT / "bindings/python/src").glob("*.rs"))
    paths += sorted((ROOT / "python/blokus_rl").glob("*.py"))
    destination = ROOT / ".local/modal-source.tar.gz"
    destination.parent.mkdir(exist_ok=True)
    # Normalize tar metadata; only these explicit project files can leave the Mac.
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in paths:
                    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
                        raise ValueError("refusing source paths outside the project allowlist")
                    data = path.read_bytes()
                    info = tarfile.TarInfo(str(path.relative_to(ROOT)))
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
    return destination


if modal.is_local():
    archive = source_archive()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "curl", "ca-certificates")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.91.1"
        )
        .env(
            {
                "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONPATH": "/opt/blokus/cloud",
            }
        )
        .pip_install(
            "torch==2.14.1",
            "torchrl==0.14.1",
            "tensordict==0.14.3",
            "numpy==2.5.4",
            "pettingzoo==1.27.0",
            "gymnasium==1.4.0",
            "matplotlib==3.11.2",
            "maturin>=1.9,<2",
        )
        .add_local_file(archive, "/tmp/blokus-source.tar.gz", copy=True)
        .run_commands(
            "mkdir -p /opt/blokus && tar -xzf /tmp/blokus-source.tar.gz -C /opt/blokus",
            "python -m pip install --no-deps /opt/blokus",
        )
        .add_local_file(__file__, "/opt/blokus/cloud/modal_blokus.py", copy=True)
    )
else:
    image = None

app = modal.App(APP)


def run_benchmark(checkpoint):
    import contextlib
    import platform
    import tempfile

    import numpy as np
    import torch
    from blokus_rl._native import DuoGame
    from blokus_rl.duo_az import AzConfig, fit
    from blokus_rl.duo_conv import DuoConvActorCritic
    from blokus_rl.duo_env import decision
    from blokus_rl.duo_learning import load_duo, pack
    from blokus_rl.duo_mcts import search_games
    from blokus_rl.learning import configure_torch

    configure_torch(260642)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    assert torch.cuda.is_available()
    assert len(DuoGame().legal_actions()) == 828
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frozen.pt"
        path.write_bytes(checkpoint)
        base = load_duo(path)
    result = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "measurements": [],
        "search": [],
        "training": [],
        "caveat": "Current batched pipeline, one search coordinator; no shared multiworker GPU inference server.",
    }
    observation, _, candidates = decision(DuoGame())
    for architecture in ("additive", "conv"):
        for mode in ("cpu", "cuda", "cuda_bfloat16"):
            if time.monotonic() - started > 210:
                break
            device = "cpu" if mode == "cpu" else "cuda"
            # Each mode starts with identical weights, even after synthetic updates.
            configure_torch(260642)
            if architecture == "additive":
                model = type(base)(base.hidden_size)
                model.load_state_dict(base.state_dict())
            else:
                model = DuoConvActorCritic()
            model = model.eval().to(device)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if mode == "cuda_bfloat16"
                else contextlib.nullcontext()
            )
            for size in (1, 8, 32, 128):
                batch = pack([{"observation": observation, "candidates": candidates}] * size).to(
                    device
                )
                timings = []
                with torch.no_grad(), context:
                    for iteration in range(13):
                        start = time.perf_counter()
                        distribution, value = model.distribution(
                            batch["observation"], batch["candidates"], batch["action_mask"]
                        )
                        distribution.probs.float().cpu().numpy()
                        value.float().cpu().numpy()
                        if iteration >= 3:
                            timings.append(time.perf_counter() - start)
                result["measurements"].append(
                    {
                        "architecture": architecture,
                        "mode": mode,
                        "batch": size,
                        "milliseconds": float(np.median(timings) * 1000),
                    }
                )
            # Real live-game tree traversal + legality + packing + neural inference.
            for live in (8, 32):
                games = [DuoGame() for _ in range(live)]
                # Independent reproducible openings exercise ragged legal lists.
                for index, game in enumerate(games):
                    opening_rng = np.random.default_rng(260642 + index)
                    for _ in range(4):
                        game.step(int(opening_rng.choice(game.legal_actions())))
                metrics = {}
                start = time.monotonic()
                with context:
                    for _ in range(4):
                        searched = search_games(model, games, 128, 8, metrics=metrics)
                        for game, search in zip(games, searched, strict=True):
                            game.step(search.action())
                seconds = time.monotonic() - start
                result["search"].append(
                    {
                        "architecture": architecture,
                        "mode": mode,
                        "live_games": live,
                        "plies": 4,
                        "seconds": seconds,
                        "decisions_per_second": 4 * live / seconds,
                        **metrics,
                    }
                )
            # Timing-only synthetic targets, never saved as a trained agent.
            records = [
                {
                    "observation": observation,
                    "candidates": candidates,
                    "policy_target": np.full(
                        len(candidates), 1 / len(candidates), dtype=np.float32
                    ),
                    "value_target": float(2 * (i % 2) - 1),
                }
                for i in range(128)
            ]
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            start = time.monotonic()
            with context:
                fit(
                    model,
                    optimizer,
                    records,
                    np.random.default_rng(260642),
                    AzConfig(gradient_steps=12, minibatch_size=128),
                )
            result["training"].append(
                {
                    "architecture": architecture,
                    "mode": mode,
                    "steps": 12,
                    "batch": 128,
                    "seconds": time.monotonic() - start,
                }
            )
    result["seconds"] = time.monotonic() - started
    return result


options = {
    "image": image,
    "cpu": (CPU_CORES, CPU_CORES),
    "memory": (MEMORY_GIB * 1024, MEMORY_GIB * 1024),
    "timeout": MAX_SECONDS,
    "max_containers": 1,
    "scaledown_window": 2,
    "retries": 0,
    "include_source": False,
}


@app.function(gpu="L4", **options)
def benchmark_l4(checkpoint: bytes):
    return run_benchmark(checkpoint)


@app.function(gpu="H100!", **options)
def benchmark_h100(checkpoint: bytes):
    return run_benchmark(checkpoint)


@app.local_entrypoint()
def main(gpu: str = "l4", out: str = "runs/modal-benchmark-l4-260642.json"):
    if gpu not in RATES:
        raise ValueError("choose l4 or h100; one GPU only")
    destination = ROOT / out
    if destination.exists():
        raise ValueError("refusing to overwrite an existing benchmark")
    checkpoint = ROOT / "runs/duo-frontier-training-342/best.pt"
    maximum = (RATES[gpu] + CPU_CORES * 0.0000131 + MEMORY_GIB * 0.00000222) * (MAX_SECONDS + 60)
    ledger = ROOT / "runs/modal-cost-ledger.jsonl"
    prior = (
        [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    )
    if sum(r["reserved_compute_usd"] for r in prior) + maximum > 5:
        raise ValueError(
            "initial $5 benchmark reservation exhausted; inspect ledger before further spend"
        )
    entry = {
        "gpu": gpu,
        "profile": "linuxfan2718",
        "status": "reserved",
        "reserved_compute_usd": maximum,
        "function_timeout_seconds": MAX_SECONDS,
        "cpu_limit": CPU_CORES,
        "memory_limit_gib": MEMORY_GIB,
        "source_sha256": hashlib.sha256(
            (ROOT / ".local/modal-source.tar.gz").read_bytes()
        ).hexdigest(),
        "out": str(destination),
        "benchmark_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    with ledger.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    function = benchmark_l4 if gpu == "l4" else benchmark_h100
    result = function.remote(checkpoint.read_bytes())
    result["budget"] = entry
    result["estimated_execution_usd"] = (
        RATES[gpu] + CPU_CORES * 0.0000131 + MEMORY_GIB * 0.00000222
    ) * result["seconds"]
    result["billing_note"] = (
        "Estimate excludes build/startup and metering differences; inspect Modal billing for actual charges."
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "gpu": result["gpu"],
                "seconds": result["seconds"],
                "estimated_execution_usd": result["estimated_execution_usd"],
                "out": str(destination),
            }
        )
    )
