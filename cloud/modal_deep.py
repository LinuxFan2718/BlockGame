"""Bounded overnight study: private checkpoints, one GPU, explicit cost reservations.

Only the existing allowlisted game/training sources and original frontier342
checkpoint leave the Mac. No third-party model, credentials or run collection is
uploaded. Benchmark weights are discarded; the main study starts from scratch.
"""

import hashlib
import json
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import modal
from modal_blokus import ROOT
from modal_blokus import image as base_image

APP_NAME = "blokus-deep-selfplay-study"
VOLUME_NAME = "blokus-deep-selfplay-20261026"
GPU_HOURLY = {"l4": 0.8, "l40s": 1.95, "h100": 3.95}
CPU_CORES, MEMORY_GIB = 8, 24
STUDY_TIMEOUT = 25200
PHASE_CEILING_USD = 35

if modal.is_local():
    image = (
        base_image.apt_install("cmake", "git")
        .run_commands(
            "git clone --depth 1 --branch v31.0 https://github.com/enz/pentobi.git /opt/pentobi-source",
            "python -c \"import subprocess; assert subprocess.check_output(['git','-C','/opt/pentobi-source','rev-parse','HEAD']).decode().strip() == 'd5374ddc20eced902788e733d67ee16bd1836fd3'\"",
            "cmake -S /opt/pentobi-source -B /opt/pentobi-build -DCMAKE_BUILD_TYPE=Release -DPENTOBI_BUILD_GUI=OFF -DPENTOBI_BUILD_GTP=ON",
            "cmake --build /opt/pentobi-build --target pentobi-gtp --parallel 4",
        )
        .add_local_file(__file__, "/opt/blokus/cloud/modal_deep.py", copy=True)
    )
else:
    image = None
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


class Telemetry:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                row = {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_busy_percent,memory_io_percent,memory_used_mib,memory_total_mib,power_watts": result.stdout.strip(),
                }
            except Exception as error:
                row = {"telemetry_error": str(error)}
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            self.stop.wait(5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


def remote_job(stage, run_name, reference_bytes, config_dict, resume=False):
    import numpy as np
    import torch
    from blokus_rl.duo_crossplay import opening_book
    from blokus_rl.duo_deep import DeepConfig, atomic_json, fit_indexed, train
    from blokus_rl.duo_deep_evaluate import evaluate_many, evaluate_pentobi
    from blokus_rl.duo_learning import load_duo
    from blokus_rl.duo_residual import DuoResidualActorCritic
    from blokus_rl.duo_shared import SharedSearchPool
    from blokus_rl.learning import configure_torch

    if not re.fullmatch(r"[a-z][a-z0-9-]{1,79}", run_name):
        raise ValueError("invalid artifact directory")
    if (
        hashlib.sha256(reference_bytes).hexdigest()
        != "259b9c8fe18968ad8bef6fa05271807a3280b2208d92ed8fa5ea21f22386f326"
    ):
        raise ValueError("only the approved frozen frontier342 reference may be uploaded")
    root = Path("/data") / run_name
    if (root / "finished.json").exists():
        raise ValueError("study already finished; refusing duplicate compute")
    if root.exists() and not resume:
        raise ValueError("artifact directory exists; inspect and resume exact config if needed")
    root.mkdir(exist_ok=True)
    reference = root / "reference.pt"
    if reference.exists() and reference.read_bytes() != reference_bytes:
        raise ValueError("stored reference changed")
    reference.write_bytes(reference_bytes)
    config = DeepConfig(**config_dict)
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "config": config_dict,
        "stage": stage,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "start_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(root / "hardware.json", metadata)
    volume.commit()
    started = time.monotonic()
    with Telemetry(root / "telemetry.jsonl"):
        if stage == "benchmark":
            configure_torch(config.seed)
            torch.set_float32_matmul_precision("high")
            model = (
                DuoResidualActorCritic(config.hidden_size, config.channels, config.blocks)
                .eval()
                .cuda()
            )
            rows = []
            for workers in (1, config.workers):
                benchmark = replace(
                    config,
                    workers=workers,
                    fast_probability=0,
                    full_simulations=64,
                    live_games=16,
                )
                print(f"Benchmarking {workers} CPU search workers", flush=True)
                with SharedSearchPool(benchmark) as pool:
                    records, scores, metrics = pool.collect(model, 1, 48)
                rows.append(
                    {
                        "workers": workers,
                        "games": len(scores),
                        "metrics": metrics,
                        "games_per_second": len(scores) / metrics["collect_seconds"],
                    }
                )
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            fit_start = time.monotonic()
            losses = fit_indexed(
                model, optimizer, records, np.random.default_rng(config.seed), config, 12
            )
            result = {
                "collection": rows,
                "fit_12_steps_seconds": time.monotonic() - fit_start,
                "losses": losses,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "note": "Timing/finite-loss screen only. All updated weights discarded.",
            }
            # Exercise CUDA optimizer, checkpoint and RNG restoration before a long run.
            smoke = replace(
                config,
                rounds=2,
                games_per_round=12,
                workers=2,
                live_games=6,
                full_simulations=16,
                fast_simulations=4,
                eval_every=1,
                validation_openings=1,
                validation_simulations=8,
                validation_seed=2561000,
                minibatch_size=16,
                training_ratio=1,
                max_seconds=120,
            )
            with tempfile.TemporaryDirectory() as temporary:
                smoke_out = Path(temporary) / "cuda-smoke"
                train(smoke, smoke_out, reference, stop_after_round=1)
                train(smoke, smoke_out, reference, resume=True)
                completion = json.loads((smoke_out / "completed.json").read_text())
                if completion["games"] != 24:
                    raise RuntimeError("CUDA resumed smoke run did not complete exactly24 games")
                result["cuda_resume_smoke"] = completion
            atomic_json(root / "benchmark.json", result)
        elif stage == "study":
            if not (root / "rules-crosscheck.json").exists():
                from blokus_rl.duo_pentobi import Pentobi, match

                with Pentobi(
                    binary="/opt/pentobi-build/pentobi_gtp/pentobi-gtp", seed=42
                ) as external:
                    crosscheck = match(external, "largest", 0, 42, check_legal=True)
                atomic_json(root / "rules-crosscheck.json", crosscheck)
                volume.commit()
            # Same game/search/optimizer settings and game seed, separate random models.
            # Jobs are serial so the study owns at most one GPU throughout.
            for architecture in ("residual", "conv"):
                selected = replace(config, architecture=architecture)
                if architecture == "conv":
                    completed_deep = json.loads((root / "residual/completed.json").read_text())
                    selected = replace(selected, rounds=completed_deep["rounds"])
                out = root / architecture
                if (out / "completed.json").exists():
                    print(f"Preserving completed {architecture} run", flush=True)
                    continue
                print(f"Starting {architecture} cold self-play", flush=True)
                train(
                    selected,
                    out,
                    reference,
                    resume=(out / "resume.pt").exists(),
                    commit=volume.commit,
                )
                volume.commit()
            # Freeze selection before generating the reserved test openings.
            frozen = {}
            for architecture in ("residual", "conv"):
                source = root / architecture / "best.pt"
                destination = root / f"frozen-{architecture}.pt"
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
                frozen[architecture] = hashlib.sha256(destination.read_bytes()).hexdigest()
            atomic_json(
                root / "heldout-protocol.json",
                {
                    "seed": config.test_seed,
                    "opening_pairs": 100,
                    "checkpoint_sha256": frozen,
                    "primary": "residual search128 versus shallow CNN search128; whole opening-pair intervals",
                    "secondary": "each frozen model raw/search against original reference and Pentobi31.0 level1; warm round8 comparison local after download",
                    "actual_training_budgets": {
                        a: json.loads((root / a / "completed.json").read_text())
                        for a in ("residual", "conv")
                    },
                    "selection": "searched validation against original reference; test never used for selection",
                },
            )
            volume.commit()
            book = opening_book(100, seed=config.test_seed)
            models = {a: load_duo(root / f"frozen-{a}.pt").cuda() for a in frozen}
            models["reference"] = load_duo(reference).cuda()
            cases = [
                ("residual_vs_conv_search128", "residual", "conv", 128),
                ("residual_vs_conv_raw", "residual", "conv", 0),
                ("residual_vs_reference_search128", "residual", "reference", 128),
                ("conv_vs_reference_search128", "conv", "reference", 128),
                ("residual_vs_reference_raw", "residual", "reference", 0),
                ("conv_vs_reference_raw", "conv", "reference", 0),
            ]
            for name, left, right, budget in cases:
                destination = root / f"heldout-{name}.json"
                if destination.exists():
                    continue
                print(f"Held-out evaluation: {name}", flush=True)
                result = evaluate_many(
                    models[left],
                    models[right],
                    book,
                    simulations=budget,
                    opponent_simulations=budget,
                    seed=config.test_seed,
                    precision=config.precision,
                )
                atomic_json(destination, result)
                volume.commit()
                print(json.dumps({"case": name, "score_rate": result["score_rate"]}), flush=True)
            result = {"frozen": frozen, "cases": [case[0] for case in cases]}
            for architecture in ("residual", "conv"):
                for budget in (0, 128):
                    name = f"{architecture}_vs_pentobi_{budget}"
                    destination = root / f"heldout-{name}.json"
                    if not destination.exists():
                        print(f"Held-out Pentobi evaluation: {name}", flush=True)
                        evaluate_pentobi(
                            models[architecture],
                            "/opt/pentobi-build/pentobi_gtp/pentobi-gtp",
                            book,
                            destination,
                            simulations=budget,
                            seed=config.test_seed,
                            precision=config.precision,
                            commit=volume.commit,
                        )
                    result["cases"].append(name)
                for budget in (0, 128):
                    name = f"{architecture}_normal_start_{budget}"
                    destination = root / f"descriptive-{name}.json"
                    if not destination.exists():
                        atomic_json(
                            destination,
                            evaluate_many(
                                models[architecture],
                                models["reference"],
                                [[]],
                                simulations=budget,
                                opponent_simulations=budget,
                                seed=config.test_seed,
                                precision=config.precision,
                            ),
                        )
                        volume.commit()
        else:
            raise ValueError("choose benchmark or study")
    atomic_json(
        root / "finished.json",
        {
            "stage": stage,
            "seconds": time.monotonic() - started,
            "finish_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    volume.commit()
    return metadata | {
        "result": result,
        "seconds": time.monotonic() - started,
        "directory": run_name,
    }


OPTIONS = {
    "image": image,
    "cpu": (CPU_CORES, CPU_CORES),
    "memory": (MEMORY_GIB * 1024, MEMORY_GIB * 1024),
    "max_containers": 1,
    "scaledown_window": 2,
    "retries": 0,
    "include_source": False,
    "volumes": {"/data": volume},
}


@app.function(gpu="L4", timeout=600, **OPTIONS)
def benchmark_l4(*args):
    return remote_job(*args)


@app.function(gpu="L40S", timeout=600, **OPTIONS)
def benchmark_l40s(*args):
    return remote_job(*args)


@app.function(gpu="H100!", timeout=600, **OPTIONS)
def benchmark_h100(*args):
    return remote_job(*args)


@app.function(gpu="L4", timeout=STUDY_TIMEOUT, **OPTIONS)
def study_l4(*args):
    return remote_job(*args)


@app.function(gpu="L40S", timeout=STUDY_TIMEOUT, **OPTIONS)
def study_l40s(*args):
    return remote_job(*args)


@app.function(gpu="H100!", timeout=STUDY_TIMEOUT, **OPTIONS)
def study_h100(*args):
    return remote_job(*args)


@app.local_entrypoint()
def main(
    stage: str = "benchmark",
    gpu: str = "l4",
    run_name: str = "deep-benchmark-l4-764",
    config: str = "runs/deep-study-config.json",
    resume: bool = False,
):
    if gpu not in GPU_HOURLY or stage not in ("benchmark", "study"):
        raise ValueError("invalid stage/GPU")
    # A local config is data, not uploaded credentials or arbitrary paths.
    configuration = json.loads((ROOT / config).read_text())
    destination = ROOT / f"runs/{run_name}-modal-result.json"
    if destination.exists():
        raise ValueError("refusing to repeat a completed local result")
    timeout = 600 if stage == "benchmark" else STUDY_TIMEOUT
    hourly = GPU_HOURLY[gpu] + CPU_CORES * 0.0473 + MEMORY_GIB * 0.008
    reservation = hourly * (timeout + 90) / 3600
    ledger = ROOT / "runs/modal-deep-cost-ledger.jsonl"
    prior = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    if sum(row["reserved_usd"] for row in prior) + reservation > PHASE_CEILING_USD:
        raise ValueError(
            "overnight phase reservation ceiling reached; inspect costs before more compute"
        )
    entry = {
        "stage": stage,
        "gpu": gpu,
        "profile": "linuxfan2718",
        "run_name": run_name,
        "reserved_usd": reservation,
        "timeout_seconds": timeout,
        "utc": datetime.now(timezone.utc).isoformat(),
        "resume": resume,
        "source_sha256": hashlib.sha256(
            (ROOT / ".local/modal-source.tar.gz").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_runner_sha256": hashlib.sha256(
            (ROOT / "cloud/modal_blokus.py").read_bytes()
        ).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "config": configuration,
    }
    with ledger.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    # Retain the exact payload/code locally, even if a later fix changes the image.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload_prefix = ROOT / f"runs/{run_name}-{timestamp}"
    Path(str(payload_prefix) + "-source.tar.gz").write_bytes(
        (ROOT / ".local/modal-source.tar.gz").read_bytes()
    )
    Path(str(payload_prefix) + "-runner.py").write_bytes(Path(__file__).read_bytes())
    function = globals()[f"{stage}_{gpu}"]
    reference = ROOT / "runs/duo-frontier-training-342/best.pt"
    result = function.remote(stage, run_name, reference.read_bytes(), configuration, resume)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"directory": run_name, "gpu": result["gpu"], "seconds": result["seconds"]}))
