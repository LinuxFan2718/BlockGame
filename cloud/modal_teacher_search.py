"""One bounded 1024-versus-2048 self-play study; only two source files are uploaded."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "duo-teacher-search-1066"
BASE_IMAGE = "im-BnyLYweX6vlwu6iCpDnqrC"
VOLUME = "blokus-deep-selfplay-20261026"
PARENT = "/data/duo-continuation-865/frozen-strong1024.pt"
STATE = "/data/duo-continuation-865/strong1024/resume.pt"
PARENT_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
STATE_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
EVALUATOR_SHA = "80727fc7fee1fcdab75eebc989f41f617e6721d4f546e266367ca53f294a0493"
BINARY_SHA = "ccb2362fc2eb066be9337b76488c527f22aae4ad4b72296ae105e54b2c015882"
TIMEOUT, CEILING = 43200, 32.0
HOURLY = 1.95 + 8 * 0.0473 + 24 * 0.008
HELPER = ROOT / "python/blokus_rl/duo_continuation.py"
TRAINING_MODULES = ("duo_deep", "duo_shared", "duo_mcts", "duo_deep_evaluate", "duo_conv")
image = (
    modal.Image.from_id(BASE_IMAGE)
    .add_local_file(__file__, "/opt/blokus/cloud/modal_teacher_search.py", copy=True)
    .add_local_file(HELPER, "/opt/blokus/cloud/teacher_continuation.py", copy=True)
    if modal.is_local()
    else None
)
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
app = modal.App("blokus-teacher-search-study")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Telemetry:
    def __init__(self, path):
        self.path, self.stop = path, threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                row = {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_busy,memory_used,memory_total": result.strip(),
                }
            except Exception as error:
                row = {"error": str(error)}
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.stop.wait(5)

    def __enter__(self):
        self.thread.start()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


@app.function(
    image=image,
    gpu="L40S",
    cpu=(8, 8),
    memory=(24576, 24576),
    timeout=TIMEOUT,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": volume},
)
def study(execution):
    import numpy as np
    import torch
    from blokus_rl.duo_crossplay import opening_book, paired_interval
    from blokus_rl.duo_deep import DeepConfig, atomic_json, train
    from blokus_rl.duo_deep_evaluate import evaluate_many
    from blokus_rl.duo_learning import load_duo

    root = Path("/data") / RUN
    if root.exists():
        raise ValueError("existing study; no duplicate or automatic retry")
    if digest(PARENT) != PARENT_SHA or digest(STATE) != STATE_SHA:
        raise ValueError("private parent weights/replay state changed")
    helper_path = Path("/opt/blokus/cloud/teacher_continuation.py")
    evaluator_path = Path("/opt/blokus/cloud/duo_pentobi_steps.py")
    binary = Path("/opt/pentobi-build/pentobi_gtp/pentobi-gtp")
    if (
        digest(helper_path) != execution["helper_sha256"]
        or digest(evaluator_path) != EVALUATOR_SHA
        or digest(binary) != BINARY_SHA
    ):
        raise ValueError("source/evaluator/Pentobi binary changed")
    for name, expected in execution["training_module_sha256"].items():
        if digest(importlib.import_module("blokus_rl." + name).__file__) != expected:
            raise ValueError("immutable installed training module differs: " + name)
    initialize = load_module("teacher_continuation", helper_path).initialize_branch
    evaluator = load_module("steps", evaluator_path)
    root.mkdir()
    atomic_json(root / "execution.json", execution)
    started = time.monotonic()
    torch.set_float32_matmul_precision("high")
    config = DeepConfig(
        architecture="conv",
        seed=1066,
        rounds=64,
        eval_every=16,
        full_simulations=1024,
        validation_openings=32,
        validation_simulations=512,
        validation_seed=2605300,
        test_seed=2605900,
        max_seconds=9000,
    )
    configs = {
        "control1024": config,
        "teacher2048": replace(config, full_simulations=2048, max_seconds=16200),
    }
    atomic_json(
        root / "study-protocol.json",
        {
            "parent_sha256": PARENT_SHA,
            "source_resume_sha256": STATE_SHA,
            "configs": {k: asdict(v) for k, v in configs.items()},
            "initialization": "selected best round112 weights; both AdamW optimizers reset; same 100000 latest round128 replay positions",
            "primary": "paired Pentobi4 score difference teacher2048 minus preserved parent, both playing2048",
            "secondary": "teacher2048 minus control1024 against Pentobi4; head-to-head; raw policies; Pentobi3",
            "stretch": "teacher2048 absolute Pentobi4 score interval above0.5",
            "selection": "every16 rounds against parent512 on32 validation pairs; parent selectable at0.5",
            "heldout_seed": 2605900,
            "heldout_pairs": 100,
            "hardware": {
                "gpu": torch.cuda.get_device_name(),
                "cpu_cap": 8,
                "ram_gib_cap": 24,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
            },
        },
    )
    volume.commit()
    try:
        with Telemetry(root / "telemetry.jsonl"):
            smoke = replace(
                config,
                rounds=2,
                games_per_round=8,
                workers=2,
                live_games=4,
                full_simulations=16,
                fast_simulations=4,
                eval_every=2,
                validation_openings=1,
                validation_simulations=8,
                max_seconds=300,
            )
            initialize(smoke, root / "smoke", PARENT, STATE, selected_best=True)
            train(smoke, root / "smoke", PARENT, resume=True, stop_after_round=1)
            train(smoke, root / "smoke", PARENT, resume=True, commit=volume.commit)
            atomic_json(root / "cuda-smoke-passed.json", {"rounds": 2, "unpromoted": True})
            for name, cfg in configs.items():
                initialize(cfg, root / name, PARENT, STATE, selected_best=True)
                volume.commit()
                print("Starting training " + name, flush=True)
                train(cfg, root / name, PARENT, resume=True, commit=volume.commit)
            frozen = {}
            for name in configs:
                path = root / f"frozen-{name}.pt"
                path.write_bytes((root / name / "best.pt").read_bytes())
                frozen[name] = digest(path)
            # Freeze before generating any test opening. Test results never select a checkpoint.
            atomic_json(
                root / "heldout-protocol.json",
                {
                    "checkpoint_sha256": frozen,
                    "parent_sha256": PARENT_SHA,
                    "seed": 2605900,
                    "pairs": 100,
                    "simulations": 2048,
                    "levels": [3, 4],
                    "precision": "bfloat16",
                },
            )
            volume.commit()
            book = opening_book(100, seed=2605900)
            paths = {"parent": Path(PARENT)} | {
                name: root / f"frozen-{name}.pt" for name in configs
            }
            results = {}
            for name, path in paths.items():
                for level in (3, 4):
                    case = f"{name}-pentobi{level}"
                    print("Starting held-out " + case, flush=True)
                    out = root / "heldout" / case
                    out.parent.mkdir(exist_ok=True)
                    results[case] = evaluator.run(
                        {
                            "first_level": level,
                            "last_level": level,
                            "opening_pairs": 100,
                            "seed": 2605900,
                            "device": "cuda",
                            "precision": "bfloat16",
                            "simulations": 2048,
                        },
                        path,
                        binary,
                        out,
                        commit=volume.commit,
                    )[str(level)]
                    atomic_json(root / "heldout-results.json", results)
                    volume.commit()
            models = {name: load_duo(path).to("cuda").eval() for name, path in paths.items()}
            comparisons = [
                ("teacher-vs-control-search2048", "teacher2048", "control1024", 2048),
                ("teacher-vs-control-raw", "teacher2048", "control1024", 0),
                ("teacher-vs-parent-raw", "teacher2048", "parent", 0),
                ("control-vs-parent-raw", "control1024", "parent", 0),
            ]
            for name, left, right, budget in comparisons:
                print("Starting held-out " + name, flush=True)
                row = evaluate_many(
                    models[left],
                    models[right],
                    book,
                    simulations=budget,
                    opponent_simulations=budget,
                    seed=2605900,
                    precision="bfloat16",
                )
                atomic_json(root / f"heldout-{name}.json", row)
                volume.commit()

            def difference(left, right):
                delta = np.asarray(results[left]["opening_pair_scores"]) - np.asarray(
                    results[right]["opening_pair_scores"]
                )
                return {
                    "score_difference": float(delta.mean()),
                    "nominal_paired_95pct_interval": paired_interval(delta),
                }

            atomic_json(
                root / "paired-differences.json",
                {
                    "teacher-minus-parent-pentobi4": difference(
                        "teacher2048-pentobi4", "parent-pentobi4"
                    ),
                    "teacher-minus-control-pentobi4": difference(
                        "teacher2048-pentobi4", "control1024-pentobi4"
                    ),
                    "control-minus-parent-pentobi4": difference(
                        "control1024-pentobi4", "parent-pentobi4"
                    ),
                },
            )
            if digest(PARENT) != PARENT_SHA or digest(STATE) != STATE_SHA:
                raise ValueError("preserved source changed during study")
            atomic_json(
                root / "finished.json",
                {
                    "seconds": time.monotonic() - started,
                    "finish_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_sha256": frozen,
                    "parent_sha256": PARENT_SHA,
                    "source_resume_sha256": STATE_SHA,
                    "training": {
                        name: json.loads((root / name / "completed.json").read_text())
                        for name in configs
                    },
                    "external_games": 1200,
                    "head_to_head_games": 800,
                },
            )
            volume.commit()
        return {"run_name": RUN, "checkpoint_sha256": frozen, "seconds": time.monotonic() - started}
    except BaseException:
        atomic_json(
            root / "failure.json",
            {"traceback": traceback.format_exc(), "seconds": time.monotonic() - started},
        )
        volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-teacher-search-cost-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    reservation = HOURLY * (TIMEOUT + 90) / 3600
    if sum(r["reserved_usd"] for r in rows) + reservation > CEILING:
        raise ValueError("study reservation ceiling reached; no duplicate launch")
    command = [sys.executable, "-m", "modal"]
    apps = json.loads(
        subprocess.check_output(
            command + ["app", "list", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    if any(int(r["tasks"]) for r in apps):
        raise ValueError("active apps; inspect before launching")
    billing = json.loads(
        subprocess.check_output(
            command + ["billing", "summary", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    if max(40.0, float(billing["metered_cost"])) + reservation > 130 - 50:
        raise ValueError("preserve the $50 reserve")
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN,
        "profile": "linuxfan2718",
        "reserved_usd": reservation,
        "phase_ceiling_usd": CEILING,
        "timeout_seconds": TIMEOUT,
        "hourly_usd": HOURLY,
        "base_image": BASE_IMAGE,
        "parent_sha256": PARENT_SHA,
        "source_resume_sha256": STATE_SHA,
        "runner_sha256": digest(__file__),
        "helper_sha256": digest(HELPER),
        "evaluator_sha256": EVALUATOR_SHA,
        "binary_sha256": BINARY_SHA,
        "training_module_sha256": {
            name: digest(ROOT / f"python/blokus_rl/{name}.py") for name in TRAINING_MODULES
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "uploads": "runner and continuation helper only; no model/replay",
    }
    prefix = ROOT / "runs" / RUN
    Path(str(prefix) + "-runner.py").write_bytes(Path(__file__).read_bytes())
    Path(str(prefix) + "-helper.py").write_bytes(HELPER.read_bytes())
    Path(str(prefix) + "-billing-before.json").write_text(json.dumps(billing, indent=2) + "\n")
    with ledger.open("a") as f:
        f.write(json.dumps(execution) + "\n")
    result = study.remote(execution)
    Path(str(prefix) + "-modal-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
