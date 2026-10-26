"""One bounded, fixed-model playing-search sweep against Pentobi level2."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# Includes the exact validated evaluator from the completed level2 test.
BASE_IMAGE = "im-BnyLYweX6vlwu6iCpDnqrC"
EVALUATOR_SHA = "80727fc7fee1fcdab75eebc989f41f617e6721d4f546e266367ca53f294a0493"
RUN_NAME = "duo-playing-search-sweep-2601000"
APP_NAME = "blokus-frozen-playing-search-sweep"
MODEL = "/data/duo-continuation-865/frozen-strong1024.pt"
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
BUDGETS = (128, 512, 2048)
TIMEOUT, CEILING = 7200, 5.0
HOURLY = 1.95 + 2 * 0.0473 + 8 * 0.008
CONFIG = {
    "first_level": 2,
    "last_level": 2,
    "opening_pairs": 100,
    "seed": 2601000,
    "device": "cuda",
    "precision": "bfloat16",
}
image = (
    modal.Image.from_id(BASE_IMAGE).add_local_file(
        __file__, "/opt/blokus/cloud/modal_search_sweep.py", copy=True
    )
    if modal.is_local()
    else None
)
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L40S",
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=TIMEOUT,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": volume},
)
def benchmark(execution):
    import torch
    from blokus_rl.duo_deep import atomic_json

    root = Path("/data") / RUN_NAME
    if root.exists():
        raise ValueError("existing sweep; no duplicate or automatic recovery")
    if hashlib.sha256(Path(MODEL).read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError("preserved frozen model changed")
    evaluator = Path("/opt/blokus/cloud/duo_pentobi_steps.py")
    if hashlib.sha256(evaluator.read_bytes()).hexdigest() != EVALUATOR_SHA:
        raise ValueError("immutable evaluator changed")
    spec = importlib.util.spec_from_file_location("steps", evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root.mkdir()
    atomic_json(
        root / "protocol.json",
        {
            "execution": execution,
            "budgets": list(BUDGETS),
            "shared_config": CONFIG,
            "model_sha256": MODEL_SHA,
            "evaluator_sha256": EVALUATOR_SHA,
            "primary": "paired score difference: search2048 minus search128",
            "secondary": "512 minus128; 2048 minus512; absolute Pentobi scores and runtime",
            "stopping": "complete all3 fixed budgets, regardless of intermediate scores",
            "inference": "whole-opening paired bootstrap, nominal95%; one frozen lineage",
        },
    )
    volume.commit()
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    results = {}
    try:
        for budget in BUDGETS:
            print(f"Starting frozen model at {budget} simulations", flush=True)
            arm_started = time.monotonic()
            result = module.run(
                CONFIG | {"simulations": budget},
                MODEL,
                "/opt/pentobi-build/pentobi_gtp/pentobi-gtp",
                root / f"search-{budget}",
                commit=volume.commit,
            )["2"]
            results[str(budget)] = result
            atomic_json(
                root / f"search-{budget}/timing.json",
                {
                    "seconds": time.monotonic() - arm_started,
                    "finish_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            atomic_json(root / "results.json", results)
            volume.commit()
            print(
                json.dumps(
                    {
                        "simulations": budget,
                        "score": result["score_rate"],
                        "wins": result["wins"],
                        "draws": result["draws"],
                        "losses": result["losses"],
                    }
                ),
                flush=True,
            )
        if hashlib.sha256(Path(MODEL).read_bytes()).hexdigest() != MODEL_SHA:
            raise ValueError("private source model changed during evaluation")
        atomic_json(
            root / "hardware.json",
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cpu_cap": 2,
                "ram_gib_cap": 8,
                "seconds": time.monotonic() - started,
                "finish_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        atomic_json(
            root / "finished.json",
            {
                "budgets": list(BUDGETS),
                "games": sum(r["games"] for r in results.values()),
                "model_sha256": MODEL_SHA,
                "seconds": time.monotonic() - started,
            },
        )
        volume.commit()
        return {"run_name": RUN_NAME, "scores": {k: r["score_rate"] for k, r in results.items()}}
    except BaseException:
        atomic_json(
            root / "failure.json",
            {
                "traceback": traceback.format_exc(),
                "seconds": time.monotonic() - started,
                "completed_budgets": list(results),
            },
        )
        volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-search-sweep-cost-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    reservation = HOURLY * (TIMEOUT + 90) / 3600
    if sum(r["reserved_usd"] for r in rows) + reservation > CEILING:
        raise ValueError("phase cap reached; no repeated launches")
    command = [sys.executable, "-m", "modal"]
    apps = json.loads(
        subprocess.check_output(
            command + ["app", "list", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    if any(int(r["tasks"]) for r in apps):
        raise ValueError("inspect active apps before launching")
    billing = json.loads(
        subprocess.check_output(
            command + ["billing", "summary", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    if max(40.0, float(billing["metered_cost"])) + reservation > 130 - 50:
        raise ValueError("preserve the $50 reserve")
    data = Path(__file__).read_bytes()
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "profile": "linuxfan2718",
        "reserved_usd": reservation,
        "phase_ceiling_usd": CEILING,
        "timeout_seconds": TIMEOUT,
        "hourly_usd": HOURLY,
        "base_image": BASE_IMAGE,
        "model_sha256": MODEL_SHA,
        "model_source": "existing private Volume; no model upload",
        "runner_sha256": hashlib.sha256(data).hexdigest(),
        "evaluator_sha256": EVALUATOR_SHA,
        "budgets": list(BUDGETS),
        "shared_config": CONFIG,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    prefix = ROOT / f"runs/{RUN_NAME}"
    Path(str(prefix) + "-runner.py").write_bytes(data)
    Path(str(prefix) + "-billing-before.json").write_text(json.dumps(billing, indent=2) + "\n")
    with ledger.open("a") as handle:
        handle.write(json.dumps(execution) + "\n")
    result = benchmark.remote(execution)
    Path(str(prefix) + "-modal-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
