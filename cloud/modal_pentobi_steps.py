"""Bounded sequential Pentobi levels for the new frozen strongest model."""

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
BASE_IMAGE = "im-uHIEZ1NXBeX8O5FJeGIZqA"
RUN_NAME = "duo-strong-pentobi-steps-2599000"
MODEL = "/data/duo-continuation-865/frozen-strong1024.pt"
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
TIMEOUT, CEILING = 7200, 5.0
HOURLY = 1.95 + 2 * 0.0473 + 8 * 0.008
CONFIG = {
    "first_level": 2,
    "last_level": 9,
    "opening_pairs": 100,
    "seed": 2599000,
    "simulations": 128,
    "device": "cuda",
    "precision": "bfloat16",
}
MODULE = ROOT / "python/blokus_rl/duo_pentobi_steps.py"
image = (
    modal.Image.from_id(BASE_IMAGE)
    .add_local_file(MODULE, "/opt/blokus/cloud/duo_pentobi_steps.py", copy=True)
    .add_local_file(__file__, "/opt/blokus/cloud/modal_pentobi_steps.py", copy=True)
    if modal.is_local()
    else None
)
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)
app = modal.App("blokus-strong-sequential-pentobi")


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

    out = Path("/data") / RUN_NAME
    if out.exists():
        raise ValueError("existing run; no duplicate or automatic recovery")
    if hashlib.sha256(Path(MODEL).read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError("private frozen model hash changed")
    spec = importlib.util.spec_from_file_location("steps", "/opt/blokus/cloud/duo_pentobi_steps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    try:
        results = module.run(
            CONFIG, MODEL, "/opt/pentobi-build/pentobi_gtp/pentobi-gtp", out, commit=volume.commit
        )
        atomic_json(
            out / "hardware.json",
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cpu_cap": 2,
                "ram_gib_cap": 8,
                "seconds": time.monotonic() - started,
                "finish_utc": datetime.now(timezone.utc).isoformat(),
                "execution": execution,
            },
        )
        volume.commit()
        return {"run_name": RUN_NAME, "scores": {k: r["score_rate"] for k, r in results.items()}}
    except BaseException:
        if out.exists():
            atomic_json(out / "failure.json", {"traceback": traceback.format_exc()})
            volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-pentobi-steps-cost-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    reservation = HOURLY * (TIMEOUT + 90) / 3600
    if sum(r["reserved_usd"] for r in rows) + reservation > CEILING:
        raise ValueError("phase ceiling reached; no repeated launches")
    command = [sys.executable, "-m", "modal"]
    apps = json.loads(
        subprocess.check_output(
            command + ["app", "list", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    if any(int(r["tasks"]) for r in apps):
        raise ValueError("inspect active apps before launching this benchmark")
    billing = json.loads(
        subprocess.check_output(
            command + ["billing", "summary", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    if max(35.0, float(billing["metered_cost"])) + reservation > 130 - 50:
        raise ValueError("preserve the $50 credit reserve")
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
        "config": CONFIG,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    prefix = ROOT / f"runs/{RUN_NAME}"
    for source, label in ((MODULE, "module"), (Path(__file__), "runner")):
        data = source.read_bytes()
        execution[label + "_sha256"] = hashlib.sha256(data).hexdigest()
        Path(str(prefix) + f"-{label}.py").write_bytes(data)
    Path(str(prefix) + "-billing-before.json").write_text(json.dumps(billing, indent=2) + "\n")
    with ledger.open("a") as handle:
        handle.write(json.dumps(execution) + "\n")
    result = benchmark.remote(execution)
    Path(str(prefix) + "-modal-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
