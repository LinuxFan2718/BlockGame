"""CPU-only fixed-policy benchmark, using an already-authorized private Volume model."""

import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
import modal_deep as prior

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "blokus-pentobi-level-benchmark"
RUN_NAME = "duo-pentobi-ladder-2579000"
TIMEOUT = 21600
CPU_CORES, MEMORY_GIB = 8, 24
PHASE_CEILING = 5.0
BEST_SHA256 = "0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86"
MODEL_IN_VOLUME = "/data/duo-deep-study-764/frozen-conv.pt"
CONFIG = {
    "opening_pairs": 100,
    "seed": 2579000,
    "simulations": 128,
    "workers": 8,
    "command_timeout": 600,
    "levels": list(range(1, 10)),
}
image = (
    prior.image.add_local_file(__file__, "/opt/blokus/cloud/modal_pentobi_ladder.py", copy=True)
    if modal.is_local()
    else None
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=(CPU_CORES, CPU_CORES),
    memory=(MEMORY_GIB * 1024, MEMORY_GIB * 1024),
    timeout=TIMEOUT,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": prior.volume},
)
def benchmark(run_name, configuration, resume=False):
    from blokus_rl.duo_deep import atomic_json
    from blokus_rl.duo_pentobi_ladder import LadderConfig, run

    if not re.fullmatch(r"[a-z][a-z0-9-]{1,79}", run_name):
        raise ValueError("invalid artifact folder")
    model = Path(MODEL_IN_VOLUME)
    if hashlib.sha256(model.read_bytes()).hexdigest() != BEST_SHA256:
        raise ValueError("private frozen best-model hash mismatch")
    out = Path("/data") / run_name
    if (out / "completed.json").exists():
        raise ValueError("benchmark already finished; do not duplicate")
    if out.exists() and not resume:
        raise ValueError("existing incomplete run requires explicit resume")
    # Keep the per-container advisory lock off the Volume's cached filesystem.
    started = time.monotonic()
    result = run(
        LadderConfig(**configuration),
        model,
        "/opt/pentobi-build/pentobi_gtp/pentobi-gtp",
        out,
        resume=resume,
        commit=prior.volume.commit,
        lock_path=Path("/tmp") / f"{run_name}.lock",
    )
    atomic_json(
        out / "hardware.json",
        {
            "cpu_request_and_cap": CPU_CORES,
            "memory_gib_request_and_cap": MEMORY_GIB,
            "gpu": None,
            "machine": platform.machine(),
            "seconds_this_call": time.monotonic() - started,
            "finish_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
        },
    )
    prior.volume.commit()
    return {
        "directory": run_name,
        "seconds": time.monotonic() - started,
        "games": sum(row["games"] for row in result.values()),
        "scores": {level: row["score_rate"] for level, row in result.items()},
    }


@app.local_entrypoint()
def main(run_name: str = RUN_NAME, resume: bool = False):
    if run_name != RUN_NAME:
        raise ValueError("this bounded benchmark uses its predeclared artifact name")
    result_path = ROOT / f"runs/{run_name}-modal-result.json"
    if result_path.exists():
        raise ValueError("completed local result already exists")
    ledger = ROOT / "runs/modal-pentobi-ladder-cost-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    hourly = CPU_CORES * 0.0473 + MEMORY_GIB * 0.008
    reservation = hourly * (TIMEOUT + 90) / 3600
    if sum(r["reserved_usd"] for r in rows) + reservation > PHASE_CEILING:
        raise ValueError("benchmark phase reservation ceiling exceeded; no automatic retry")
    utc = datetime.now(timezone.utc)
    payload = ROOT / ".local/modal-source.tar.gz"
    saved_payload = ROOT / f"runs/{run_name}-{utc:%Y%m%dT%H%M%SZ}-source.tar.gz"
    saved_payload.write_bytes(payload.read_bytes())
    saved_runner = saved_payload.with_name(saved_payload.name.replace("source.tar.gz", "runner.py"))
    saved_runner.write_bytes(Path(__file__).read_bytes())
    entry = {
        "utc": utc.isoformat(),
        "profile": "linuxfan2718",
        "run_name": run_name,
        "reserved_usd": reservation,
        "phase_ceiling_usd": PHASE_CEILING,
        "hourly_resource_estimate_usd": hourly,
        "timeout_seconds": TIMEOUT,
        "config": CONFIG,
        "resume": resume,
        "gpu": None,
        "source_sha256": hashlib.sha256(saved_payload.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(saved_runner.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint_sha256": BEST_SHA256,
        "checkpoint_source": "existing private Modal Volume; no model upload",
    }
    with ledger.open("a") as file:
        file.write(json.dumps(entry) + "\n")
    result = benchmark.remote(run_name, CONFIG, resume)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
