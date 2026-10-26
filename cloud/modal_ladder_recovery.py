"""Prepared one-shot recovery; requires Dennis's explicit approval before launch.

Reuse the exact original installed engine/PyTorch image. Only this wrapper is
new; the benchmark's configuration, binary, model and cached games stay intact.
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
BASE_IMAGE = "im-uHIEZ1NXBeX8O5FJeGIZqA"
RUN_NAME = "duo-pentobi-ladder-2579000"
TIMEOUT = 2700
# Non-preemptible CPU/RAM have a verified provider 3x price multiplier.
HOURLY = 3 * (8 * 0.0473 + 24 * 0.008)
CEILING = 5.0
image = (
    modal.Image.from_id(BASE_IMAGE).add_local_file(
        __file__, "/opt/blokus/cloud/modal_ladder_recovery.py", copy=True
    )
    if modal.is_local()
    else None
)
app = modal.App("blokus-pentobi-diagnosed-recovery")
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(24576, 24576),
    nonpreemptible=True,
    timeout=TIMEOUT,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": volume},
)
def recover(expected_fingerprint, execution):
    from blokus_rl.duo_deep import atomic_json
    from blokus_rl.duo_pentobi_ladder import LadderConfig, run

    out = Path("/data") / RUN_NAME
    protocol = json.loads((out / "protocol.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("original frozen protocol changed")
    if (out / "completed.json").exists():
        raise ValueError("benchmark already finished; refusing duplicate recovery")
    before = {
        str(level): len(list((out / f"level-{level:02}").glob("pair-*.json")))
        for level in protocol["config"]["levels"]
    }
    atomic_json(out / "recovery-execution.json", {**execution, "pairs_before": before})
    volume.commit()
    started = time.monotonic()
    try:
        result = run(
            LadderConfig(**protocol["config"]),
            out / "model.pt",
            "/opt/pentobi-build/pentobi_gtp/pentobi-gtp",
            out,
            resume=True,
            commit=volume.commit,
            lock_path=Path("/tmp") / f"{RUN_NAME}.lock",
        )
        atomic_json(
            out / "hardware.json",
            {
                "cpu_request_and_cap": 8,
                "memory_gib_request_and_cap": 24,
                "gpu": None,
                "seconds_this_call": time.monotonic() - started,
                "finish_utc": datetime.now(timezone.utc).isoformat(),
                "run_name": RUN_NAME,
                "recovery": True,
                "nonpreemptible": True,
                "base_image": BASE_IMAGE,
                "first_attempt_app": "ap-8J7F0oIUjPAAYL4nbs591O",
                "note": "this call's seconds exclude original preempted attempt; report both",
            },
        )
        volume.commit()
        return {
            "directory": RUN_NAME,
            "seconds_this_call": time.monotonic() - started,
            "games": sum(row["games"] for row in result.values()),
        }
    finally:
        volume.commit()


@app.local_entrypoint()
def main(approved: bool = False):
    if not approved:
        raise ValueError("recovery awaits Dennis's explicit approval; do not launch automatically")
    apps = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "modal", "app", "list", "--json", "--profile", "linuxfan2718"],
            text=True,
        )
    )
    if any("pentobi" in row["description"] and int(row["tasks"]) > 0 for row in apps):
        raise ValueError("another Pentobi benchmark/recovery is already active")
    ledger = ROOT / "runs/modal-pentobi-ladder-cost-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    reservation = HOURLY * (TIMEOUT + 90) / 3600
    if sum(row["reserved_usd"] for row in rows) + reservation > CEILING:
        raise ValueError("original benchmark reservation ceiling would be exceeded")
    protocol = json.loads(
        (ROOT / "runs/pentobi-ladder-original-protocol-20261026.json").read_text()
    )
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "profile": "linuxfan2718",
        "kind": "explicitly approved diagnosed recovery after provider preemption",
        "reserved_usd": reservation,
        "hourly_resource_estimate_usd": HOURLY,
        "timeout_seconds": TIMEOUT,
        "base_image": BASE_IMAGE,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experimental_protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "resume": True,
        "nonpreemptible": True,
        "phase_ceiling_usd": CEILING,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (ROOT / f"runs/{RUN_NAME}-recovery-{stamp}-runner.py").write_bytes(Path(__file__).read_bytes())
    with ledger.open("a") as handle:
        handle.write(json.dumps(execution) + "\n")
    result = recover.remote(execution["experimental_protocol_sha256"], execution)
    (ROOT / f"runs/{RUN_NAME}-recovery-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
