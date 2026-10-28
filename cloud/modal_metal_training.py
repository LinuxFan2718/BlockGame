"""One bounded L40S comparison for the Metal training port; $1.50 ceiling."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROFILE = "linuxfan2718"
RUN = "duo-metal-training-benchmark-1090"
BASE_IMAGE = "im-yX3zmnEKl2EgUtndThw9li"
NATIVE_SHA = "257b5a55c2c9fefcf3af5f15e240c7b18245abe49c44b781956b2b5351dac30e"
PAYLOAD = ROOT / "runs/metal-training-payload-1090"
MODULES = ("duo_deep", "duo_continuation", "duo_training_device", "duo_training_benchmark",
           "duo_conv", "duo_model", "duo_learning")
TIMEOUT, CAP, HOURLY = 900, 1.50, 2.5204
RESERVATION = HOURLY * (TIMEOUT + 90) / 3600 + .15
MODEL = "/data/duo-continuation-865/frozen-strong1024.pt"
STATE = "/data/duo-continuation-865/strong1024/resume.pt"
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)
app = modal.App("blokus-metal-training-comparison-1090")

if modal.is_local():
    image = modal.Image.from_id(BASE_IMAGE).add_local_file(
        __file__, "/opt/blokus/cloud/modal_metal_training.py", copy=True
    )
    for name in MODULES:
        image = image.add_local_file(
            PAYLOAD / (name + ".py"),
            "/opt/integrated/payload/python/blokus_rl/" + name + ".py", copy=True,
        )
    image = image.env({"PYTHONPATH": "/opt/blokus/cloud:/opt/integrated/payload/python"}).run_commands(
        "python -c 'import importlib.util, py_compile; "
        "s=importlib.util.find_spec(\"modal_metal_training\"); assert s is not None; "
        "py_compile.compile(s.origin,doraise=True); "
        "import blokus_rl.duo_training_benchmark; print(\"Training benchmark imports verified\")'"
    )
else:
    image = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image, gpu="L40S", cpu=(8, 8), memory=(24576, 24576),
    timeout=TIMEOUT, retries=0, max_containers=1, scaledown_window=2,
    include_source=False, volumes={"/data": volume},
)
def benchmark(execution):
    root = Path("/data") / RUN
    root.mkdir(exist_ok=False)
    (root / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    volume.commit()
    started = time.monotonic()
    try:
        if sha(__file__) != execution["runner_sha256"]:
            raise RuntimeError("runner changed")
        if sha(importlib.util.find_spec("blokus_rl._native").origin) != NATIVE_SHA:
            raise RuntimeError("production native binary changed")
        for name, expected in execution["payload_sha256"].items():
            if sha(importlib.util.find_spec("blokus_rl." + name).origin) != expected:
                raise RuntimeError("payload mismatch: " + name)
        for variant in ("eager", "production"):
            remaining = TIMEOUT - (time.monotonic() - started) - 30
            if remaining <= 0:
                raise TimeoutError("benchmark function budget exhausted")
            print("Starting fixed-work L40S " + variant, flush=True)
            command = [sys.executable, "-m", "blokus_rl.duo_training_benchmark",
                       "--device", "cuda", "--variant", variant,
                       "--model", MODEL, "--source-state", STATE,
                       "--out", str(root / variant), "--deadline", str(int(remaining)-10)]
            subprocess.run(command, check=True, timeout=remaining, env=dict(os.environ))
            volume.commit()
        (root / "finished.json").write_text(json.dumps({
            "seconds":time.monotonic()-started, "variants":["eager", "production"],
            "finish_utc":datetime.now(timezone.utc).isoformat(),
            "benchmark_optimizer_updates_discarded":True,
        }, indent=2) + "\n")
        volume.commit()
        return {"run":RUN, "seconds":time.monotonic()-started}
    except BaseException:
        (root / "failure.json").write_text(json.dumps({"traceback":traceback.format_exc()}) + "\n")
        volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-metal-training-cost-ledger.jsonl"
    if ledger.exists():
        raise ValueError("this comparison was already reserved; never automatically repeat")
    command = [sys.executable, "-m", "modal"]

    def cli(*args):
        return json.loads(subprocess.check_output(command + list(args) + ["--json", "--profile", PROFILE], text=True))

    apps, billing, rates = cli("app", "list"), cli("billing", "summary"), cli("billing", "rates")
    if any(int(row["tasks"]) for row in apps):
        raise ValueError("another Modal task is active")
    actual_rate = float(rates["gpu_hour_cost_l40s"]) + 8*float(rates["cpu_hour_cost"]) + 24*float(rates["mem_gib_hour_cost"])
    reserve = actual_rate * (TIMEOUT+90)/3600 + .15
    if abs(actual_rate-HOURLY) > 1e-6 or reserve > CAP or max(56, float(billing["metered_cost"]))+reserve > 80:
        raise ValueError("rates/budget changed; retain $50 of original $130 credits")
    execution = {
        "utc":datetime.now(timezone.utc).isoformat(), "run_name":RUN,
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "runner_sha256":sha(__file__), "base_image":BASE_IMAGE,
        "payload_sha256":{name:sha(PAYLOAD/(name+".py")) for name in MODULES},
        "timeout_seconds":TIMEOUT, "reserved_usd":reserve, "phase_cap_usd":CAP,
        "hourly_usd":actual_rate, "profile":PROFILE,
        "purpose":"discarded fixed-work optimizer/self-play timing; no strength test or new selected model",
        "source_only_uploads":True,
    }
    for name, value in (("apps-before", apps), ("billing-before", billing), ("rates", rates)):
        (ROOT / f"runs/metal-training-{name}.json").write_text(json.dumps(value,indent=2)+"\n")
    with ledger.open("x") as handle:
        handle.write(json.dumps(execution)+"\n")
    call = benchmark.spawn(execution)
    receipt = {"function_call_id":call.object_id, "run_name":RUN}
    (ROOT / "runs/metal-training-submission.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt), flush=True)
