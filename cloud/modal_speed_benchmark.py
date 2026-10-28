"""One bounded fixed-work GPU benchmark of the requested implementation changes."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "duo-speed-benchmark-1067"
VOLUME = "blokus-deep-selfplay-20261026"
BASE_IMAGE = "im-BnyLYweX6vlwu6iCpDnqrC"
MODEL = "/data/duo-continuation-865/frozen-strong1024.pt"
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
MODULES = ("duo_deep", "duo_shared", "duo_mcts", "duo_indices", "duo_inference", "duo_replay")
TIMEOUT, CAP = 1200, 1.25
HOURLY = 2.5204
PAYLOAD = ROOT / "runs/duo-speed-benchmark-payload"
image = (
    modal.Image.from_id(BASE_IMAGE)
    .add_local_file(__file__, "/opt/blokus/cloud/modal_speed_benchmark.py", copy=True)
    .add_local_dir(PAYLOAD, "/opt/speed", copy=True)
    if modal.is_local()
    else None
)
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
app = modal.App("blokus-fixed-work-speed-benchmark")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
def benchmark(execution):
    import importlib.util
    import os

    root = Path("/data") / RUN
    if root.exists():
        raise RuntimeError("benchmark already exists; no automatic retry")
    if digest(MODEL) != MODEL_SHA:
        raise RuntimeError("private frozen checkpoint changed")
    for name, expected in execution["baseline_module_sha256"].items():
        if digest(importlib.util.find_spec("blokus_rl." + name).origin) != expected:
            raise RuntimeError("original baseline implementation changed")
    for name, expected in execution["payload_sha256"].items():
        if digest(Path("/opt/speed") / name) != expected:
            raise RuntimeError("benchmark payload changed")
    root.mkdir()
    (root / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    volume.commit()
    started, results = time.monotonic(), []
    order = ("baseline", "lookup-queue", "shared", "shared-cache")
    try:
        for repeat, variants in ((1, order), (2, tuple(reversed(order)))):
            for variant in variants:
                out = root / f"{repeat}-{variant}.json"
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                if variant != "baseline":
                    env["PYTHONPATH"] = "/opt/speed/optimized"
                command = [
                    sys.executable,
                    "/opt/speed/duo_speed_benchmark.py",
                    "--variant",
                    variant,
                    "--model",
                    MODEL,
                    "--out",
                    str(out),
                    "--round",
                    str(repeat),
                ]
                print("Benchmark " + str(repeat) + " " + variant, flush=True)
                subprocess.run(command, env=env, check=True, timeout=150)
                results.append(json.loads(out.read_text()))
                volume.commit()
        if digest(MODEL) != MODEL_SHA:
            raise RuntimeError("preserved parent changed")
        summary = {
            "seconds": time.monotonic() - started,
            "results": results,
            "model_sha256": MODEL_SHA,
            "gradient_updates": 0,
            "finish_utc": datetime.now(timezone.utc).isoformat(),
        }
        (root / "finished.json").write_text(json.dumps(summary, indent=2) + "\n")
        volume.commit()
        return {"run": RUN, "seconds": summary["seconds"]}
    except BaseException:
        import traceback

        (root / "failure.json").write_text(json.dumps({"traceback": traceback.format_exc()}) + "\n")
        volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-speed-benchmark-cost-ledger.jsonl"
    if ledger.exists():
        raise ValueError("one bounded speed benchmark already submitted")
    command = [sys.executable, "-m", "modal"]
    apps = json.loads(
        subprocess.check_output(
            command + ["app", "list", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    if any(int(a["tasks"]) for a in apps):
        raise ValueError("existing active compute; inspect before benchmark")
    billing = json.loads(
        subprocess.check_output(
            command + ["billing", "summary", "--json", "--profile", "linuxfan2718"], text=True
        )
    )
    reserved = HOURLY * (TIMEOUT + 90) / 3600
    if reserved > CAP or max(45, float(billing["metered_cost"])) + reserved > 80:
        raise ValueError("preserve original $130 budget and $50 reserve")
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "reserved_usd": reserved,
        "phase_cap_usd": CAP,
        "timeout_seconds": TIMEOUT,
        "profile": "linuxfan2718",
        "base_image": BASE_IMAGE,
        "model_sha256": MODEL_SHA,
        "gradient_updates": 0,
        "runner_sha256": digest(__file__),
        "baseline_module_sha256": {
            Path(path).stem: sha
            for path, sha in json.loads(
                (
                    ROOT
                    / Path("runs/duo-teacher-user-stopped-current.txt").read_text().strip()
                    / "original-source/sha256.json"
                ).read_text()
            ).items()
            if Path(path).parent.name == "blokus_rl"
        },
        "payload_sha256": {
            str(p.relative_to(PAYLOAD)): digest(p)
            for p in sorted(PAYLOAD.rglob("*"))
            if p.is_file()
        },
        "purpose": "fixed frozen-policy implementation benchmark; no training or strength claim",
    }
    ledger.write_text(json.dumps(execution) + "\n")
    call = benchmark.spawn(execution)
    receipt = {"function_call_id": call.object_id, "run_name": RUN}
    (ROOT / "runs/duo-speed-benchmark-submission.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
