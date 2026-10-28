"""Finish only the five uncommitted H100 comparison cases after preemption."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROFILE = "linuxfan2718"
PRIOR = "ap-pcbKy7PHqSUn8AsUyZG4xD"
IMAGE = "im-g024WcGyhkVNjjImEuEZ1k"
RUN = "duo-cuda-graphs-1069-r1"
MODEL = "/data/duo-continuation-865/frozen-strong1024.pt"
RESUME = "/data/duo-continuation-865/strong1024/resume.pt"
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
RESUME_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
ORDER = ("current-eager", "current-graph", "native-eager", "native-graph", "native-graph-wide")
image = (
    modal.Image.from_id(IMAGE).add_local_file(
        __file__, "/opt/blokus/cloud/modal_cuda_recovery.py", copy=True
    )
    if modal.is_local()
    else None
)
app = modal.App("blokus-frozen-cuda-h100-recovery")
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.function(
    image=image,
    gpu="H100!",
    cpu=(8, 8),
    memory=(24576, 24576),
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": volume},
)
def recover(execution):
    import torch

    source = Path("/data") / RUN / "h100"
    root = source.with_name("h100-recovery")
    # This exclusive directory guard also blocks Modal infrastructure restarts.
    root.mkdir(exist_ok=False)
    for name, h in execution["preserved_files"].items():
        assert sha(source / name) == h, name
    shutil.copytree(source, root, dirs_exist_ok=True)
    (root / "recovery-execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    initial = root / "initial-run"
    initial.mkdir()
    for name in ("phases.jsonl", "telemetry.jsonl", "failure.json"):
        (root / name).rename(initial / name)
    volume.commit()
    assert sha(MODEL) == MODEL_SHA and sha(RESUME) == RESUME_SHA
    spec = importlib.util.spec_from_file_location(
        "original_sampler", "/opt/blokus/cloud/modal_cuda_benchmark.py"
    )
    sampler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sampler
    spec.loader.exec_module(sampler)
    old = json.loads((root / "execution.json").read_text())
    assert sha("/opt/blokus/cloud/modal_cuda_benchmark.py") == old["runner_sha256"]
    for name, h in old["payload_sha256"].items():
        assert sha(Path("/opt/cg/payload") / name) == h
    (root / "recovery-hardware.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cpu_limit": 8,
                "memory_limit_gib": 24,
                "smi": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
                    text=True,
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )
    env = dict(os.environ, PYTHONPATH="/opt/cg/payload/optimized")
    env.pop("BLOKUS_NATIVE_PATH", None)
    env.pop("BLOKUS_BASELINE_EXTENSION", None)
    optimized = dict(env, BLOKUS_NATIVE_PATH="/opt/cg/native-extension")
    command = [
        sys.executable,
        "/opt/cg/payload/duo_cuda_benchmark.py",
        "--model",
        MODEL,
        "--fixture",
        str(root / "inputs.npz"),
    ]
    started = time.monotonic()
    try:
        with sampler.Telemetry(root / "telemetry.jsonl"):
            check = (
                """import importlib.util,json,torch
s=importlib.util.spec_from_file_location('bench','/opt/cg/payload/duo_cuda_benchmark.py')
b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
b.configure_torch(1069);torch.set_float32_matmul_precision('high')
m=b.load_duo(MODEL).cuda().eval()
open(OUT,'w').write(json.dumps(b.correctness(m,b.fixture(FIXTURE)),indent=2)+'\\n')
""".replace("MODEL", repr(MODEL))
                .replace("OUT", repr(str(root / "recovery-correctness.json")))
                .replace("FIXTURE", repr(str(root / "inputs.npz")))
            )
            subprocess.run([sys.executable, "-c", check], env=env, check=True, timeout=60)
            for variant in reversed(ORDER):
                tag = "2-" + variant
                out = root / (tag + ".json")
                assert not out.exists()
                print("Recovery collection " + tag, flush=True)
                subprocess.run(
                    command
                    + ["--variant", variant, "--repeat", "2", "--tag", tag, "--out", str(out)],
                    env=optimized if variant.startswith("native") else env,
                    check=True,
                    timeout=160,
                )
                volume.commit()
        for name, h in execution["preserved_files"].items():
            target = (
                initial / name
                if name in ("phases.jsonl", "telemetry.jsonl", "failure.json")
                else root / name
            )
            assert sha(target) == h, name
        assert sha(MODEL) == MODEL_SHA and sha(RESUME) == RESUME_SHA
        elapsed = time.monotonic() - started
        finish = {
            "gpu_type": "h100",
            "gradient_updates": 0,
            "model_sha256": MODEL_SHA,
            "source_resume_sha256": RESUME_SHA,
            "seconds": elapsed,
            "seconds_scope": "recovery function only; original app preempted",
            "preserved_case_count": 5,
            "recovered_case_count": 5,
            "finish_utc": datetime.now(timezone.utc).isoformat(),
            "results": [
                json.loads((root / f"{repeat}-{variant}.json").read_text())
                for repeat, variants in ((1, ORDER), (2, tuple(reversed(ORDER))))
                for variant in variants
            ],
        }
        (root / "finished.json").write_text(json.dumps(finish, indent=2) + "\n")
        volume.commit()
        return {"seconds": elapsed}
    except BaseException:
        import traceback

        (root / "recovery-failure.json").write_text(
            json.dumps({"traceback": traceback.format_exc()}) + "\n"
        )
        volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-cuda-h100-recovery-cost-ledger.jsonl"
    assert not ledger.exists(), "one recovery only"
    base = [sys.executable, "-m", "modal"]
    apps = json.loads(
        subprocess.check_output(base + ["app", "list", "--json", "--profile", PROFILE], text=True)
    )
    assert all(int(a["tasks"]) == 0 for a in apps)
    assert any(a["app_id"] == PRIOR and a["state"] == "stopped" for a in apps)
    rates = json.loads(
        subprocess.check_output(
            base + ["billing", "rates", "--json", "--profile", PROFILE], text=True
        )
    )
    billing = json.loads(
        subprocess.check_output(
            base
            + [
                "billing",
                "report",
                "--start",
                "2026-10-28",
                "--end",
                "2026-10-29",
                "--resolution",
                "h",
                "--show-resources",
                "--json",
                "--profile",
                PROFILE,
            ],
            text=True,
        )
    )
    costs = {}
    for r in billing:
        costs[r["object_id"]] = costs.get(r["object_id"], 0) + float(r["cost"])
    hourly = (
        float(rates["gpu_hour_cost_h100"])
        + 8 * float(rates["cpu_hour_cost"])
        + 24 * float(rates["mem_gib_hour_cost"])
    )
    reserved = hourly * 990 / 3600 + 0.025
    assert costs[PRIOR] + reserved <= hourly * 1890 / 3600, "original H100 reservation"
    ids = [PRIOR, "ap-Chtgg9VvtUYBSNzcOXV1az", "ap-yGUCk4BBAXkUA1RrJmyJze"]
    actual = sum(costs[i] for i in ids)
    summary = json.loads(
        subprocess.check_output(
            base + ["billing", "summary", "--json", "--profile", PROFILE], text=True
        )
    )
    assert actual + reserved < 5 and max(45, float(summary["metered_cost"])) + reserved <= 80
    source = ROOT / "runs/cuda-graphs-h100-preempted-20261028T051312Z/h100"
    assert not (source / "finished.json").exists()
    assert all((source / ("1-" + v + ".json")).exists() for v in ORDER)
    assert not any(source.glob("2-*.json"))
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "prior_app_id": PRIOR,
        "prior_actual_usd": costs[PRIOR],
        "reserved_usd": reserved,
        "settled_phase_actual_usd": actual,
        "phase_cap_usd": 5,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha(__file__),
        "image": IMAGE,
        "timeout_seconds": 900,
        "preserved_files": {
            str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()
        },
    }
    ledger.write_text(json.dumps(execution) + "\n")
    call = recover.spawn(execution)
    receipt = {"function_call_id": call.object_id, "run": RUN + "/h100-recovery"}
    (ROOT / "runs/cuda-h100-recovery-submission.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
