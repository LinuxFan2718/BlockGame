"""Conditional final H100 coalescing-delay probe within the original $5 cap."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROFILE = "linuxfan2718"
RUN = "duo-cuda-latency-1069"
IMAGE = "im-g024WcGyhkVNjjImEuEZ1k"
MODEL = "/data/duo-continuation-865/frozen-strong1024.pt"
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)
image = (
    modal.Image.from_id(IMAGE)
    .add_local_file(__file__, "/opt/blokus/cloud/modal_cuda_latency.py", copy=True)
    .add_local_file(
        ROOT / "cloud/duo_cuda_layout_probe.py",
        "/opt/blokus/cloud/duo_cuda_layout_probe.py",
        copy=True,
    )
    if modal.is_local()
    else None
)
app = modal.App("blokus-frozen-cuda-latency")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.function(
    image=image,
    gpu="H100!",
    cpu=(8, 8),
    memory=(24576, 24576),
    timeout=780,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": volume},
)
def evaluate(execution):
    import importlib.util
    import os
    import shutil

    import torch

    # Import only the tested sampler class from the immutable comparison image.
    spec = importlib.util.spec_from_file_location(
        "saved_sampler", "/opt/blokus/cloud/modal_cuda_benchmark.py"
    )
    sampler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sampler
    spec.loader.exec_module(sampler)
    root = Path("/data") / RUN
    root.mkdir(exist_ok=False)
    assert sha(MODEL) == MODEL_SHA
    source_fixture = Path("/data/duo-cuda-graphs-1069-r1/h100/inputs.npz")
    assert sha(source_fixture) == execution["fixture_sha256"]
    shutil.copy2(source_fixture, root / "inputs.npz")
    (root / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    (root / "hardware.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "smi": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
                    text=True,
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )
    volume.commit()
    variables = dict(os.environ)
    variables["PYTHONPATH"] = "/opt/cg/payload/optimized"
    variables["BLOKUS_NATIVE_PATH"] = "/opt/cg/native-extension"
    variables.pop("BLOKUS_BASELINE_EXTENSION", None)
    started = time.monotonic()
    results = []
    helper = "/opt/blokus/cloud/duo_cuda_layout_probe.py"
    assert sha(helper) == execution["helper_sha256"]
    command = [sys.executable, helper, "--model", MODEL, "--fixture", str(root / "inputs.npz")]
    try:
        with sampler.Telemetry(root / "telemetry.jsonl"):
            layout_ok = False
            try:
                subprocess.run(
                    command + ["--mode", "check", "--out", str(root / "layout-check.json")],
                    env=variables,
                    check=True,
                    timeout=45,
                )
                layout_ok = json.loads((root / "layout-check.json").read_text())["passed"]
                (root / "nchw-micro").mkdir()
                subprocess.run(
                    command + ["--mode", "micro", "--out", str(root / "nchw-micro/micro.json")],
                    env=variables,
                    check=True,
                    timeout=120,
                )
                if layout_ok:
                    (root / "channels-last-micro").mkdir()
                    subprocess.run(
                        command
                        + [
                            "--mode",
                            "micro",
                            "--layout",
                            "channels_last",
                            "--out",
                            str(root / "channels-last-micro/micro.json"),
                        ],
                        env=variables,
                        check=True,
                        timeout=120,
                    )
            except subprocess.SubprocessError as error:
                (root / "layout-failure.json").write_text(json.dumps({"error": str(error)}) + "\n")
                layout_ok = False
            order = ((1.0, "nchw"), (0.1, "nchw"), (0.0, "nchw"))
            if layout_ok:
                order += ((0.1, "channels_last"),)
            for repeat, settings in ((3, order), (4, tuple(reversed(order)))):
                for delay, layout in settings:
                    tag = f"{repeat}-{layout}-wait-{delay}"
                    target = root / (tag + ".json")
                    print("Latency " + tag, flush=True)
                    subprocess.run(
                        command
                        + [
                            "--mode",
                            "collection",
                            "--layout",
                            layout,
                            "--delay",
                            str(delay),
                            "--repeat",
                            str(repeat),
                            "--tag",
                            tag,
                            "--out",
                            str(target),
                        ],
                        env=variables,
                        check=True,
                        timeout=105,
                    )
                    results.append(json.loads(target.read_text()))
                    volume.commit()
        assert sha(MODEL) == MODEL_SHA
        finish = {
            "results": results,
            "seconds": time.monotonic() - started,
            "gradient_updates": 0,
            "model_sha256": MODEL_SHA,
            "channels_last_enabled": layout_ok,
            "finish_utc": datetime.now(timezone.utc).isoformat(),
        }
        (root / "finished.json").write_text(json.dumps(finish, indent=2) + "\n")
        volume.commit()
        return {"seconds": finish["seconds"]}
    except BaseException:
        import traceback

        (root / "failure.json").write_text(json.dumps({"traceback": traceback.format_exc()}) + "\n")
        volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-cuda-latency-cost-ledger.jsonl"
    if ledger.exists():
        raise ValueError("final latency probe already submitted")
    reports = [
        json.loads((ROOT / f"runs/cuda-graphs-{gpu}-verified.json").read_text())
        for gpu in ("l40s", "h100")
    ]
    if not all(r["verified"] for r in reports):
        raise ValueError("complete both comparisons first")
    command = [sys.executable, "-m", "modal"]
    apps = json.loads(
        subprocess.check_output(
            command + ["app", "list", "--json", "--profile", PROFILE], text=True
        )
    )
    if any(int(a["tasks"]) for a in apps):
        raise ValueError("compute still active")
    rates = json.loads(
        subprocess.check_output(
            command + ["billing", "rates", "--json", "--profile", PROFILE], text=True
        )
    )
    billing = json.loads(
        subprocess.check_output(
            command
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
    ids = {name for r in reports for name in r.get("app_ids", [r["app_id"]])} | {
        "ap-yGUCk4BBAXkUA1RrJmyJze"
    }
    actual = sum(float(r["cost"]) for r in billing if r["object_id"] in ids)
    hourly = (
        float(rates["gpu_hour_cost_h100"])
        + 8 * float(rates["cpu_hour_cost"])
        + 24 * float(rates["mem_gib_hour_cost"])
    )
    reserved = hourly * 870 / 3600 + 0.05
    summary = json.loads(
        subprocess.check_output(
            command + ["billing", "summary", "--json", "--profile", PROFILE], text=True
        )
    )
    if actual + reserved > 5 or max(45, float(summary["metered_cost"])) + reserved > 80:
        raise ValueError("original $5 phase cap / $50 reserve cannot fund this probe")
    h100 = Path(reports[1]["run"])
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "runner_sha256": sha(__file__),
        "image": IMAGE,
        "fixture_sha256": sha(h100 / "inputs.npz"),
        "helper_sha256": sha(ROOT / "cloud/duo_cuda_layout_probe.py"),
        "prior_apps": sorted(ids),
        "prior_actual_usd": actual,
        "prior_unused_reservation_settled_usd": 4.19642 - actual,
        "reserved_usd": reserved,
        "combined_committed_usd": actual + reserved,
        "combined_cap_usd": 5,
        "hourly_usd": hourly,
        "model_sha256": MODEL_SHA,
        "timeout_seconds": 780,
        "settings": "same frozen native+graph 384-game workload, live64/full1024/fast32; compare wait1/.1/0ms NCHW, plus .1ms channels-last if output tolerances pass; repeats3/4 reversed",
    }
    ledger.write_text(json.dumps(execution) + "\n")
    call = evaluate.spawn(execution)
    receipt = {"function_call_id": call.object_id, "run": RUN}
    (ROOT / "runs/cuda-latency-submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
