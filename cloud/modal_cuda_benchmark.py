"""One serial L40S/H100 frozen CUDA Graph evaluation, bounded by $5 total."""

import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "duo-cuda-graphs-1069-r1"
PROFILE = "linuxfan2718"
VOLUME = "blokus-deep-selfplay-20261026"
BASE_IMAGE = "im-BnyLYweX6vlwu6iCpDnqrC"
MODEL = "/data/duo-continuation-865/frozen-strong1024.pt"
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
RESUME = "/data/duo-continuation-865/strong1024/resume.pt"
RESUME_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
PAYLOAD = ROOT / "runs/duo-cuda-payload-r1"
TIMEOUT = 1800
CAP = 5.0
RESERVATION = 4.19642
image = (
    modal.Image.from_id(BASE_IMAGE)
    .add_local_file(__file__, "/opt/blokus/cloud/modal_cuda_benchmark.py", copy=True)
    .add_local_dir(PAYLOAD, "/opt/cg/payload", copy=True)
    .run_commands("python /opt/cg/payload/build_cuda_native.py")
    if modal.is_local()
    else None
)
app = modal.App("blokus-frozen-cuda-graphs")
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Telemetry:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            row = {"monotonic": time.monotonic(), "utc": datetime.now(timezone.utc).isoformat()}
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=3,
                )
                row["gpu"] = out.strip()
                for p in (
                    "/sys/fs/cgroup/cpu/cpuacct.usage",
                    "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
                    "/sys/fs/cgroup/cpuacct/cpuacct.usage",
                ):
                    if Path(p).exists():
                        row["cpu_ns"] = int(Path(p).read_text())
                        break
                if "cpu_ns" not in row and Path("/sys/fs/cgroup/cpu.stat").exists():
                    fields = dict(
                        line.split()
                        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines()
                    )
                    row["cpu_ns"] = int(fields["usage_usec"]) * 1000
                for p in (
                    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                    "/sys/fs/cgroup/memory.current",
                ):
                    if Path(p).exists():
                        row["memory_bytes"] = int(Path(p).read_text())
                        break
            except Exception as error:
                row["error"] = str(error)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.stop.wait(0.4)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


def run_stage(gpu, execution):
    import gc
    import os
    import shutil

    import numpy as np
    import torch

    root = Path("/data") / RUN / gpu
    root.mkdir(parents=True, exist_ok=False)
    (root / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    volume.commit()
    assert sha(MODEL) == MODEL_SHA and sha(RESUME) == RESUME_SHA
    for name, h in execution["payload_sha256"].items():
        assert sha(Path("/opt/cg/payload") / name) == h
    shutil.copy2("/opt/cg/native-build.json", root / "native-build.json")
    props = torch.cuda.get_device_properties(0)
    hardware = {
        "gpu": torch.cuda.get_device_name(),
        "memory_bytes": props.total_memory,
        "multiprocessors": props.multi_processor_count,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cpu_limit": 8,
        "memory_limit_gib": 24,
        "smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }
    (root / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    saved = torch.load(RESUME, map_location="cpu", weights_only=False)
    replay = saved["replay"]
    selected = np.random.default_rng(1069).choice(len(replay), 2048, replace=False)
    rows = [replay[int(i)] for i in selected]
    np.savez_compressed(
        root / "inputs.npz",
        observations=np.stack([r["observation"] for r in rows]),
        indices=np.concatenate([r["indices"] for r in rows]).astype(np.int64),
        offsets=np.cumsum([0] + [len(r["indices"]) for r in rows]),
        selected_positions=selected,
    )
    del saved, replay, rows
    gc.collect()
    env = dict(os.environ)
    env["PYTHONPATH"] = "/opt/cg/payload/optimized"
    env.pop("BLOKUS_NATIVE_PATH", None)
    env.pop("BLOKUS_BASELINE_EXTENSION", None)
    optimized_env = dict(env)
    optimized_env["BLOKUS_NATIVE_PATH"] = "/opt/cg/native-extension"
    started = time.monotonic()
    try:
        with Telemetry(root / "telemetry.jsonl"):
            subprocess.run(
                [
                    sys.executable,
                    "/opt/cg/payload/duo_cuda_native_check.py",
                    str(root / "native-correctness.json"),
                ],
                env=optimized_env,
                check=True,
                timeout=90,
            )
            command = [
                sys.executable,
                "/opt/cg/payload/duo_cuda_benchmark.py",
                "--model",
                MODEL,
                "--fixture",
                str(root / "inputs.npz"),
            ]
            subprocess.run(
                command + ["--out", str(root / "micro.json")], env=env, check=True, timeout=240
            )
            order = (
                "current-eager",
                "current-graph",
                "native-eager",
                "native-graph",
                "native-graph-wide",
            )
            results = []
            for repeat, variants in ((1, order), (2, tuple(reversed(order)))):
                for variant in variants:
                    tag = f"{repeat}-{variant}"
                    out = root / (tag + ".json")
                    print("Collection " + gpu + " " + tag, flush=True)
                    subprocess.run(
                        command
                        + [
                            "--variant",
                            variant,
                            "--repeat",
                            str(repeat),
                            "--tag",
                            tag,
                            "--out",
                            str(out),
                        ],
                        env=optimized_env if variant.startswith("native") else env,
                        check=True,
                        timeout=160,
                    )
                    results.append(json.loads(out.read_text()))
                    volume.commit()
        assert sha(MODEL) == MODEL_SHA and sha(RESUME) == RESUME_SHA
        finish = {
            "gpu_type": gpu,
            "results": results,
            "seconds": time.monotonic() - started,
            "gradient_updates": 0,
            "model_sha256": MODEL_SHA,
            "source_resume_sha256": RESUME_SHA,
            "finish_utc": datetime.now(timezone.utc).isoformat(),
        }
        (root / "finished.json").write_text(json.dumps(finish, indent=2) + "\n")
        volume.commit()
        return {"gpu": gpu, "seconds": finish["seconds"]}
    except BaseException:
        import traceback

        (root / "failure.json").write_text(json.dumps({"traceback": traceback.format_exc()}) + "\n")
        volume.commit()
        raise


options = dict(
    image=image,
    cpu=(8, 8),
    memory=(24576, 24576),
    timeout=TIMEOUT,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": volume},
)


@app.function(gpu="L40S", **options)
def l40s(execution):
    return run_stage("l40s", execution)


@app.function(gpu="H100!", **options)
def h100(execution):
    return run_stage("h100", execution)


@app.local_entrypoint()
def main(gpu: str = "l40s"):
    if gpu not in ("l40s", "h100"):
        raise ValueError("only planned GPUs")
    ledger = ROOT / f"runs/modal-cuda-graphs-{gpu}-r1-cost-ledger.jsonl"
    if ledger.exists():
        raise ValueError("stage already submitted; no duplicate")
    if gpu == "h100" and not (ROOT / "runs/cuda-graphs-l40s-verified.json").exists():
        raise ValueError("verify stopped L40S stage first")
    base = [sys.executable, "-m", "modal"]
    apps = json.loads(
        subprocess.check_output(base + ["app", "list", "--json", "--profile", PROFILE], text=True)
    )
    if any(int(a["tasks"]) for a in apps):
        raise ValueError("existing active compute")
    stopped_attempt = json.loads(
        (ROOT / "runs/duo-cuda-l40s-interrupted-20261028T043915Z/apps-stopped.json").read_text()
    )
    if not any(
        a["app_id"] == "ap-yGUCk4BBAXkUA1RrJmyJze"
        and a["state"] == "stopped"
        and int(a["tasks"]) == 0
        for a in stopped_attempt
    ):
        raise ValueError("preserve stopped original attempt first")
    billing = json.loads(
        subprocess.check_output(
            base + ["billing", "summary", "--json", "--profile", PROFILE], text=True
        )
    )
    rates = json.loads(
        subprocess.check_output(
            base + ["billing", "rates", "--json", "--profile", PROFILE], text=True
        )
    )
    combined = (
        sum(
            (
                float(rates["gpu_hour_cost_" + g])
                + 8 * float(rates["cpu_hour_cost"])
                + 24 * float(rates["mem_gib_hour_cost"])
            )
            * 1890
            / 3600
            for g in ("l40s", "h100")
        )
        + 0.5
    )
    if (
        combined > RESERVATION + 0.000001
        or RESERVATION > CAP
        or max(45, float(billing["metered_cost"])) + RESERVATION > 80
    ):
        raise ValueError("cost/reserve cap violated")
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run": RUN,
        "gpu": gpu,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "runner_sha256": sha(__file__),
        "payload_sha256": {
            str(p.relative_to(PAYLOAD)): sha(p) for p in sorted(PAYLOAD.rglob("*")) if p.is_file()
        },
        "combined_reserved_usd": RESERVATION,
        "previous_attempt_app_id": "ap-yGUCk4BBAXkUA1RrJmyJze",
        "previous_attempt_metered_usd": "0.19414255",
        "reservation_detail": "3.69642 for both remaining GPU hard limits, .25 for stopped attempt, .25 build allowance; shared reservation, not additive per ledger",
        "combined_cap_usd": CAP,
        "timeout_seconds_per_gpu": TIMEOUT,
        "hourly_usd": float(rates["gpu_hour_cost_" + gpu])
        + 8 * float(rates["cpu_hour_cost"])
        + 24 * float(rates["mem_gib_hour_cost"]),
        "rates": rates,
        "billing_before": billing,
        "base_image": BASE_IMAGE,
        "model_sha256": MODEL_SHA,
        "source_resume_sha256": RESUME_SHA,
        "gradient_updates": 0,
        "protocol": {
            "games_per_case": 384,
            "warmup_games_per_case": 12,
            "repeats": 2,
            "full_simulations": 1024,
            "fast_simulations": 32,
            "workers": 6,
            "live_games": [32, 64],
            "precision": "bfloat16",
            "seed": 1069,
        },
    }
    ledger.write_text(json.dumps(execution) + "\n")
    call = (l40s if gpu == "l40s" else h100).spawn(execution)
    receipt = {"function_call_id": call.object_id, "gpu": gpu, "run": RUN}
    (ROOT / f"runs/cuda-graphs-{gpu}-submission.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
