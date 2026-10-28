"""One independently seeded control replication; four-hour function and $11 cap."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROFILE = "linuxfan2718"
VOLUME = "blokus-deep-selfplay-20261026"
BASE_IMAGE = "im-yX3zmnEKl2EgUtndThw9li"
PAYLOAD = ROOT / "runs/integrated-payload-1070"
MANIFEST = ROOT / "runs/integrated-payload-1070-sha256.json"
CAP = 11.0
HOURLY = 2.5204
NAME = "duo-control-replication-1071"
TIMEOUT = 14400
HELPER = ROOT / "cloud/run_control_replication.py"
PROTOCOL = ROOT / "docs/control-replication-protocol.json"
image = (
    modal.Image.from_id(BASE_IMAGE)
    .add_local_file(__file__, "/opt/blokus/cloud/modal_control_replication.py", copy=True)
    .add_local_file(HELPER, "/opt/blokus/cloud/run_control_replication.py", copy=True)
    .env({"PYTHONPATH": "/opt/blokus/cloud"})
    .run_commands(
        'python -c \'import importlib.util, py_compile; s = importlib.util.find_spec("modal_control_replication"); assert s is not None; py_compile.compile(s.origin, doraise=True); py_compile.compile("/opt/blokus/cloud/run_control_replication.py", doraise=True); print("Replication source discovery verified")\'',
    )
    if modal.is_local()
    else None
)
app = modal.App("blokus-control-replication-1071")
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Telemetry:
    def __init__(self, path):
        self.path, self.stop = path, threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            row = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic()}
            try:
                row["gpu"] = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=3,
                ).strip()
                for p in (
                    "/sys/fs/cgroup/cpuacct/cpuacct.usage",
                    "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
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
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            self.stop.wait(5)

    def __enter__(self):
        self.thread.start()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


def run(execution):
    import torch

    root = Path("/data") / NAME
    root.mkdir(exist_ok=False)  # Refuse an infrastructure replay or duplicate.
    started = time.monotonic()
    (root / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    volume.commit()
    try:
        assert sha("/opt/blokus/cloud/modal_control_replication.py") == execution["runner_sha256"]
        assert sha("/opt/blokus/cloud/run_control_replication.py") == execution["helper_sha256"]
        for name, expected in execution["payload_sha256"].items():
            assert sha(Path("/opt/integrated/payload") / name) == expected, name
        for name, expected in execution["immutable_modules"].items():
            assert sha(importlib.util.find_spec("blokus_rl." + name).origin) == expected, name
        native = next(Path("/opt/integrated/native-extension").glob("_native.*.so"))
        assert sha(native) == execution["native_sha256"]
        shutil.copy2("/opt/integrated/native-build.json", root / "native-build.json")
        (root / "hardware.json").write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                    "cpu_cap": 8,
                    "ram_gib_cap": 24,
                    "smi": subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,driver_version,uuid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).strip(),
                },
                indent=2,
            )
            + "\n"
        )
        volume.commit()
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                ["/opt/integrated/payload/python", *[p for p in sys.path if p]]
            ),
        )
        env.pop("BLOKUS_NATIVE_PATH", None)
        env.pop("BLOKUS_BASELINE_EXTENSION", None)
        # Exercise the exact child SDK path and mounted-volume commit before science.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import modal; v = modal.Volume.from_name('blokus-deep-selfplay-20261026', create_if_missing=False); v.commit(); print('Child SDK and Volume commit verified', flush=True)",
            ],
            env=env,
            check=True,
            timeout=30,
        )
        command = [
            sys.executable,
            "/opt/blokus/cloud/run_control_replication.py",
            "--out",
            str(root),
            "--execution",
            str(root / "execution.json"),
        ]
        # All children are in this function's container; the function hard limit bounds spend.
        with Telemetry(root / "telemetry.jsonl"):
            subprocess.run(command, env=env, check=True, timeout=TIMEOUT - 60)
        marker = root / "finished.json"
        assert marker.exists()
        (root / "wrapper-finished.json").write_text(
            json.dumps(
                {
                    "seconds": time.monotonic() - started,
                    "finish_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
        volume.commit()
        return {"run": NAME, "seconds": time.monotonic() - started}
    except BaseException:
        (root / "failure.json").write_text(
            json.dumps(
                {"traceback": traceback.format_exc(), "seconds": time.monotonic() - started},
                indent=2,
            )
            + "\n"
        )
        volume.commit()
        raise


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
    return run(execution)


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-control-replication-cost-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    if rows:
        raise ValueError("stage already reserved; no duplicate or automatic retry")
    reservation = HOURLY * (TIMEOUT + 90) / 3600 + 0.15
    if sum(row["reserved_usd"] for row in rows) + reservation > CAP:
        raise ValueError("$11 phase ceiling exceeded")
    command = [sys.executable, "-m", "modal"]

    def cli(*args):
        return json.loads(
            subprocess.check_output(
                command + list(args) + ["--json", "--profile", PROFILE], text=True
            )
        )

    apps, billing, rates = cli("app", "list"), cli("billing", "summary"), cli("billing", "rates")
    if any(
        int(row["tasks"]) or row["state"].lower() not in ("stopped",)
        for row in apps
        if row["app_id"] != app.app_id
    ):
        raise ValueError("active apps; inspect before launch")
    actual_rate = (
        float(rates["gpu_hour_cost_l40s"])
        + 8 * float(rates["cpu_hour_cost"])
        + 24 * float(rates["mem_gib_hour_cost"])
    )
    if abs(actual_rate - HOURLY) > 1e-8:
        raise ValueError("rates changed")
    # Count prior commitments conservatively even when provider billing is delayed.
    committed = (
        max(54.0, float(billing["metered_cost"]))
        + sum(row["reserved_usd"] for row in rows)
        + reservation
    )
    if committed > 80:
        raise ValueError("preserve $50 of original $130 credits")
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest.items():
        assert sha(PAYLOAD / name) == expected, name
    record = json.loads((ROOT / "runs/integrated-validation-verified.json").read_text())
    if not record["passed"] or record["payload_sha256"] != manifest:
        raise ValueError("no passing validation of this exact inherited payload")
    selected = record["selected_runtime"]
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["run_name"] == NAME and protocol["config"]["seed"] == 1071
    preserved = json.loads((ROOT / "runs/control-replication-preserved-before.json").read_text())
    for name, expected in preserved.items():
        assert sha(ROOT / name) == expected, name
    execution = {
        "stage": "replication",
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_name": NAME,
        "profile": PROFILE,
        "reserved_usd": reservation,
        "phase_ceiling_usd": CAP,
        "timeout_seconds": TIMEOUT,
        "hourly_usd": HOURLY,
        "base_image": BASE_IMAGE,
        "runner_sha256": sha(__file__),
        "payload_sha256": manifest,
        "helper_sha256": sha(HELPER),
        "protocol_sha256": sha(PROTOCOL),
        "config": protocol["config"],
        "native_sha256": "257b5a55c2c9fefcf3af5f15e240c7b18245abe49c44b781956b2b5351dac30e",
        "preserved_sha256": sha(ROOT / "runs/control-replication-preserved-before.json"),
        "immutable_modules": {
            name: sha(ROOT / f"python/blokus_rl/{name}.py")
            for name in ("duo_conv", "duo_residual", "duo_learning", "learning", "duo_crossplay")
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "selected_runtime": selected,
        "account_commitment_ceiling_usd": committed,
        "uploads": "only modal_control_replication.py and run_control_replication.py; inherited validated image; no model/replay or unrelated files",
        "authorization": "Dennis: Please continue with your recommendation — independently seeded replication using cheaper1024 baseline; $11 phase ceiling and $50 reserve",
    }
    prefix = ROOT / "runs" / NAME
    Path(str(prefix) + "-runner.py").write_bytes(Path(__file__).read_bytes())
    Path(str(prefix) + "-helper.py").write_bytes(HELPER.read_bytes())
    Path(str(prefix) + "-billing-before.json").write_text(json.dumps(billing, indent=2) + "\n")
    Path(str(prefix) + "-apps-before.json").write_text(json.dumps(apps, indent=2) + "\n")
    Path(str(prefix) + "-rates.json").write_text(json.dumps(rates, indent=2) + "\n")
    with ledger.open("a") as handle:
        handle.write(json.dumps(execution) + "\n")
    call = study.spawn(execution)
    receipt = {"run_name": NAME, "function_call_id": call.object_id, "submitted": True}
    Path(str(prefix) + "-submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
