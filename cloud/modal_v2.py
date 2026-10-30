"""Modal functions for the v2 agent: Pentobi data on CPU, throughput benchmarks
and self-play training on GPU. Results persist in the `blokus-v2` Volume.

Spend only Modal credits: every launch goes through `cloud/v2_budget.py`, which
refuses work whose worst-case cost would push total metered spend past the cap.

    .venv-modal/bin/modal run cloud/modal_v2.py::pentobi_data --games 2000 ...
    .venv-modal/bin/modal run cloud/modal_v2.py::benchmark --gpu L4 --cpu 16
    .venv-modal/bin/modal run --detach cloud/modal_v2.py::train_remote --run NAME ...
"""

import io
import json
import os
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VOLUME = "blokus-v2"
APP = "blokus-v2"


def source_archive():
    """Only the files needed to build and run the package."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md"):
            archive.add(ROOT / name, arcname=name)
        for directory in ("src", "bindings/python/src", "python/blokus_rl"):
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".so":
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
        for name in ("bindings/python/Cargo.toml", "bindings/python/Cargo.lock"):
            archive.add(ROOT / name, arcname=name)
    path = Path("/tmp/blokus-v2-source.tar.gz")
    path.write_bytes(buffer.getvalue())
    return path


if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "curl", "ca-certificates", "cmake", "git")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y "
            "--profile minimal --default-toolchain 1.91.1",
            "git clone --depth 1 --branch v31.0 https://github.com/enz/pentobi.git /opt/pentobi-source",
            "cmake -S /opt/pentobi-source -B /opt/pentobi-build -DCMAKE_BUILD_TYPE=Release "
            "-DPENTOBI_BUILD_GUI=OFF -DPENTOBI_BUILD_GTP=ON",
            "cmake --build /opt/pentobi-build --target pentobi-gtp --parallel 8",
        )
        .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
        .pip_install(
            "torch==2.14.1",
            "torchrl==0.14.1",
            "tensordict==0.14.3",
            "numpy==2.5.4",
            "pettingzoo==1.27.0",
            "gymnasium==1.4.0",
            "matplotlib==3.11.2",
            "maturin>=1.9,<2",
        )
        .add_local_file(source_archive(), "/tmp/blokus-source.tar.gz", copy=True)
        .run_commands(
            "mkdir -p /opt/blokus && tar -xzf /tmp/blokus-source.tar.gz -C /opt/blokus",
            "python -m pip install --no-deps /opt/blokus",
            "python -c 'from blokus_rl._native import duo_v2_shapes; print(duo_v2_shapes())'",
        )
        .env({"PENTOBI": "/opt/pentobi-build/pentobi_gtp/pentobi-gtp"})
    )
else:
    image = None

app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


def allotted_cpus():
    """CPU quota of this container (Modal reports more logical CPUs than it
    grants; oversubscribing rayon threads gets them throttled)."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    return os.cpu_count()


def limit_threads():
    os.environ.setdefault("RAYON_NUM_THREADS", str(allotted_cpus()))
    return int(os.environ["RAYON_NUM_THREADS"])


class Telemetry:
    """Sample GPU and CPU use every few seconds into a JSONL file."""

    def __init__(self, path, interval=10):
        self.path, self.interval = Path(path), interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        last = None
        with self.path.open("a") as handle:
            while not self.stop.is_set():
                row = {"time": time.time()}
                try:
                    row["gpu"] = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    ).strip()
                except Exception:
                    pass
                try:
                    usage = int(
                        next(
                            line.split()[1]
                            for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines()
                            if line.startswith("usage_usec")
                        )
                    )
                    if last:
                        row["cpu_cores"] = round(
                            (usage - last[1]) / 1e6 / (row["time"] - last[0]), 2
                        )
                    last = (row["time"], usage)
                except Exception:
                    pass
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)


@app.function(image=image, cpu=32, memory=16384, timeout=4 * 3600, volumes={"/vol": volume})
def pentobi_data(name: str, levels: list, games: int, workers: int, seed: int, max_seconds: float):
    from blokus_rl.v2 import pentobi_data as data

    out = Path("/vol/pentobi") / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            volume.commit()

    thread = threading.Thread(target=committer, daemon=True)
    thread.start()
    try:
        data.generate(
            out, levels, games, workers, seed, Path(os.environ["PENTOBI"]), [0, 1, 2, 3],
            max_seconds,
        )
    finally:
        stop.set()
        volume.commit()
    rows = out.read_text().count("\n")
    return {"games": rows, "path": str(out)}


@app.function(image=image, gpu="L4", cpu=16, memory=32768, timeout=1800, volumes={"/vol": volume})
def benchmark(configs: list):
    """Run self-play throughput benchmarks; each config is a list of CLI args."""
    threads = limit_threads()
    results = []
    for args in configs:
        output = subprocess.run(
            ["python", "-m", "blokus_rl.v2.benchmark", "--network", "v2", *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        row = json.loads(output.strip().splitlines()[-1])
        row["nproc"] = os.cpu_count()
        row["rayon_threads"] = threads
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


@app.function(image=image, gpu="L40S", cpu=32, memory=32768, timeout=6 * 3600, volumes={"/vol": volume})
def train_remote(run: str, args: list, init: str = "", imitation: str = "", resume: bool = False,
                 seed_replay: str = ""):
    """Self-play training into /vol/runs/<run>; resumable. `args` are train.py flags."""
    limit_threads()
    from blokus_rl.v2 import train as trainer

    out = Path("/vol/runs") / run
    command = ["--out", str(out), *args]
    if init:
        command += ["--init", init]
    if imitation:
        command += ["--imitation", imitation]
    if resume:
        command.append("--resume")
    if seed_replay:
        command += ["--seed-replay", seed_replay]
    parser_args = command
    import sys

    sys.argv = ["train", *parser_args]
    out.parent.mkdir(parents=True, exist_ok=True)
    telemetry_dir = Path("/vol/telemetry")
    telemetry_dir.mkdir(exist_ok=True)
    with Telemetry(telemetry_dir / f"{run}.jsonl"):
        from dataclasses import asdict

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--init", type=Path)
        parser.add_argument("--imitation", type=Path)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--seed-replay", type=Path)
        defaults = trainer.TrainConfig()
        for name, value in asdict(defaults).items():
            parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
        parsed = parser.parse_args(parser_args)
        config = trainer.TrainConfig(**{k: getattr(parsed, k) for k in asdict(defaults)})
        stats = trainer.train(
            config,
            parsed.out,
            init=parsed.init,
            imitation=parsed.imitation,
            resume=parsed.resume,
            commit=volume.commit,
            seed_replay=parsed.seed_replay,
        )
    volume.commit()
    return stats


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": volume})
def imitate_remote(name: str, inputs: list, args: list):
    """Supervised Pentobi imitation into /vol/imitation/<name>."""
    out = Path("/vol/imitation") / name
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            volume.commit()

    threading.Thread(target=committer, daemon=True).start()
    try:
        subprocess.run(
            ["python", "-m", "blokus_rl.v2.imitate", "--out", str(out), "--device", "cuda",
             "--precision", "bfloat16", "--inputs", *inputs, *args],
            check=True,
        )
    finally:
        stop.set()
        volume.commit()
    return (out / "progress.jsonl").read_text().splitlines()[-1]


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=2 * 3600, volumes={"/vol": volume})
def distill_remote(name: str, args: list):
    """Warm-start a larger network into /vol/distill/<name>; `args` are distill.py flags."""
    out = Path("/vol/distill") / name
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            volume.commit()

    threading.Thread(target=committer, daemon=True).start()
    try:
        subprocess.run(
            ["python", "-m", "blokus_rl.v2.distill", "--out", str(out), "--device", "cuda",
             "--precision", "bfloat16", *args],
            check=True,
        )
    finally:
        stop.set()
        volume.commit()
    return (out / "progress.jsonl").read_text().splitlines()[-1]


@app.local_entrypoint()
def distill(name: str, args: str):
    """Larger-network warm start on L40S + 8 cores (see cloud/v2_budget.py first)."""
    print(distill_remote.remote(name, args.split()))


@app.local_entrypoint()
def data(name: str, levels: str, games: int, workers: int = 48, seed: int = 71_000_000,
         max_seconds: float = 3600.0):
    """Pentobi games on a 32-core CPU container, e.g. --levels 7,8."""
    result = pentobi_data.remote(
        name, [int(x) for x in levels.split(",")], games, workers, seed, max_seconds
    )
    print(json.dumps(result))


@app.local_entrypoint()
def bench(configs: str, gpu: str = "L4", cpu: int = 16):
    """Benchmarks; configs separated by ';', each a space-separated arg list."""
    function = benchmark.with_options(gpu=gpu, cpu=cpu)
    rows = function.remote([c.split() for c in configs.split(";")])
    print(json.dumps(rows))


@app.local_entrypoint()
def imitate(name: str, inputs: str, args: str = "", gpu: str = "L40S"):
    """Imitation training; inputs are comma-separated Volume paths."""
    print(imitate_remote.with_options(gpu=gpu).remote(name, inputs.split(","), args.split()))


@app.local_entrypoint()
def train(run: str, args: str = "", init: str = "", imitation: str = "", resume: bool = False,
          seed_replay: str = "", gpu: str = "L40S", hours: float = 6.0):
    """Self-play RL chunk on a GPU + 32 cores; `hours` is the container timeout
    (see cloud/v2_budget.py first)."""
    function = train_remote.with_options(gpu=gpu, timeout=int(hours * 3600))
    print(json.dumps(function.remote(run, args.split(), init, imitation, resume, seed_replay)))
