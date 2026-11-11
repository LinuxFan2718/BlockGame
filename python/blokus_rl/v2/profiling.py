"""Where a run's time and money go.

Self-play and training share one process, one GPU and one hourly price, so the
questions that matter are: is the GPU busy, are the CPU cores busy, what are
the threads waiting for, and what does a thousand games cost? Every row of a
run's `progress.jsonl` carries a `profile` object that answers them for the
interval since the previous row:

- `seconds`: wall time covered.
- `phases`: seconds per loop phase. `prepare` (Rust search), `submit` (batch to
  the device), `collect` (waiting for the network) and `apply` (Rust backup)
  are summed over the self-play threads. `callbacks` is the main thread;
  `train_sample`, `train_step` and `checkpoint` are parts of it.
- `cpu_cores` of `cpus`: cores this process kept busy, of those it may use.
- `gpu_util`, `gpu_memory_mb`: mean nvidia-smi utilisation and peak memory.
  "Busy" means some kernel was running, so it overstates how full the GPU is:
  read it together with the batch size and `benchmark --capacity`.
- `evaluations`, `batches`, `rows`: positions the network evaluated, in how
  many batches, and how many rows it actually ran (the excess is padding).
- any other key: how much a run counter (games, steps, simulations…) grew.

Summarise a run:

    python -m blokus_rl.v2.profiling runs/NAME --usd-per-hour 3.72

Reading the summary:
- GPU mostly idle and the self-play threads mostly in `prepare`: the CPU is
  the limit. More cores or slots help; so does a cheaper GPU.
- GPU busy and the threads mostly in `collect`: the GPU is the limit. Idle
  cores are then money spent for nothing.
- Much padding, or small batches: the GPU does work that evaluates nothing.
  More slots fill the batches.

This module does not import torch, so the report works anywhere.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

SELF_PLAY = ("prepare", "submit", "collect", "apply")
TRAINER = ("train_sample", "train_step", "checkpoint")
NVIDIA_SMI = (
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
)


@contextlib.contextmanager
def timed(timings, key):
    """Add the wall time of the block to `timings[key]`."""
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - started


def available_cpus():
    """Cores this process may use: container quota, else affinity, else count."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


def read_nvidia_smi():
    """(utilisation %, memory MB) averaged / summed over the visible GPUs."""
    text = subprocess.check_output(NVIDIA_SMI, text=True, timeout=5, stderr=subprocess.DEVNULL)
    gpus = [[float(value) for value in line.split(",")] for line in text.strip().splitlines()]
    return sum(g[0] for g in gpus) / len(gpus), sum(g[1] for g in gpus)


class GpuSampler:
    """Mean GPU utilisation and peak memory between laps, sampled on a
    background thread. Without nvidia-smi it does nothing, and it gives up
    quietly after repeated failures: profiling must never hurt the run."""

    def __init__(self, interval=5.0, read=None):
        self.interval = interval
        self._read = read
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._sum = self._peak = 0.0
        self._count = 0

    def start(self):
        if self._read is None:
            if shutil.which(NVIDIA_SMI[0]) is None:
                return self
            self._read = read_nvidia_smi
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        failures = 0
        while not self._stop.is_set() and failures < 3:
            try:
                util, memory = self._read()
            except Exception:
                failures += 1
            else:
                failures = 0
                with self._lock:
                    self._sum += util
                    self._count += 1
                    self._peak = max(self._peak, memory)
            self._stop.wait(self.interval)

    def lap(self):
        """Readings since the previous lap; empty if there were none."""
        with self._lock:
            total, count, peak = self._sum, self._count, self._peak
            self._sum = self._peak = 0.0
            self._count = 0
        if not count:
            return {}
        return {"gpu_util": round(total / count, 1), "gpu_memory_mb": round(peak), "gpu_samples": count}

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


class Profile:
    """Turns cumulative timers and counters into one row per interval.

    `timings` is the dict the run loop and `timed` add seconds to. `evaluator`
    supplies `positions`, `batches` and `rows` if it has them. `counters()`
    returns any other cumulative counts (games, gradient steps, engine counters).
    """

    def __init__(self, timings, evaluator=None, gpu=None, counters=None, *,
                 wall=time.perf_counter, cpu=time.process_time):
        self.timings, self.evaluator, self.gpu, self.counters = timings, evaluator, gpu, counters
        self.cpus = available_cpus()
        self._wall_clock, self._cpu_clock = wall, cpu
        self._wall, self._cpu = wall(), cpu()
        self._phases = dict(timings)
        self._totals = self._read_totals()

    def _read_totals(self):
        totals = dict(self.counters()) if self.counters else {}
        for name, attribute in (("evaluations", "positions"), ("batches", "batches"), ("rows", "rows")):
            value = getattr(self.evaluator, attribute, None)
            if value is not None:
                totals[name] = value
        return totals

    def lap(self):
        """Everything that happened since the previous lap (or construction)."""
        wall, cpu = self._wall_clock(), self._cpu_clock()
        seconds = max(wall - self._wall, 1e-9)
        row = {
            "seconds": round(seconds, 6),
            "cpus": self.cpus,
            # All threads of this process: Rust search, PyTorch and Python glue.
            "cpu_cores": round((cpu - self._cpu) / seconds, 2),
        }
        self._wall, self._cpu = wall, cpu
        phases = dict(self.timings)
        row["phases"] = {k: round(v - self._phases.get(k, 0.0), 6) for k, v in phases.items()}
        self._phases = phases
        totals = self._read_totals()
        row.update({k: v - self._totals.get(k, 0) for k, v in totals.items()})
        self._totals = totals
        if self.gpu is not None:
            row.update(self.gpu.lap())
        return row


def load_rows(path):
    """Rows of a run's progress log; `path` is the run directory or the file."""
    path = Path(path)
    if path.is_dir():
        path = path / "progress.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows, usd_per_hour=None):
    """Totals, rates and shares over every row that carries a profile."""
    profiles = [row["profile"] for row in rows if isinstance(row.get("profile"), dict)]
    if not profiles:
        raise ValueError("no profile data in these rows (was the run made before profiling?)")

    def total(key):
        return sum(p.get(key, 0) for p in profiles)

    seconds = max(total("seconds"), 1e-9)
    phases = {}
    for p in profiles:
        for key, value in p.get("phases", {}).items():
            phases[key] = phases.get(key, 0.0) + value
    summary = {
        "rows": len(profiles),
        "seconds": seconds,
        "games": total("games"),
        "steps": total("steps"),
        "evaluations": total("evaluations"),
        "games_per_second": total("games") / seconds,
        "evaluations_per_second": total("evaluations") / seconds,
        "steps_per_second": total("steps") / seconds,
        "cpus": max(p.get("cpus", 0) for p in profiles),
        "cpu_cores": sum(p.get("cpu_cores", 0.0) * p["seconds"] for p in profiles) / seconds,
        "phases": phases,
    }
    sampled = [p for p in profiles if "gpu_util" in p]
    if sampled:
        weight = sum(p["seconds"] for p in sampled)
        summary["gpu_util"] = sum(p["gpu_util"] * p["seconds"] for p in sampled) / weight
        summary["gpu_memory_mb"] = max(p["gpu_memory_mb"] for p in sampled)
    if total("batches"):
        summary["mean_batch"] = total("evaluations") / total("batches")
    if total("rows"):
        summary["padding"] = 1 - total("evaluations") / total("rows")
    self_play = sum(phases.get(k, 0.0) for k in SELF_PLAY)
    if self_play:
        summary["self_play"] = {k: phases.get(k, 0.0) / self_play for k in SELF_PLAY}
    # The main thread's time, as shares of the wall clock.
    summary["main_thread"] = {k: phases.get(k, 0.0) / seconds for k in TRAINER}
    summary["main_thread"]["other"] = max(
        phases.get("callbacks", 0.0) - sum(phases.get(k, 0.0) for k in TRAINER), 0.0
    ) / seconds
    if usd_per_hour is not None:
        cost = usd_per_hour * seconds / 3600
        summary["usd_per_hour"] = usd_per_hour
        summary["usd"] = cost
        if summary["games"]:
            summary["usd_per_1000_games"] = 1000 * cost / summary["games"]
        if summary["evaluations"]:
            summary["usd_per_million_evaluations"] = 1e6 * cost / summary["evaluations"]
    return summary


def render(summary, name="run"):
    """The summary as a few aligned lines of text."""

    def percent(value):
        return f"{100 * value:.0f}%"

    seconds = summary["seconds"]
    span = f"{seconds / 3600:.2f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"
    lines = [f"{name}: {span} profiled over {summary['rows']} log rows"]

    def add(label, text):
        lines.append(f"  {label:<12} {text}")

    add(
        "throughput",
        f"{summary['games_per_second']:.1f} games/s · "
        f"{summary['evaluations_per_second']:,.0f} evaluations/s · "
        f"{summary['steps_per_second']:.1f} gradient steps/s",
    )
    if "gpu_util" in summary:
        add("GPU", f"{summary['gpu_util']:.0f}% busy · {summary['gpu_memory_mb'] / 1024:.1f} GB peak")
    else:
        add("GPU", "not sampled (no nvidia-smi)")
    cpus = summary["cpus"]
    share = f" ({percent(summary['cpu_cores'] / cpus)})" if cpus else ""
    add("CPU", f"{summary['cpu_cores']:.1f} of {cpus} cores busy{share}")
    if "mean_batch" in summary:
        text = f"{summary['mean_batch']:,.0f} positions per batch"
        if "padding" in summary:
            text += f" · {percent(summary['padding'])} of network rows were padding"
        add("batches", text)
    if "self_play" in summary:
        s = summary["self_play"]
        add(
            "self-play",
            f"search {percent(s['prepare'])} · to device {percent(s['submit'])} · "
            f"waiting for network {percent(s['collect'])} · backup {percent(s['apply'])}",
        )
    m = summary["main_thread"]
    add(
        "main thread",
        f"gradient steps {percent(m['train_step'])} · sampling {percent(m['train_sample'])} · "
        f"checkpoints {percent(m['checkpoint'])} · other {percent(m['other'])} (of wall time)",
    )
    if "usd" in summary:
        text = f"${summary['usd']:.2f} at ${summary['usd_per_hour']:.2f}/h"
        if "usd_per_1000_games" in summary:
            text += f" · ${summary['usd_per_1000_games']:.3f} per 1,000 games"
        if "usd_per_million_evaluations" in summary:
            text += f" · ${summary['usd_per_million_evaluations']:.4f} per million evaluations"
        add("cost", text)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="run directory or its progress.jsonl")
    parser.add_argument("--usd-per-hour", type=float, help="what the machine costs, for cost lines")
    parser.add_argument("--last", type=int, help="only the last N log rows")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()
    rows = load_rows(args.run)
    if args.last:
        rows = rows[-args.last :]
    summary = summarize(rows, args.usd_per_hour)
    print(json.dumps(summary) if args.json else render(summary, str(args.run)))


if __name__ == "__main__":
    main()
