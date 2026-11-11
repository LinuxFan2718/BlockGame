"""Profiling helpers. Nothing here needs torch or the Rust extension."""

import json
import time
from types import SimpleNamespace

import pytest
from blokus_rl.v2 import profiling


def test_timed_adds_wall_time_even_when_the_block_raises():
    timings = {"step": 1.0}
    with profiling.timed(timings, "step"):
        time.sleep(0.01)
    with pytest.raises(RuntimeError), profiling.timed(timings, "other"):
        raise RuntimeError
    assert 1.01 <= timings["step"] < 1.5
    assert timings["other"] >= 0


def test_profile_reports_increases_since_the_previous_lap():
    clock = {"wall": 100.0, "cpu": 10.0}
    timings = {"prepare": 5.0}
    evaluator = SimpleNamespace(positions=1000, batches=10, rows=1280)
    counts = {"games": 440_000, "steps": 7}  # e.g. a resumed run
    profile = profiling.Profile(
        timings, evaluator, counters=lambda: counts,
        wall=lambda: clock["wall"], cpu=lambda: clock["cpu"],
    )

    clock.update(wall=160.0, cpu=250.0)
    timings.update(prepare=35.0, train_step=12.0)
    evaluator.positions, evaluator.batches, evaluator.rows = 7000, 16, 7936
    counts.update(games=440_120, steps=19)
    row = profile.lap()
    assert row["seconds"] == 60.0
    assert row["cpu_cores"] == 4.0
    assert row["phases"] == {"prepare": 30.0, "train_step": 12.0}
    assert (row["evaluations"], row["batches"], row["rows"]) == (6000, 6, 6656)
    assert (row["games"], row["steps"]) == (120, 12)
    assert "gpu_util" not in row
    json.dumps(row)

    clock.update(wall=170.0, cpu=255.0)
    again = profile.lap()
    assert again["seconds"] == 10.0 and again["cpu_cores"] == 0.5
    assert again["phases"] == {"prepare": 0.0, "train_step": 0.0}
    assert again["games"] == 0 and again["evaluations"] == 0


def test_profile_works_without_an_evaluator_or_counters():
    row = profiling.Profile({}).lap()
    assert row["phases"] == {} and row["cpus"] >= 1 and row["seconds"] >= 0


def wait_for(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert condition()


def test_gpu_sampler_reports_mean_utilisation_and_peak_memory_per_lap():
    readings = iter([(40.0, 1000.0), (60.0, 3000.0), (80.0, 2000.0)])
    taken = []

    def read():
        taken.append(next(readings))  # StopIteration afterwards counts as a failure
        return taken[-1]

    sampler = profiling.GpuSampler(interval=0.001, read=read).start()
    wait_for(lambda: len(taken) == 3 and not sampler._thread.is_alive())
    assert sampler.lap() == {"gpu_util": 60.0, "gpu_memory_mb": 3000, "gpu_samples": 3}
    assert sampler.lap() == {}
    sampler.stop()


def test_gpu_sampler_gives_up_quietly_when_readings_fail():
    calls = []

    def read():
        calls.append(1)
        raise OSError("no GPU here")

    sampler = profiling.GpuSampler(interval=0.001, read=read).start()
    wait_for(lambda: not sampler._thread.is_alive())
    assert len(calls) == 3
    assert sampler.lap() == {}
    sampler.stop()


def test_gpu_sampler_is_inert_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(profiling.shutil, "which", lambda name: None)
    sampler = profiling.GpuSampler().start()
    assert sampler._thread is None and sampler.lap() == {}
    sampler.stop()


def test_nvidia_smi_output_is_parsed_across_gpus(monkeypatch):
    monkeypatch.setattr(profiling.subprocess, "check_output", lambda *a, **k: "50, 1000\n70, 3000\n")
    assert profiling.read_nvidia_smi() == (60.0, 4000.0)


def profile_row(seconds, **fields):
    return {"games": 0, "profile": {"seconds": seconds, "cpus": 32, **fields}}


def test_summary_weights_by_time_and_prices_the_run():
    rows = [
        {"games": 5},  # a row logged before profiling existed
        profile_row(
            60.0, cpu_cores=16.0, gpu_util=50.0, gpu_memory_mb=4000, games=600, steps=120,
            evaluations=6_000_000, batches=6000, rows=7_500_000,
            phases={"prepare": 30.0, "submit": 10.0, "collect": 70.0, "apply": 10.0,
                    "callbacks": 24.0, "train_sample": 3.0, "train_step": 18.0},
        ),
        profile_row(
            180.0, cpu_cores=32.0, gpu_util=90.0, gpu_memory_mb=9000, games=3000, steps=600,
            evaluations=24_000_000, batches=24000, rows=30_000_000,
            phases={"prepare": 90.0, "submit": 30.0, "collect": 210.0, "apply": 30.0,
                    "callbacks": 60.0, "train_sample": 9.0, "train_step": 36.0, "checkpoint": 3.0},
        ),
    ]
    summary = profiling.summarize(rows, usd_per_hour=3.6)
    assert summary["rows"] == 2 and summary["seconds"] == 240.0
    assert summary["games"] == 3600 and summary["games_per_second"] == 15.0
    assert summary["evaluations_per_second"] == 125_000
    assert summary["cpu_cores"] == pytest.approx(28.0)
    assert summary["gpu_util"] == pytest.approx(80.0) and summary["gpu_memory_mb"] == 9000
    assert summary["mean_batch"] == 1000 and summary["padding"] == pytest.approx(0.2)
    assert summary["self_play"] == pytest.approx(
        {"prepare": 0.25, "submit": 1 / 12, "collect": 7 / 12, "apply": 1 / 12}
    )
    assert summary["main_thread"] == pytest.approx(
        {"train_sample": 0.05, "train_step": 0.225, "checkpoint": 0.0125, "other": 0.0625}
    )
    assert summary["usd"] == pytest.approx(0.24)
    assert summary["usd_per_1000_games"] == pytest.approx(0.24 / 3.6)
    assert summary["usd_per_million_evaluations"] == pytest.approx(0.008)

    text = profiling.render(summary, "runs/demo")
    assert "80% busy" in text and "28.0 of 32 cores" in text
    assert "waiting for network 58%" in text and "20% of network rows were padding" in text
    assert "$0.067 per 1,000 games" in text
    json.dumps(summary)


def test_summary_without_gpu_samples_or_price():
    summary = profiling.summarize([profile_row(10.0, cpu_cores=4.0, games=20, phases={"prepare": 9.0})])
    assert "gpu_util" not in summary and "usd" not in summary and "mean_batch" not in summary
    text = profiling.render(summary)
    assert "not sampled" in text and "cost" not in text


def test_summary_needs_profile_rows():
    with pytest.raises(ValueError, match="no profile data"):
        profiling.summarize([{"games": 5}])


def test_report_reads_a_run_directory(tmp_path, capsys, monkeypatch):
    rows = [profile_row(30.0, cpu_cores=2.0, games=60, phases={"prepare": 20.0})] * 3
    (tmp_path / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(profiling.load_rows(tmp_path)) == 3
    monkeypatch.setattr("sys.argv", ["profiling", str(tmp_path), "--last", "2", "--json"])
    profiling.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 2 and summary["games"] == 120
