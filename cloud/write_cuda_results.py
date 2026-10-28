"""Render report tables from verified, local frozen-GPU performance artifacts."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(Path(path).read_text())


def resource_summary(report, names):
    rows = [
        v
        for k, v in report["telemetry_by_phase"].items()
        if k.split("/")[-1] in names and v.get("samples")
    ]
    assert len(rows) == len(names)
    samples = sum(r["samples"] for r in rows)
    return {
        "gpu": sum(r["gpu_busy_mean"] * r["samples"] for r in rows) / samples,
        "cpu": sum(r["cpu_cores_mean"] * r["samples"] for r in rows) / samples,
        "ram": max(r["host_memory_peak_gib"] for r in rows),
        "vram": max(r["vram_peak_mib"] for r in rows),
    }


def main():
    a, b = [load(ROOT / f"runs/cuda-graphs-{g}-verified.json") for g in ("l40s", "h100")]
    c = load(ROOT / "runs/cuda-latency-verified.json")
    assert all(r["verified"] for r in (a, b, c))
    variants = (
        "current-eager",
        "current-graph",
        "native-eager",
        "native-graph",
        "native-graph-wide",
    )
    lines = [
        "## Verified results — 28 October 2026",
        "",
        "CUDA Graphs improved actual collection on both GPUs. On L40S, graphs alone",
        f"raised throughput **{a['collection']['current-graph']['games_per_second'] / a['collection']['current-eager']['games_per_second']:.2f}×**; native changes plus graphs and live64 raised it",
        f"**{a['collection']['native-graph-wide']['games_per_second'] / a['collection']['current-eager']['games_per_second']:.2f}×**, from {a['collection']['current-eager']['games_per_second']:.2f} to {a['collection']['native-graph-wide']['games_per_second']:.2f} games/s. These are frozen self-play",
        "collection measurements, not complete training-job or playing-strength gains.",
        "",
        "| Configuration | L40S seconds / 384 | L40S games/s | H100 seconds / 384 | H100 games/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in variants:
        x, y = a["collection"][k], b["collection"][k]
        lines.append(
            f"| {k} | {x['mean_seconds']:.2f} | {x['games_per_second']:.2f} | {y['mean_seconds']:.2f} | {y['games_per_second']:.2f} |"
        )
    lines += [
        "",
        "![Two repetitions of each collection case and GPU-resident inference ceiling](assets/cuda-graphs-results.png)",
        "",
        "Bars show two means; whiskers show the observed repeat range, not confidence",
        "intervals. H100 repetition 2 ran on a replacement container after preemption.",
        "",
        "### Same-device H100 batching/layout diagnostic",
        "",
        "| Layout / batching delay | Seconds / 384 | Games/s | Simulations/s | GPU busy | CPU cores | Estimated $ / 1,000 games |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, v in c["latency"].items():
        layout, delay = k.rsplit("-", 1)
        r = resource_summary(c, {f"{i}-{layout}-wait-{delay}" for i in (3, 4)})
        lines.append(
            f"| {layout}, {delay} ms | {v['mean_seconds']:.2f} | {v['games_per_second']:.2f} | {v['simulations_per_second']:,.0f} | {r['gpu']:.1f}% | {r['cpu']:.2f}/8 | ${v['rate_estimated_usd_per_1000_games']:.3f} |"
        )
    best_key = max(c["latency"], key=lambda k: c["latency"][k]["games_per_second"])
    best = c["latency"][best_key]
    base = c["latency"]["nchw-1.0"]
    lines += [
        "",
        f"Best tested H100 setting: **{best_key}**, {best['games_per_second'] / base['games_per_second']:.2f}× the same-device 1 ms NCHW baseline.",
        "This is a finite diagnostic; it does not establish the globally optimal delay,",
        "live-game count or hardware configuration. Layout can alter bf16 rounding and",
        "search trajectories, so all actual simulation counts accompany game throughput.",
        "",
        "![Same-device H100 batching delay and layout results](assets/cuda-latency-results.png)",
        "",
        "### GPU activity, memory and the remaining bottleneck",
        "",
        "| Measured collection configuration | GPU busy | CPU cores | Peak host GiB / 24 | Peak VRAM MiB | Physical / requested neural rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in (a, b):
        for k in ("current-eager", "native-graph-wide"):
            r = resource_summary(report, {f"{i}-{k}" for i in (1, 2)})
            lines.append(
                f"| {report['gpu'].upper()} {k} | {r['gpu']:.1f}% | {r['cpu']:.2f}/8 | {r['ram']:.2f} | {r['vram']:.0f} | {report['collection'][k]['padding_factor']:.3f}× |"
            )
    lines += [
        "",
        "| GPU / layout | Graph positions/s, batch128 | Graph positions/s, batch2048 |",
        "|---|---:|---:|",
    ]
    for report in (a, b):
        lines.append(
            f"| {report['gpu'].upper()} NCHW, comparison stage | {report['micro_summary']['128-graph']:,.0f} | {report['micro_summary']['2048-graph']:,.0f} |"
        )
    for layout in ("nchw", "channels-last"):
        path = Path(c["run"]) / (layout + "-micro/micro.json")
        if path.exists():
            cases = load(path)["cases"]
            vals = [
                statistics.median(
                    x["positions_per_second"]
                    for x in cases
                    if x["rows"] == n and x["method"] == "graph" and not x.get("sustained")
                )
                for n in (128, 2048)
            ]
            lines.append(
                f"| H100 {layout}, same-device final stage | {vals[0]:,.0f} | {vals[1]:,.0f} |"
            )
    lines += [
        "",
        "These GPU-resident tests exclude request packing, transfer and search. In",
        "collection the inference service includes packing/copy/response work and each",
        "search worker waits for its result. Low aggregate CPU use does not prove the",
        "CPU-side search work is cheap: dependencies and handoffs leave cores idle.",
        "The gap between bulk inference and delivered collection supports focusing on",
        "concurrency and handoff overhead. It does not identify a single kernel or",
        "establish peak FLOP/bandwidth saturation. GPU and host memory capacity were",
        "not exhausted in the successful comparisons.",
        "",
        "### Audit, actual work and costs",
        "",
        "All 10 cases per primary GPU passed the saved-artifact audit: 3,840 measured",
        "games plus 120 warmups each. The final probe ran "
        + str(len(load(Path(c["run"]) / "finished.json")["results"]))
        + " cases, each 384 measured",
        "games plus 12 warmups. There were **zero gradient updates**, no new model",
        "selection and no opponent benchmarks. Runtime tables exclude setup/warmup;",
        "provider costs include paid setup, the failed loader attempt and preemption.",
        "",
        "| Stage | Measured simulations | Requested neural positions | Physical neural rows, including padding |",
        "|---|---:|---:|---:|",
    ]
    for label, groups in (
        ("L40S", a["collection"]),
        ("H100", b["collection"]),
        ("H100 final probe", c["latency"]),
    ):
        cases = [case for value in groups.values() for case in value["cases"]]
        lines.append(
            f"| {label} | {sum(x['simulations'] for x in cases):,} | {sum(x['neural_positions'] for x in cases):,} | {sum(x['physical_neural_positions'] for x in cases):,} |"
        )
    lines += [
        "",
        "| Phase app | Actual provider metering USD |",
        "|---|---:|",
    ]
    for name, cost in c["cost_by_phase_app"].items():
        lines.append(f"| `{name}` | ${cost} |")
    lines += [
        f"| **Total** | **${c['combined_actual_usd']}** |",
        "",
        "This is CLI provider metering at the final snapshot, not an invoice. The",
        "original shared reservation was $4.19642 under a $5 ceiling; unused stopped",
        "reservations were settled before the bounded final probe. The $50 credit",
        "reserve remains protected. Estimated dollars per 1,000 games use collection",
        "time and the GPU+8-core+24-GiB rate, excluding setup and failures.",
        "",
        "The native implementation passed exact ordered-state and arbitrary-prior",
        "root-statistic comparisons. GPU checks covered seven batch sizes, padded",
        "capture equality, unpadded probabilities/values and output lifetime. Layout",
        "checks retained identical weights; their actual errors are in the final",
        "verification JSON. The two GPU fixtures have exactly identical arrays.",
        "",
        "Audits verify source/payload/native hashes, original completed-case bytes,",
        "all counters, physical padding, stopped apps and preserved certified/Blokus",
        "checkpoints. They do not replay full game action traces: this performance",
        "study saved collection counters and record digests, not game traces.",
        "",
        "Reproduction after downloading into unused directories:",
        "",
        "```sh",
        ".venv/bin/python cloud/report_cuda_benchmark.py --run <stage> --apps <stopped.json> --app-id <id> --billing <report.json> --gpu <l40s|h100>",
        ".venv/bin/python cloud/report_cuda_latency.py --run <final-probe> --apps <stopped.json> --app-id <id> --billing <report.json>",
        "MPLCONFIGDIR=.cache/matplotlib .venv/bin/python cloud/plot_cuda_benchmark.py",
        ".venv/bin/python cloud/write_cuda_results.py",
        "```",
        "",
        "Full verified counters and telemetry are in `runs/cuda-graphs-l40s-verified.json`,",
        "`runs/cuda-graphs-h100-verified.json` and `runs/cuda-latency-verified.json`.",
        "",
        "Training remains stopped. The next implementation target is a persistent",
        "native actor/inference pipeline with more independent games and fewer",
        "Python queue/packing handoffs, measuring useful simulations/s and cost.",
        "Graph/layout integration into training must also verify behavior across",
        "weight updates; this study evaluated a frozen model only. No further paid",
        "experiment or training is launched automatically.",
        "",
    ]
    path = ROOT / "docs/cuda-graphs.md"
    text = path.read_text()
    protocol = text.split("## Prespecified execution", 1)[1]
    prefix = text.split("## Prespecified execution", 1)[0].split("## Verified results", 1)[0]
    path.write_text(prefix + "\n".join(lines) + "\n## Prespecified execution" + protocol)


if __name__ == "__main__":
    main()
