"""GPU checks and fixed-work comparisons of the integrated production pipeline."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import modal
import numpy as np
import torch
from blokus_rl.duo_continuation import initialize_branch
from blokus_rl.duo_cuda_graphs import FrozenGraphEvaluator
from blokus_rl.duo_deep import DeepConfig, atomic_json, fit_indexed, train
from blokus_rl.duo_inference import IndexedBlock
from blokus_rl.duo_learning import load_duo
from blokus_rl.duo_replay import pack_replay, replay_records
from blokus_rl.duo_shared import SharedSearchPool, evaluate_indexed
from blokus_rl.learning import configure_torch

MODEL = Path("/data/duo-continuation-865/frozen-strong1024.pt")
STATE = Path("/data/duo-continuation-865/strong1024/resume.pt")
FIXTURE = Path("/data/duo-cuda-graphs-1069-r1/l40s/inputs.npz")
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def fixture():
    with np.load(FIXTURE) as data:
        obs, indices, offsets = data["observations"], data["indices"], data["offsets"]
    return [
        {"observation": obs[i], "indices": indices[offsets[i] : offsets[i + 1]]}
        for i in range(len(obs))
    ]


def blocks(rows):
    result = []
    for offset in range(0, len(rows), 17):
        group = rows[offset : offset + 17]
        width = max(len(r["indices"]) for r in group)
        ids = np.zeros((len(group), width + 7), np.uint16)
        for i, row in enumerate(group):
            ids[i, : len(row["indices"])] = row["indices"]
        result.append(
            IndexedBlock(
                np.stack([r["observation"] for r in group]),
                ids,
                np.asarray([len(r["indices"]) for r in group], np.uint16),
            )
        )
    return result


def numeric_checks(model, rows):
    graph = FrozenGraphEvaluator(model, fallback=evaluate_indexed, precision="bfloat16")
    graph.prepare()
    checks = []
    for n in (1, 9, 17, 33, 65, 127, 256):
        subset = rows[:n]
        expected, actual = (
            evaluate_indexed(model, subset, "bfloat16"),
            graph.evaluate_blocks(blocks(subset)),
        )
        error = [float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)]
        assert error[0] <= 0.002 and error[1] <= 0.02
        row_graph = graph(subset)
        assert all(np.array_equal(a, b) for a, b in zip(row_graph, actual, strict=True))
        saved = tuple(a.copy() for a in actual)
        graph.evaluate_blocks(blocks(rows[-n:]))
        assert all(np.array_equal(a, b) for a, b in zip(saved, actual, strict=True))
        checks.append({"rows": n, "max_abs_error": error, "bulk_matches_row_graph_exactly": True})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=False)
    configure_torch(1070)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    original = sha(MODEL), sha(STATE)
    rows = fixture()
    model = load_duo(MODEL).cuda().eval()
    first = numeric_checks(model, rows)
    training = [
        r | {"visits": np.ones(len(r["indices"]), np.uint16), "value_target": float(i % 2)}
        for i, r in enumerate(rows[:64])
    ]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit_indexed(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.001),
        training,
        np.random.default_rng(1070),
        DeepConfig(architecture="conv", minibatch_size=32),
        2,
    )
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    updated = numeric_checks(model, rows)
    atomic_json(
        out / "numerical-checks.json",
        {"initial": first, "after_optimizer_updates": updated, "discarded_gradient_steps": 2},
    )
    volume.commit()
    # Small inherited replay is used ONLY in discarded recovery checks.
    source = torch.load(STATE, map_location="cpu", weights_only=False)
    source["replay"] = pack_replay(replay_records(source["replay"])[:256])
    torch.save(source, out / "smoke-source.pt")
    del source
    cfg = DeepConfig(
        architecture="conv",
        seed=1070,
        rounds=2,
        workers=1,
        live_games=4,
        games_per_round=4,
        full_simulations=16,
        fast_simulations=4,
        validation_openings=1,
        validation_simulations=8,
        eval_every=2,
        max_seconds=300,
        batch_wait_ms=0,
    )
    runtime = dict(inference_backend="cuda_graphs", evaluation_backend="cuda_graphs")
    for name in ("uninterrupted", "recovered"):
        initialize_branch(cfg, out / name, MODEL, out / "smoke-source.pt", selected_best=True)
    train(cfg, out / "uninterrupted", MODEL, resume=True, **runtime)
    train(cfg, out / "recovered", MODEL, resume=True, stop_after_round=1, **runtime)
    train(cfg, out / "recovered", MODEL, resume=True, **runtime)
    left, right = [
        torch.load(out / name / "resume.pt", map_location="cpu", weights_only=False)
        for name in ("uninterrupted", "recovered")
    ]
    fields = (
        "model_state",
        "optimizer",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "replay",
        "best_checkpoint",
    )
    for name in fields:
        assert same(left[name], right[name]), name
    atomic_json(
        out / "recovery-check.json",
        {
            "exact_single_worker_gpu_recovery": True,
            "fields": fields,
            "rounds_per_smoke": 2,
            "games_per_smoke": 8,
            "inherited_smoke_positions": 256,
            "unpromoted": True,
        },
    )
    volume.commit()
    del left, right, before, model
    model = load_duo(MODEL).cuda().eval()
    cases = []
    modes = {
        "legacy": dict(bulk_inference=False, compact_native=False, ready_batching=False),
        "compact": dict(bulk_inference=False, compact_native=True, ready_batching=True),
        "bulk": dict(bulk_inference=True, compact_native=True, ready_batching=True),
    }
    order = tuple((full, mode) for full in (1024, 2048) for mode in modes)
    for rep, variants in ((1, order), (2, tuple(reversed(order)))):
        for full, mode in variants:
            tag = f"{rep}-full{full}-{mode}"
            print("Integrated collection " + tag, flush=True)
            config = DeepConfig(
                architecture="conv",
                seed=1070,
                rounds=1,
                games_per_round=256,
                live_games=64,
                full_simulations=full,
                batch_wait_ms=0,
            )
            with SharedSearchPool(config, inference_backend="cuda_graphs", **modes[mode]) as pool:
                pool.collect(model, 0, 12)
                records, scores, metrics = pool.collect(model, rep, 256)
            assert len(scores) == 256 and all(int(r["visits"].sum()) == full for r in records)
            assert (
                metrics["gpu_positions"]
                == metrics["neural_positions"]
                == metrics["graph_requested_positions"]
            )
            digest = hashlib.sha256()
            for row in records:
                for key, dtype in (
                    ("observation", np.float32),
                    ("indices", np.uint16),
                    ("visits", np.uint16),
                ):
                    digest.update(np.asarray(row[key], dtype=dtype).tobytes())
                digest.update(np.asarray([row["value_target"]], np.float64).tobytes())
            result = {
                "tag": tag,
                "repeat": rep,
                "full_simulations": full,
                "mode": mode,
                "runtime": modes[mode],
                "config": asdict(config),
                "games": 256,
                "warmup_games": 12,
                "recorded_roots": len(records),
                "record_sha256": digest.hexdigest(),
                "metrics": metrics,
            }
            atomic_json(out / (tag + ".json"), result)
            volume.commit()
            cases.append(result)
            print(json.dumps({"tag": tag, "seconds": metrics["collect_seconds"]}), flush=True)
    assert (sha(MODEL), sha(STATE)) == original
    atomic_json(
        out / "finished.json",
        {
            "cases": cases,
            "seconds": time.monotonic() - started,
            "model_sha256": original[0],
            "source_resume_sha256": original[1],
            "discarded_smoke_games": 16,
            "discarded_optimizer_probe_steps": 2,
        },
    )


if __name__ == "__main__":
    main()
