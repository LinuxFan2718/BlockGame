"""One fresh, matched study using the locally audited integrated pipeline."""

import argparse
import hashlib
import importlib.util
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
import torch
from blokus_rl.duo_continuation import initialize_branch
from blokus_rl.duo_crossplay import opening_book, paired_interval
from blokus_rl.duo_deep import DeepConfig, atomic_json, train
from blokus_rl.duo_deep_evaluate import evaluate_many, evaluation_backend
from blokus_rl.duo_learning import load_duo

PARENT = Path("/data/duo-continuation-865/frozen-strong1024.pt")
STATE = PARENT.parent / "strong1024/resume.pt"
PARENT_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
STATE_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
EVALUATOR_SHA = "80727fc7fee1fcdab75eebc989f41f617e6721d4f546e266367ca53f294a0493"
BINARY_SHA = "ccb2362fc2eb066be9337b76488c527f22aae4ad4b72296ae105e54b2c015882"
volume = modal.Volume.from_name("blokus-deep-selfplay-20261026", create_if_missing=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    root = args.out
    # The wrapper owns the exclusive run directory; no infrastructure replay.
    execution = json.loads(args.execution.read_text())
    runtime = execution["selected_runtime"]
    assert sha(PARENT) == PARENT_SHA and sha(STATE) == STATE_SHA
    evaluator_path = Path("/opt/blokus/cloud/duo_pentobi_steps.py")
    binary = Path("/opt/pentobi-build/pentobi_gtp/pentobi-gtp")
    assert sha(evaluator_path) == EVALUATOR_SHA and sha(binary) == BINARY_SHA
    spec = importlib.util.spec_from_file_location("steps", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    torch.set_float32_matmul_precision("high")
    config = DeepConfig(
        architecture="conv",
        seed=1070,
        rounds=64,
        eval_every=16,
        live_games=64,
        batch_wait_ms=0,
        full_simulations=1024,
        validation_openings=32,
        validation_simulations=512,
        validation_seed=2607300,
        test_seed=2607900,
        max_seconds=6000,
    )
    configs = {
        "control1024": config,
        "teacher2048": replace(config, full_simulations=2048, max_seconds=10000),
    }
    protocol = {
        "parent_sha256": PARENT_SHA,
        "source_resume_sha256": STATE_SHA,
        "configs": {k: asdict(v) for k, v in configs.items()},
        "runtime": runtime,
        "initialization": "selected best round112 weights; fresh AdamW for both; identical100000 inherited latest round128 replay positions; fresh seed1070",
        "primary": "teacher minus preserved parent Pentobi4 score difference at2048 playing simulations; nominal whole-opening95% interval above0",
        "secondary": "teacher minus control Pentobi4 and head-to-head; raw/search separately; Pentobi3",
        "stretch": "teacher absolute Pentobi4 nominal score interval above50%",
        "heldout_seed": 2607900,
        "heldout_pairs": 100,
        "selection": "every16 additional rounds vs frozen parent512,32 validation pairs seed2607300; parent selectable at50%",
        "no_extension": "no next level, search increase, architecture change or further training",
    }
    atomic_json(root / "study-protocol.json", protocol)
    volume.commit()
    started = time.monotonic()
    for name, cfg in configs.items():
        print("Starting integrated training " + name, flush=True)
        initialize_branch(cfg, root / name, PARENT, STATE, selected_best=True)
        volume.commit()
        train(cfg, root / name, PARENT, resume=True, commit=volume.commit, **runtime)
    frozen = {}
    for name in configs:
        path = root / f"frozen-{name}.pt"
        path.write_bytes((root / name / "best.pt").read_bytes())
        frozen[name] = sha(path)
    atomic_json(
        root / "heldout-protocol.json",
        {
            "checkpoint_sha256": frozen,
            "parent_sha256": PARENT_SHA,
            "seed": 2607900,
            "pairs": 100,
            "simulations": 2048,
            "levels": [3, 4],
            "precision": "bfloat16",
            "inference_backend": "cuda_graphs",
        },
    )
    volume.commit()
    # Freeze before generating any test opening. Test outcomes never select weights.
    book = opening_book(100, seed=2607900)
    paths = {"parent": PARENT} | {name: root / f"frozen-{name}.pt" for name in configs}
    results = {}
    with evaluation_backend("cuda_graphs"):
        for name, path in paths.items():
            for level in (3, 4):
                case = f"{name}-pentobi{level}"
                print("Starting held-out " + case, flush=True)
                out = root / "heldout" / case
                out.parent.mkdir(exist_ok=True)
                results[case] = evaluator.run(
                    {
                        "first_level": level,
                        "last_level": level,
                        "opening_pairs": 100,
                        "seed": 2607900,
                        "device": "cuda",
                        "precision": "bfloat16",
                        "simulations": 2048,
                    },
                    path,
                    binary,
                    out,
                    commit=volume.commit,
                )[str(level)]
                atomic_json(root / "heldout-results.json", results)
                volume.commit()
        models = {name: load_duo(path).cuda().eval() for name, path in paths.items()}
        comparisons = [
            ("teacher-vs-control-search2048", "teacher2048", "control1024", 2048),
            ("teacher-vs-control-raw", "teacher2048", "control1024", 0),
            ("teacher-vs-parent-raw", "teacher2048", "parent", 0),
            ("control-vs-parent-raw", "control1024", "parent", 0),
        ]
        for name, left, right, budget in comparisons:
            print("Starting held-out " + name, flush=True)
            row = evaluate_many(
                models[left],
                models[right],
                book,
                simulations=budget,
                opponent_simulations=budget,
                seed=2607900,
                precision="bfloat16",
                inference_backend="cuda_graphs",
            )
            atomic_json(root / f"heldout-{name}.json", row)
            volume.commit()
    differences = {}
    for level in (3, 4):
        for left, right in (
            ("teacher2048", "parent"),
            ("teacher2048", "control1024"),
            ("control1024", "parent"),
        ):
            delta = np.asarray(
                results[f"{left}-pentobi{level}"]["opening_pair_scores"]
            ) - np.asarray(results[f"{right}-pentobi{level}"]["opening_pair_scores"])
            differences[f"{left}-minus-{right}-pentobi{level}"] = {
                "score_difference": float(delta.mean()),
                "nominal_paired_95pct_interval": paired_interval(delta),
            }
    atomic_json(root / "paired-differences.json", differences)
    assert sha(PARENT) == PARENT_SHA and sha(STATE) == STATE_SHA
    assert all(sha(root / f"frozen-{name}.pt") == h for name, h in frozen.items())
    atomic_json(
        root / "finished.json",
        {
            "seconds": time.monotonic() - started,
            "finish_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": frozen,
            "parent_sha256": PARENT_SHA,
            "source_resume_sha256": STATE_SHA,
            "training": {
                name: json.loads((root / name / "completed.json").read_text()) for name in configs
            },
            "external_games": 1200,
            "head_to_head_games": 800,
        },
    )
    volume.commit()


if __name__ == "__main__":
    main()
