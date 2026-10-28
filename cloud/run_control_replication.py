"""One independent control replication using the unchanged validated runtime."""

import argparse
import hashlib
import importlib.util
import json
import time
from dataclasses import asdict
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
from blokus_rl.duo_replay import pack_replay, replay_length


def tensor_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


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
    config = DeepConfig(**execution["config"])
    configs = {"control1024": config}
    protocol = {
        "parent_sha256": PARENT_SHA,
        "source_resume_sha256": STATE_SHA,
        "configs": {k: asdict(v) for k, v in configs.items()},
        "runtime": runtime,
        "initialization": "original selected round112 weights; fresh AdamW; full100000 inherited latest round128 replay positions; independent seed1071",
        "primary": "control minus preserved parent Pentobi4 score difference at2048 playing simulations; nominal whole-opening95% interval above0",
        "secondary": "control vs parent head-to-head, raw/search separately; Pentobi3",
        "stretch": "control absolute Pentobi4 nominal score interval above50%",
        "heldout_seed": 2608900,
        "heldout_pairs": 100,
        "selection": "every16 additional rounds vs frozen parent512,32 validation pairs seed2608300; parent selectable at50%",
        "no_extension": "no next level, search increase, architecture change or further training",
    }
    atomic_json(root / "study-protocol.json", protocol)
    volume.commit()
    started = time.monotonic()
    for name, cfg in configs.items():
        print("Starting independent replication training " + name, flush=True)
        initialize_branch(cfg, root / name, PARENT, STATE, selected_best=True)
        # Retain explicit evidence of the initial optimizer, replay and RNG before updates.
        initial = torch.load(root / name / "resume.pt", map_location="cpu", weights_only=False)
        assert initial["round"] == initial["games"] == 0
        assert initial["optimizer"]["state"] == {}
        atomic_json(
            root / "initialization-audit.json",
            {
                "round": initial["round"],
                "games": initial["games"],
                "optimizer_state_entries": len(initial["optimizer"]["state"]),
                "optimizer_groups": initial["optimizer"]["param_groups"],
                "replay_positions": replay_length(initial["replay"]),
                "packed_replay_tensor_sha256": tensor_sha(
                    {
                        k: v
                        for k, v in pack_replay(initial["replay"]).items()
                        if isinstance(v, torch.Tensor)
                    }
                ),
                "model_tensor_sha256": tensor_sha(initial["model_state"]),
                "numpy_rng": initial["numpy_rng"],
                "torch_rng_sha256": hashlib.sha256(
                    initial["torch_rng"].numpy().tobytes()
                ).hexdigest(),
                "cuda_rng_count": len(initial["cuda_rng"]),
            },
        )
        del initial
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
            "seed": 2608900,
            "pairs": 100,
            "simulations": 2048,
            "levels": [3, 4],
            "precision": "bfloat16",
            "inference_backend": "cuda_graphs",
        },
    )
    volume.commit()
    # Freeze before generating any test opening. Test outcomes never select weights.
    book = opening_book(100, seed=2608900)
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
                        "seed": 2608900,
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
            ("control-vs-parent-search2048", "control1024", "parent", 2048),
            ("control-vs-parent-raw", "control1024", "parent", 0),
        ]
        for name, left, right, budget in comparisons:
            print("Starting held-out " + name, flush=True)
            atomic_json(
                root / f"heldout-{name}-protocol.json",
                {
                    "seed": 2608900,
                    "pairs": 100,
                    "opening_book_sha256": hashlib.sha256(json.dumps(book).encode()).hexdigest(),
                    "left_sha256": sha(paths[left]),
                    "right_sha256": sha(paths[right]),
                    "simulations": budget,
                    "precision": "bfloat16",
                    "inference_backend": "cuda_graphs",
                },
            )
            volume.commit()
            row = evaluate_many(
                models[left],
                models[right],
                book,
                simulations=budget,
                opponent_simulations=budget,
                seed=2608900,
                precision="bfloat16",
                inference_backend="cuda_graphs",
            )
            atomic_json(root / f"heldout-{name}.json", row)
            volume.commit()
    differences = {}
    for level in (3, 4):
        for left, right in (("control1024", "parent"),):
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
            "external_games": 800,
            "head_to_head_games": 400,
        },
    )
    volume.commit()


if __name__ == "__main__":
    main()
