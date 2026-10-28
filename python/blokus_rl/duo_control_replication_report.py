"""Independent local final audit for control replication1071; launches no games."""

import argparse
import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

from .duo_continuation_report import check_pairs, digest, read, tensor_digest
from .duo_crossplay import opening_book, paired_interval
from .duo_integrated_report import audit_training
from .duo_replay import pack_replay
from .duo_teacher_report import audit_external, require, rows

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
STATE_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
PARENT_DIR = ROOT / "runs/duo-continuation-final-20261026T2007/duo-continuation-865"
ARMS = ("control1024",)
SEED = 2608900


def audit(run, apps_path, app_id, billing_path):
    run = Path(run)
    require(run.name == "duo-control-replication-1071", "wrong run")
    apps = read(apps_path)
    require(all(int(a["tasks"]) == 0 for a in apps), "compute still active")
    require(
        any(a["app_id"] == app_id and a["state"] == "stopped" for a in apps), "study not stopped"
    )
    require(not (run / "failure.json").exists(), "failure marker")
    require(read(run / "wrapper-finished.json")["seconds"] < 14400, "hard cap")
    execution = read(run / "execution.json")
    ledger = rows(ROOT / "runs/modal-control-replication-cost-ledger.jsonl")
    require(
        execution == next(r for r in ledger if r["stage"] == "replication"),
        "execution/ledger mismatch",
    )
    require(sum(Decimal(str(r["reserved_usd"])) for r in ledger) <= 11, "phase reservation cap")
    manifest = read(ROOT / "runs/integrated-payload-1070-sha256.json")
    require(execution["payload_sha256"] == manifest, "payload manifest changed")
    for name, expected in manifest.items():
        require(
            digest(ROOT / "runs/integrated-payload-1070" / name) == expected,
            "source payload changed: " + name,
        )
        if name.startswith("python/") and not name.endswith("__init__.py"):
            source = name
        elif name.startswith("native/"):
            source = name[len("native/") :]
        elif name.endswith("__init__.py"):
            continue  # Explicit generated namespace overlay, pinned by manifest.
        else:
            source = "cloud/" + name
        committed = subprocess.check_output(
            ["git", "show", execution["source_commit"] + ":" + source], cwd=ROOT
        )
        require(
            hashlib.sha256(committed).hexdigest() == expected, "source commit mismatch: " + source
        )
    committed = subprocess.check_output(
        ["git", "show", execution["source_commit"] + ":cloud/modal_control_replication.py"],
        cwd=ROOT,
    )
    require(
        hashlib.sha256(committed).hexdigest() == execution["runner_sha256"],
        "runner commit mismatch",
    )
    for source, key in (
        ("cloud/run_control_replication.py", "helper_sha256"),
        ("docs/control-replication-protocol.json", "protocol_sha256"),
        ("runs/control-replication-preserved-before.json", "preserved_sha256"),
    ):
        committed = subprocess.check_output(
            ["git", "show", execution["source_commit"] + ":" + source], cwd=ROOT
        )
        require(
            hashlib.sha256(committed).hexdigest() == execution[key] == digest(ROOT / source),
            "committed replication source changed: " + source,
        )
    declared = read(ROOT / "docs/control-replication-protocol.json")
    require(execution["config"] == declared["config"], "declared configuration differs")
    require(execution["base_image"] == "im-yX3zmnEKl2EgUtndThw9li", "inherited image differs")
    validation = read(ROOT / "runs/integrated-validation-verified.json")
    require(
        validation["passed"] and validation["payload_sha256"] == manifest, "validation mismatched"
    )
    require(
        read(run / "native-build.json") == read(Path(validation["run"]) / "native-build.json"),
        "native build differs from validated binary",
    )
    protocol, heldout, finished = [
        read(run / (n + ".json")) for n in ("study-protocol", "heldout-protocol", "finished")
    ]
    runtime = validation["selected_runtime"]
    require(execution["selected_runtime"] == protocol["runtime"] == runtime, "runtime changed")
    require(
        heldout
        == {
            "checkpoint_sha256": finished["checkpoint_sha256"],
            "parent_sha256": MODEL_SHA,
            "seed": SEED,
            "pairs": 100,
            "simulations": 2048,
            "levels": [3, 4],
            "precision": "bfloat16",
            "inference_backend": "cuda_graphs",
        },
        "heldout changed",
    )
    require(
        finished["parent_sha256"] == protocol["parent_sha256"] == MODEL_SHA
        and finished["source_resume_sha256"] == protocol["source_resume_sha256"] == STATE_SHA,
        "parent lineage changed",
    )
    require(
        finished["external_games"] == 800 and finished["head_to_head_games"] == 400, "game count"
    )
    preserved = read(ROOT / "runs/control-replication-preserved-before.json")
    for path, expected in preserved.items():
        require(digest(ROOT / path) == expected, "preserved model changed: " + path)
    require(
        digest(PARENT_DIR / "frozen-strong1024.pt") == MODEL_SHA
        and digest(PARENT_DIR / "strong1024/resume.pt") == STATE_SHA,
        "source parent changed",
    )
    parent = torch.load(PARENT_DIR / "frozen-strong1024.pt", map_location="cpu", weights_only=True)[
        "model_state"
    ]
    training = {}
    for arm, full, cap in (("control1024", 1024, 6000),):
        config = protocol["configs"][arm]
        for key, value in {
            "seed": 1071,
            "rounds": 64,
            "games_per_round": 256,
            "workers": 6,
            "live_games": 64,
            "fast_probability": 0.8,
            "fast_simulations": 32,
            "full_simulations": full,
            "leaf_batch": 8,
            "inference_batch": 256,
            "batch_wait_ms": 0,
            "replay_positions": 100000,
            "training_ratio": 3,
            "minibatch_size": 256,
            "learning_rate": 0.001,
            "policy_loss_weight": 0.5,
            "validation_seed": 2608300,
            "test_seed": SEED,
            "eval_every": 16,
            "validation_openings": 32,
            "validation_simulations": 512,
            "max_seconds": cap,
            "architecture": "conv",
            "precision": "bfloat16",
            "device": "cuda",
        }.items():
            require(config[key] == value, "config mismatch: " + key)
        training[arm] = audit_training(run, arm, config, parent)
        require(
            finished["training"][arm] == read(run / arm / "completed.json"), "completion disagrees"
        )
        state = torch.load(run / arm / "resume.pt", map_location="cpu", weights_only=False)
        require(
            state["runtime"]
            == dict(inference_transport="shared_memory", inference_cache_entries=0, **runtime),
            "saved runtime differs",
        )
        for row in training[arm]["rows"]:
            metrics = row["metrics"]
            require(
                metrics["graph_requested_positions"] == metrics["gpu_positions"]
                and metrics["graph_fallback_positions"] == 0,
                "captured inference work differs",
            )
        require(
            digest(run / f"frozen-{arm}.pt") == finished["checkpoint_sha256"][arm],
            "frozen hash changed",
        )
    require(
        protocol["configs"] == {"control1024": declared["config"]},
        "scientific configuration differs",
    )
    initial = read(run / "initialization-audit.json")
    source_state = torch.load(
        PARENT_DIR / "strong1024/resume.pt", map_location="cpu", weights_only=False
    )
    require(
        initial["round"] == initial["games"] == initial["optimizer_state_entries"] == 0,
        "initial counters/optimizer were not reset",
    )
    require(
        initial["replay_positions"] == 100000 and initial["cuda_rng_count"] == 1,
        "initial replay/device count changed",
    )
    require(
        initial["model_tensor_sha256"] == tensor_digest(parent), "initial tensor lineage changed"
    )
    require(
        initial["packed_replay_tensor_sha256"]
        == tensor_digest(
            {
                k: v
                for k, v in pack_replay(source_state["replay"]).items()
                if isinstance(v, torch.Tensor)
            }
        ),
        "inherited replay content/order changed",
    )
    require(
        initial["numpy_rng"] == np.random.default_rng(1071).bit_generator.state,
        "fresh NumPy seed differs",
    )
    expected_rng = torch.Generator().manual_seed(1071).get_state().numpy().tobytes()
    require(
        initial["torch_rng_sha256"] == hashlib.sha256(expected_rng).hexdigest(),
        "fresh CPU RNG differs",
    )
    require(
        all(g["lr"] == 0.001 and g["weight_decay"] == 0.0001 for g in initial["optimizer_groups"]),
        "fresh AdamW hyperparameters differ",
    )
    del source_state
    cases, replays = {}, {}
    for name, model_sha in ({"parent": MODEL_SHA} | finished["checkpoint_sha256"]).items():
        for level in (3, 4):
            case = f"{name}-pentobi{level}"
            cases[case], replays[case] = audit_external(
                run / "heldout" / case, level, model_sha, seed=SEED
            )
    require(cases == read(run / "heldout-results.json"), "external summary differs")
    differences = {}
    for level in (3, 4):
        for left, right in (("control1024", "parent"),):
            delta = np.asarray(cases[f"{left}-pentobi{level}"]["opening_pair_scores"]) - np.asarray(
                cases[f"{right}-pentobi{level}"]["opening_pair_scores"]
            )
            differences[f"{left}-minus-{right}-pentobi{level}"] = {
                "score_difference": float(delta.mean()),
                "nominal_paired_95pct_interval": paired_interval(delta),
            }
    require(differences == read(run / "paired-differences.json"), "paired differences differ")
    head = {}
    for case in (
        "control-vs-parent-search2048",
        "control-vs-parent-raw",
    ):
        row = read(run / f"heldout-{case}.json")
        check_pairs(row, 100)
        budget = 2048 if case.endswith("search2048") else 0
        require(
            read(run / f"heldout-{case}-protocol.json")
            == {
                "seed": SEED,
                "pairs": 100,
                "opening_book_sha256": hashlib.sha256(
                    json.dumps(opening_book(100, seed=SEED)).encode()
                ).hexdigest(),
                "left_sha256": finished["checkpoint_sha256"]["control1024"],
                "right_sha256": MODEL_SHA,
                "simulations": budget,
                "precision": "bfloat16",
                "inference_backend": "cuda_graphs",
            },
            "head-to-head protocol differs",
        )
        require(
            row["learner_simulations"] == row["opponent_simulations"] == budget,
            "head-to-head budget",
        )
        require(
            np.isfinite(row["mean_margin"]) and row["seconds"] > 0, "head-to-head runtime/margin"
        )
        require(
            all(np.isfinite(v) and v >= 0 for v in row["metrics"].values()), "head-to-head counters"
        )
        sims = row["metrics"].get("simulations", 0)
        require(
            sims > 0 and sims % budget == 0 if budget else sims == 0, "head-to-head simulations"
        )
        head[case] = row
    billing = read(billing_path)
    costs = {}
    for row in billing:
        if row["description"] == "blokus-control-replication-1071":
            app = row["object_id"]
            costs[app] = str(Decimal(costs.get(app, "0")) + Decimal(str(row["cost"])))
    require(app_id in costs, "billing omits successful compute")
    require(sum(Decimal(v) for v in costs.values()) <= 11, "actual phase metering exceeds cap")
    primary = differences["control1024-minus-parent-pentobi4"]["nominal_paired_95pct_interval"]
    return {
        "passed": True,
        "run": str(run.resolve()),
        "initialization": initial,
        "execution": execution,
        "external": cases,
        "external_replays": replays,
        "training": training,
        "head_to_head": head,
        "head_to_head_traces_available": False,
        "paired_differences": differences,
        "primary_verdict": "improvement"
        if primary[0] > 0
        else "negative"
        if primary[1] < 0
        else "inconclusive",
        "provider_metering_usd": costs,
        "reserved_usd": sum(r["reserved_usd"] for r in ledger),
        "preserved": preserved,
        "finished": finished,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--apps", type=Path, required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--billing", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.apps, args.app_id, args.billing)
    out = args.run.parent / "verification.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    (args.run.parent / "download-sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(args.run)): digest(p)
                for p in sorted(args.run.rglob("*"))
                if p.is_file()
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "verification": str(out),
                "primary_verdict": result["primary_verdict"],
                "external_games_replayed": 800,
            }
        )
    )


if __name__ == "__main__":
    main()
