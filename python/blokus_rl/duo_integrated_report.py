"""Independent local final audit for integrated study1070; no training/game launches."""

import argparse
import hashlib
import json
import math
import subprocess
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

from .duo_continuation_report import check_pairs, digest, read, tensor_digest
from .duo_crossplay import paired_interval
from .duo_replay import replay_length
from .duo_teacher_report import audit_external, require, rows, same_tensors

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
STATE_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
PARENT_DIR = ROOT / "runs/duo-continuation-final-20261026T2007/duo-continuation-865"
ARMS = ("control1024", "teacher2048")
SEED = 2607900


def audit_training(run, arm, config, parent):
    protocol = read(run / arm / "protocol.json")
    require(
        protocol["config"] == config
        and protocol["reference_sha256"] == MODEL_SHA
        and protocol["source_resume_sha256"] == STATE_SHA
        and protocol["inherited_replay_positions"] == 100000
        and protocol["parent_round"] == 128
        and protocol["parent_games"] == 32768
        and protocol["selected_parent_update"] == 112
        and protocol["optimizer_reset"] is True
        and protocol["parameters"] == 52978,
        "branch initialization changed",
    )
    require(digest(run / arm / "initial.pt") == MODEL_SHA, "initial parent weights changed")
    progress = rows(run / arm / "progress.jsonl")
    completed = read(run / arm / "completed.json")
    count = completed["rounds"]
    require(0 < count <= 64, "invalid training round count")
    require([r["round"] for r in progress] == list(range(1, count + 1)), "training rows missing")
    require(completed["games"] == 256 * count, "training games disagree")
    require(
        completed["reason"] in ("round_limit", "time_budget")
        and (completed["reason"] != "round_limit" or count == 64),
        "wrong training stop reason",
    )
    selected_round, selected_score = 0, 0.5
    validations = []
    for row in progress:
        require(
            row["games"] == 256 * row["round"] and row["replay_positions"] == 100000,
            "training round counters changed",
        )
        steps = max(1, math.ceil(3 * row["new_training_positions"] / 256))
        require(
            row["gradient_steps"] == steps and row["gradient_examples"] == 256 * steps,
            "gradient counters disagree",
        )
        require(all(np.isfinite(v) for v in row["losses"].values()), "nonfinite training loss")
        metrics = row["metrics"]
        fast = metrics["simulations"] - config["full_simulations"] * row["new_training_positions"]
        require(fast >= 0 and fast % 32 == 0, "training search budgets disagree")
        require(
            metrics["gpu_positions"] == metrics["neural_positions"]
            and all(np.isfinite(v) and v >= 0 for v in metrics.values()),
            "training inference counters disagree",
        )
        scheduled = row["round"] % 16 == 0 or row["round"] == 64
        require(("validation" in row) == scheduled, "validation schedule changed")
        if scheduled:
            validations.append((row["round"], row["validation"]))
    final_validation = run / arm / "final-validation.json"
    needs_final = completed["reason"] == "time_budget" and count % 16 != 0
    require(final_validation.exists() == needs_final, "final validation schedule changed")
    if needs_final:
        validations.append((count, read(final_validation)))
    for round_, result in validations:
        check_pairs(result, 32)
        require(
            result["learner_simulations"] == result["opponent_simulations"] == 512,
            "validation playing budget changed",
        )
        if result["score_rate"] > selected_score:
            selected_score, selected_round = result["score_rate"], round_
    require(completed["best_validation_score"] == selected_score, "checkpoint selection disagrees")
    frozen = torch.load(run / f"frozen-{arm}.pt", map_location="cpu", weights_only=True)
    if selected_round == 0:
        require(
            frozen["update"] == 112 and same_tensors(frozen["model_state"], parent),
            "selected parent is not unchanged",
        )
    else:
        require(
            frozen["update"] == selected_round and frozen["config"] == config,
            "selected checkpoint experience/configuration disagrees",
        )
    require(digest(run / f"frozen-{arm}.pt") == digest(run / arm / "best.pt"), "freeze changed")
    state = torch.load(run / arm / "resume.pt", map_location="cpu", weights_only=False)
    latest = torch.load(run / arm / "latest.pt", map_location="cpu", weights_only=True)
    final = torch.load(run / arm / "final.pt", map_location="cpu", weights_only=True)
    steps = sum(r["gradient_steps"] for r in progress)
    require(
        state["config"] == config
        and state["reference_sha256"] == MODEL_SHA
        and state["round"] == count
        and state["games"] == 256 * count
        and replay_length(state["replay"]) == 100000
        and {int(v["step"]) for v in state["optimizer"]["state"].values()} == {steps},
        "atomic state/optimizer disagrees with all retained rounds",
    )
    require(
        latest["update"] == final["update"] == count
        and latest["config"] == final["config"] == config
        and same_tensors(state["model_state"], latest["model_state"])
        and same_tensors(state["model_state"], final["model_state"]),
        "latest/final/atomic weights disagree",
    )
    if selected_round == count:
        require(same_tensors(frozen["model_state"], final["model_state"]), "selected final changed")
    elif selected_round:
        selected = torch.load(
            run / arm / f"round-{selected_round:03d}.pt", map_location="cpu", weights_only=True
        )
        require(
            same_tensors(frozen["model_state"], selected["model_state"]), "selected round changed"
        )
    totals = {
        key: sum(r[key] for r in progress)
        for key in (
            "new_training_positions",
            "gradient_steps",
            "gradient_examples",
            "training_seconds",
        )
    }
    totals.update(
        {
            key: sum(r["metrics"][key] for r in progress)
            for key in (
                "simulations",
                "neural_positions",
                "gpu_positions",
                "gpu_batches",
                "collect_seconds",
            )
        }
    )
    return (
        completed
        | totals
        | {
            "selected_additional_round": selected_round,
            "selected_additional_games": 256 * selected_round,
            "selected_ancestral_plus_new_games": 61440 + 256 * selected_round,
            "soft_cap_seconds": config["max_seconds"],
            "soft_cap_overshoot_seconds": max(
                0, completed["elapsed_seconds"] - config["max_seconds"]
            ),
            "mean_gpu_batch": totals["gpu_positions"] / totals["gpu_batches"],
            "selected_tensor_sha256": tensor_digest(frozen["model_state"]),
            "rows": progress,
        }
    )


def audit(run, apps_path, app_id, billing_path):
    run = Path(run)
    require(run.name == "duo-integrated-teacher-1070", "wrong run")
    apps = read(apps_path)
    require(all(int(a["tasks"]) == 0 for a in apps), "compute still active")
    require(
        any(a["app_id"] == app_id and a["state"] == "stopped" for a in apps), "study not stopped"
    )
    require(not (run / "failure.json").exists(), "failure marker")
    require(read(run / "wrapper-finished.json")["seconds"] < 28800, "hard cap")
    execution = read(run / "execution.json")
    ledger = rows(ROOT / "runs/modal-integrated-cost-ledger.jsonl")
    require(
        execution == next(r for r in ledger if r["stage"] == "study"), "execution/ledger mismatch"
    )
    require(sum(Decimal(str(r["reserved_usd"])) for r in ledger) <= 23, "phase reservation cap")
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
        ["git", "show", execution["source_commit"] + ":cloud/modal_integrated.py"], cwd=ROOT
    )
    require(
        hashlib.sha256(committed).hexdigest() == execution["runner_sha256"],
        "runner commit mismatch",
    )
    validation = read(ROOT / "runs/integrated-validation-verified.json")
    require(
        validation["passed"] and validation["payload_sha256"] == manifest, "validation mismatched"
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
        finished["external_games"] == 1200 and finished["head_to_head_games"] == 800, "game count"
    )
    preserved = read(ROOT / "runs/duo-teacher-user-stopped-20261028T032155Z/stop-verified.json")[
        "preserved"
    ]
    preserved |= read(ROOT / "runs/search-preserved-checkpoints.json")
    old = ROOT / "runs/duo-teacher-user-stopped-20261028T032155Z"
    for arm, expected in {
        "control1024": "facc6301b99ea948a3d85708b4a72e3da8bf0ac139c7601d51c90b9515f409c2",
        "teacher2048": "45423ab8085724442288386e10fa8b5b38ee3c7479e1ced2ee5e2f146a5ef9f4",
    }.items():
        preserved[str(old / "duo-teacher-search-1066" / arm / "resume.pt")] = expected
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
    for arm, full, cap in (("control1024", 1024, 6000), ("teacher2048", 2048, 10000)):
        config = protocol["configs"][arm]
        for key, value in {
            "seed": 1070,
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
            "validation_seed": 2607300,
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
    configs = [dict(protocol["configs"][a]) for a in ARMS]
    for c in configs:
        c.pop("full_simulations")
        c.pop("max_seconds")
    require(configs[0] == configs[1], "unintended branch differences")
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
        for left, right in (
            ("teacher2048", "parent"),
            ("teacher2048", "control1024"),
            ("control1024", "parent"),
        ):
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
        "teacher-vs-control-search2048",
        "teacher-vs-control-raw",
        "teacher-vs-parent-raw",
        "control-vs-parent-raw",
    ):
        row = read(run / f"heldout-{case}.json")
        check_pairs(row, 100)
        budget = 2048 if case.endswith("search2048") else 0
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
        if row["description"] == "blokus-integrated-speedups":
            app = row["object_id"]
            costs[app] = str(Decimal(costs.get(app, "0")) + Decimal(str(row["cost"])))
    require(app_id in costs and validation["app_id"] in costs, "billing omits successful compute")
    primary = differences["teacher2048-minus-parent-pentobi4"]["nominal_paired_95pct_interval"]
    return {
        "run": str(run.resolve()),
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
                "external_games_replayed": 1200,
            }
        )
    )


if __name__ == "__main__":
    main()
