"""Audit the completed teacher-search study locally; never train or launch games."""

import argparse
import hashlib
import json
import math
import subprocess
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

from ._native import DuoGame
from .duo_continuation_report import check_pairs, digest, read, tensor_digest
from .duo_crossplay import opening_book, paired_interval
from .duo_pentobi_ladder import summarize
from .duo_pentobi_level_report import BINARY_SHA, EVALUATOR_SHA, MODEL_SHA

RUN = "duo-teacher-search-1066"
ARMS = ("control1024", "teacher2048")
APPS = {
    "startup": "ap-HNw7nCM4nv87Q9cqQYx0Hm",
    "cancelled": "ap-jflsIhwP1giVlNWVH3ArFn",
    "recovery": "ap-PxLumtPesHAVP1q9Gz82TA",
}
STATE_SHA = "df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911"
RESUME_SHA = "a3a1875a9379024b9d989e52b0304caa89cefe3cf241d733dbaf7044a9162466"
PARENT_DIR = Path("runs/duo-continuation-final-20261026T2007/duo-continuation-865")
INTERRUPTED = Path("runs/duo-teacher-interrupted-20261027T0311")
SEED, PAIRS, BUDGET = 2605900, 100, 2048


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def same_tensors(first, second):
    return first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)


def audit_external(root, level, model_sha, *, seed=SEED, pairs=PAIRS, budget=BUDGET):
    """Replay every saved external move and independently recount searched roots."""
    root = Path(root)
    protocol, completed = read(root / "protocol.json"), read(root / "completed.json")
    require(
        protocol["config"]
        == {
            "first_level": level,
            "last_level": level,
            "opening_pairs": pairs,
            "seed": seed,
            "device": "cuda",
            "precision": "bfloat16",
            "simulations": budget,
        },
        "external configuration changed",
    )
    book = opening_book(pairs, seed=seed)
    require(protocol["openings"] == book, "external opening book changed")
    require(
        protocol["checkpoint_sha256"] == digest(root / "model.pt") == model_sha,
        "external checkpoint changed",
    )
    require(protocol["pentobi_binary_sha256"] == BINARY_SHA, "external binary changed")
    require(
        protocol["pentobi"]
        == "31.0; Duo; one thread; no book or resignation; default symmetric draw",
        "external opponent settings changed",
    )
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    require(
        completed["protocol_sha256"] == fingerprint
        and completed["levels"] == [str(level)]
        and completed["games"] == 2 * pairs,
        "external completion disagrees",
    )
    directory = root / f"level-{level:02}"
    records = [read(p) for p in sorted(directory.glob("pair-*.json"))]
    require([r["index"] for r in records] == list(range(pairs)), "missing/duplicate pairs")
    roots = 0
    for row in records:
        require(
            row["level"] == level and row["protocol_sha256"] == fingerprint,
            "external pair routed incorrectly",
        )
        require([g["seat"] for g in row["games"]] == [0, 1], "external seats changed")
        for detail in row["games"]:
            seat = detail["seat"]
            require(
                detail["seed"] == seed + row["index"] + seat * 1000000,
                "external seed changed",
            )
            require(detail["actions"][:4] == book[row["index"]], "external opening changed")
            game, decisions = DuoGame(), 0
            for ply, action in enumerate(detail["actions"]):
                decisions += ply >= 4 and game.current_player == seat
                game.step(action)  # Replay through native legality checks.
            require(game.terminated, "external trace is incomplete")
            require(list(game.scores) == detail["scores"], "external score disagrees with replay")
            require(
                game.scores[seat] - game.scores[1 - seat] == detail["margin"],
                "external margin disagrees with replay",
            )
            require(
                detail["policy_metrics"]["simulations"] == budget * decisions,
                "external actual simulation count changed",
            )
            require(
                detail["seconds"] >= 0
                and all(np.isfinite(v) and v >= 0 for v in detail["policy_metrics"].values()),
                "invalid external timing/counters",
            )
            roots += decisions
    result = summarize(records, pairs)
    result["win_rate"] = result["wins"] / result["games"]
    require(result == read(directory / "summary.json"), "external summary disagrees")
    require(read(root / "results.json") == {str(level): result}, "external result disagrees")
    return result, {
        "games_replayed": 2 * pairs,
        "searched_decisions": roots,
        "actual_simulations": budget * roots,
        "protocol_sha256": fingerprint,
    }


def audit_history(run):
    """Preserve historical failure records; an old cancellation is not a new failure."""
    require(not (run / "recovery-failure.json").exists(), "new recovery failure marker")
    manifest = read(INTERRUPTED / "download-sha256.json")
    previous = INTERRUPTED / RUN
    mutable = {
        "control1024/best.pt",
        "control1024/latest.pt",
        "control1024/resume.pt",
        "control1024/status.json",
        "control1024/progress.jsonl",
        "telemetry.jsonl",
    }
    for path, expected in manifest.items():
        require(digest(previous / path) == expected, "interrupted audit copy changed: " + path)
        if path not in mutable:
            require(digest(run / path) == expected, "historical artifact changed: " + path)
    old_progress = (previous / "control1024/progress.jsonl").read_bytes()
    require(
        (run / "control1024/progress.jsonl").read_bytes().startswith(old_progress),
        "original progress rows were rewritten",
    )
    require(digest(previous / "control1024/resume.pt") == RESUME_SHA, "wrong saved recovery state")
    ledger = rows("runs/modal-teacher-search-cost-ledger.jsonl")
    for name, prefix in (("execution", RUN), ("recovery-execution", RUN + "-recovery")):
        execution = read(run / f"{name}.json")
        require(execution in ledger, "execution missing from reservation ledger")
        require(
            execution["parent_sha256"] == MODEL_SHA
            and execution["source_resume_sha256"] == STATE_SHA
            and execution["evaluator_sha256"] == EVALUATOR_SHA
            and execution["binary_sha256"] == BINARY_SHA,
            "execution lineage changed",
        )
        for suffix, source in (
            ("runner", "cloud/modal_teacher_search.py"),
            ("helper", "python/blokus_rl/duo_continuation.py"),
        ):
            archived = Path(f"runs/{prefix}-{suffix}.py")
            expected = execution[f"{suffix}_sha256"]
            committed = subprocess.check_output(
                ["git", "show", execution["source_commit"] + ":" + source]
            )
            require(
                digest(archived) == hashlib.sha256(committed).hexdigest() == expected,
                "executed source archive/commit changed",
            )
        for module, expected in execution["training_module_sha256"].items():
            require(digest(f"python/blokus_rl/{module}.py") == expected, "training module changed")
    recovery = read(run / "recovery-execution.json")
    require(
        recovery["recovery"] is True
        and recovery["expected_resume_sha256"] == RESUME_SHA
        and recovery["timeout_seconds"] == 42000,
        "wrong recovery execution",
    )
    require(
        digest("runs/duo-strong-pentobi-steps-2599000-module.py") == EVALUATOR_SHA,
        "immutable evaluator archive changed",
    )
    return float(sum(Decimal(str(row["reserved_usd"])) for row in ledger))


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
        and len(state["replay"]) == 100000
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


def audit(run, apps_path, billing_path):
    run = Path(run)
    require(run.name == RUN, "wrong study directory")
    finished, protocol, heldout = [
        read(run / f"{name}.json") for name in ("finished", "study-protocol", "heldout-protocol")
    ]
    apps = read(apps_path)
    stopped = {
        name: next(a for a in apps if a["app_id"] == app_id) for name, app_id in APPS.items()
    }
    require(
        all(a["state"] == "stopped" and int(a["tasks"]) == 0 for a in stopped.values()),
        "study apps must be stopped before final audit",
    )
    reservation = audit_history(run)
    require(reservation <= 32, "reservation ceiling exceeded")
    parent_path, source_path = (
        PARENT_DIR / "frozen-strong1024.pt",
        PARENT_DIR / "strong1024/resume.pt",
    )
    preserved = read("runs/search-preserved-checkpoints.json") | {
        str(parent_path): MODEL_SHA,
        str(source_path): STATE_SHA,
        "runs/duo-deep-local-764/conv.pt": "0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86",
        str(
            PARENT_DIR / "frozen-standard256.pt"
        ): "4cbeb255cb5688a6b931ac4ac6964f21e3c6430be2e6dd558e6d7c9a14d43b19",
    }
    for path, sha in preserved.items():
        require(digest(path) == sha, "preserved checkpoint changed: " + path)
    require(
        protocol == read(INTERRUPTED / RUN / "study-protocol.json"), "original protocol changed"
    )
    require(
        heldout
        == {
            "checkpoint_sha256": finished["checkpoint_sha256"],
            "parent_sha256": MODEL_SHA,
            "seed": SEED,
            "pairs": PAIRS,
            "simulations": BUDGET,
            "levels": [3, 4],
            "precision": "bfloat16",
        },
        "held-out protocol changed",
    )
    require(
        finished["parent_sha256"] == MODEL_SHA
        and finished["source_resume_sha256"] == STATE_SHA
        and finished["external_games"] == 1200
        and finished["head_to_head_games"] == 800,
        "study completion disagrees",
    )
    old_seconds = read(run / "failure.json")["seconds"]
    require(
        0 < finished["recovery_seconds"] < 42000
        and abs(finished["seconds"] - finished["recovery_seconds"] - old_seconds) < 0.01,
        "recovery timing accounting disagrees",
    )
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)["model_state"]
    training = {arm: audit_training(run, arm, protocol["configs"][arm], parent) for arm in ARMS}
    for arm in ARMS:
        require(
            finished["checkpoint_sha256"][arm] == digest(run / f"frozen-{arm}.pt"),
            "frozen file hash changed",
        )
        require(
            finished["training"][arm] == read(run / arm / "completed.json"),
            "training completion markers disagree",
        )
    cases, replays = {}, {}
    for model, model_sha in ({"parent": MODEL_SHA} | finished["checkpoint_sha256"]).items():
        for level in (3, 4):
            case = f"{model}-pentobi{level}"
            cases[case], replays[case] = audit_external(run / "heldout" / case, level, model_sha)
    require(cases == read(run / "heldout-results.json"), "top-level external results disagree")
    differences = {}
    for left, right in (
        ("teacher2048", "parent"),
        ("teacher2048", "control1024"),
        ("control1024", "parent"),
    ):
        label = {"teacher2048": "teacher", "control1024": "control", "parent": "parent"}
        delta = np.asarray(cases[f"{left}-pentobi4"]["opening_pair_scores"]) - np.asarray(
            cases[f"{right}-pentobi4"]["opening_pair_scores"]
        )
        differences[f"{label[left]}-minus-{label[right]}-pentobi4"] = {
            "score_difference": float(delta.mean()),
            "nominal_paired_95pct_interval": paired_interval(delta),
        }
    require(differences == read(run / "paired-differences.json"), "paired differences disagree")
    head_to_head = {}
    for case in (
        "teacher-vs-control-search2048",
        "teacher-vs-control-raw",
        "teacher-vs-parent-raw",
        "control-vs-parent-raw",
    ):
        row = read(run / f"heldout-{case}.json")
        check_pairs(row, PAIRS)
        budget = BUDGET if case.endswith("search2048") else 0
        require(
            row["learner_simulations"] == row["opponent_simulations"] == budget,
            "head-to-head budget changed",
        )
        require(
            np.isfinite(row["mean_margin"]) and row["seconds"] >= 0,
            "invalid head-to-head timing/margin",
        )
        require(
            all(np.isfinite(v) and v >= 0 for v in row["metrics"].values()),
            "invalid head-to-head counters",
        )
        sims = row["metrics"].get("simulations", 0)
        require(
            (sims > 0 and sims % budget == 0) if budget else sims == 0,
            "head-to-head simulation totals disagree",
        )
        head_to_head[case] = row
    billing = read(billing_path)
    costs = {}
    for name, app in APPS.items():
        charges = [Decimal(str(row["cost"])) for row in billing if row["object_id"] == app]
        require(bool(charges), "billing snapshot omits study app: " + app)
        costs[name] = str(sum(charges))
    primary = differences["teacher-minus-parent-pentobi4"]["nominal_paired_95pct_interval"]
    return {
        "run_name": RUN,
        "external": cases,
        "external_replays": replays,
        "head_to_head": head_to_head,
        "head_to_head_traces_available": False,
        "training": training,
        "paired_differences": differences,
        "primary_verdict": "improvement"
        if primary[0] > 0
        else "negative"
        if primary[1] < 0
        else "inconclusive",
        "checkpoint_sha256": finished["checkpoint_sha256"],
        "preserved": preserved,
        "stopped_apps": stopped,
        "provider_metering_usd": costs,
        "net_reservation_usd": reservation,
        "finished": finished,
        "openings_sha256": hashlib.sha256(
            json.dumps(opening_book(PAIRS, seed=SEED)).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--apps", type=Path, required=True)
    parser.add_argument("--billing", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.apps, args.billing)
    output = args.run.parent / "verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        str(p.relative_to(args.run)): digest(p) for p in sorted(args.run.rglob("*")) if p.is_file()
    }
    (args.run.parent / "download-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verification": str(output),
                "primary_verdict": result["primary_verdict"],
                "external_games_replayed": sum(
                    r["games_replayed"] for r in result["external_replays"].values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
