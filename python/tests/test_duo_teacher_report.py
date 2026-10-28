"""Regression checks for independent saved-game and recovery accounting audits."""

import hashlib
import json
from dataclasses import asdict

import pytest
import torch
from blokus_rl import duo_teacher_report as report
from blokus_rl._native import DuoGame
from blokus_rl.duo_crossplay import opening_book, paired_interval
from blokus_rl.duo_deep import DeepConfig
from blokus_rl.duo_pentobi_ladder import summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def external(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    (root / "model.pt").write_bytes(b"frozen model fixture")
    sha = report.digest(root / "model.pt")
    book = opening_book(2, seed=report.SEED)
    protocol = {
        "config": {
            "first_level": 4,
            "last_level": 4,
            "opening_pairs": 2,
            "seed": report.SEED,
            "device": "cuda",
            "precision": "bfloat16",
            "simulations": 2048,
        },
        "openings": book,
        "checkpoint_sha256": sha,
        "pentobi_binary_sha256": report.BINARY_SHA,
        "pentobi": "31.0; Duo; one thread; no book or resignation; default symmetric draw",
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write(root / "protocol.json", protocol)
    write(root / "completed.json", {"protocol_sha256": fingerprint, "levels": ["4"], "games": 4})
    records = []
    for index, opening in enumerate(book):
        details = []
        for seat in (0, 1):
            game, actions, roots = DuoGame(), list(opening), 0
            for action in opening:
                game.step(action)
            while not game.terminated:
                roots += game.current_player == seat
                action = game.legal_actions()[0]
                game.step(action)
                actions.append(action)
            details.append(
                {
                    "seat": seat,
                    "seed": report.SEED + index + seat * 1000000,
                    "actions": actions,
                    "scores": list(game.scores),
                    "margin": game.scores[seat] - game.scores[1 - seat],
                    "seconds": 1,
                    "policy_metrics": {"simulations": roots * 2048},
                }
            )
        row = {"index": index, "level": 4, "protocol_sha256": fingerprint, "games": details}
        records.append(row)
        write(root / f"level-04/pair-{index:03}.json", row)
    summary = summarize(records, 2)
    summary["win_rate"] = summary["wins"] / summary["games"]
    write(root / "level-04/summary.json", summary)
    write(root / "results.json", {"4": summary})
    return root, sha


def test_external_replays_and_recounts(external):
    root, sha = external
    summary, audit = report.audit_external(root, 4, sha, pairs=2)
    assert summary["games"] == audit["games_replayed"] == 4
    assert audit["actual_simulations"] == 2048 * audit["searched_decisions"]


@pytest.mark.parametrize(
    "corruption", ("margin", "seed", "budget", "opening", "illegal", "missing")
)
def test_external_rejects_corruption(external, corruption):
    root, sha = external
    path = root / "level-04/pair-000.json"
    row = report.read(path)
    game = row["games"][0]
    if corruption == "margin":
        game["margin"] += 1
    elif corruption == "seed":
        game["seed"] += 1
    elif corruption == "budget":
        game["policy_metrics"]["simulations"] -= 1
    elif corruption == "opening":
        game["actions"][0] = -123
    elif corruption == "illegal":
        game["actions"][4] = -123
    else:
        path.unlink()
    if corruption != "missing":
        write(path, row)
    with pytest.raises((ValueError, OverflowError)):
        report.audit_external(root, 4, sha, pairs=2)


@pytest.mark.parametrize("improves", (False, True))
def test_training_parent_selection_and_final_validation(tmp_path, monkeypatch, improves):
    arm = "control1024"
    out = tmp_path / arm
    out.mkdir()
    config = asdict(DeepConfig(rounds=64, full_simulations=1024, max_seconds=9000))
    parent, latest = {"weight": torch.tensor([1.0])}, {"weight": torch.tensor([2.0])}
    parent_checkpoint = {"update": 112, "model_state": parent}
    torch.save(parent_checkpoint, out / "initial.pt")
    monkeypatch.setattr(report, "MODEL_SHA", report.digest(out / "initial.pt"))
    write(
        out / "protocol.json",
        {
            "config": config,
            "reference_sha256": report.MODEL_SHA,
            "source_resume_sha256": report.STATE_SHA,
            "inherited_replay_positions": 100000,
            "parent_round": 128,
            "parent_games": 32768,
            "selected_parent_update": 112,
            "optimizer_reset": True,
            "parameters": 52978,
        },
    )
    progress = {
        "round": 1,
        "games": 256,
        "new_training_positions": 1,
        "replay_positions": 100000,
        "gradient_steps": 1,
        "gradient_examples": 256,
        "training_seconds": 0.1,
        "losses": {"policy_loss": 1.0, "value_loss": 0.5},
        "metrics": {
            "simulations": 1056,
            "neural_positions": 1000,
            "gpu_positions": 1000,
            "gpu_batches": 10,
            "collect_seconds": 1,
        },
    }
    (out / "progress.jsonl").write_text(json.dumps(progress) + "\n")
    outcomes = [[1 if improves else 0.5] * 2 for _ in range(32)]
    validation = {
        "opening_pair_scores": outcomes,
        "games": 64,
        "opening_pairs": 32,
        "wins": 64 if improves else 0,
        "draws": 0 if improves else 64,
        "losses": 0,
        "score_rate": 1 if improves else 0.5,
        "score_rate_95pct_cluster_bootstrap": paired_interval(outcomes),
        "learner_simulations": 512,
        "opponent_simulations": 512,
    }
    write(out / "final-validation.json", validation)
    write(
        out / "completed.json",
        {
            "rounds": 1,
            "games": 256,
            "reason": "time_budget",
            "elapsed_seconds": 9001,
            "best_validation_score": validation["score_rate"],
        },
    )
    final = {"update": 1, "model_state": latest, "config": config}
    for name in ("latest", "final"):
        torch.save(final, out / f"{name}.pt")
    selected = final if improves else parent_checkpoint
    # A reserialized unchanged parent still has ancestral update112, not 112 new rounds.
    torch.save(selected, out / "best.pt")
    (tmp_path / f"frozen-{arm}.pt").write_bytes((out / "best.pt").read_bytes())
    state = {
        "config": config,
        "reference_sha256": report.MODEL_SHA,
        "round": 1,
        "games": 256,
        "replay": [0] * 100000,
        "model_state": latest,
        "optimizer": {"state": {0: {"step": torch.tensor(1)}}},
    }
    torch.save(state, out / "resume.pt")
    summary = report.audit_training(tmp_path, arm, config, parent)
    assert summary["selected_additional_round"] == int(improves)
    assert summary["selected_additional_games"] == 256 * int(improves)
    assert summary["soft_cap_overshoot_seconds"] == 1
    # A reset/lost optimizer after recovery must not pass as a complete cumulative run.
    state["optimizer"]["state"][0]["step"] = torch.tensor(0)
    torch.save(state, out / "resume.pt")
    with pytest.raises(ValueError, match="optimizer"):
        report.audit_training(tmp_path, arm, config, parent)
