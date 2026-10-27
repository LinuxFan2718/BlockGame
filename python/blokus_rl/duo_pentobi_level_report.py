"""Replay and audit a completed fixed-model level 3, 4 or 5 benchmark; no games launched."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from ._native import DuoGame
from .duo_crossplay import opening_book
from .duo_pentobi_ladder import summarize

MODEL_SHA = "95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512"
EVALUATOR_SHA = "80727fc7fee1fcdab75eebc989f41f617e6721d4f546e266367ca53f294a0493"
BINARY_SHA = "ccb2362fc2eb066be9337b76488c527f22aae4ad4b72296ae105e54b2c015882"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, level, apps_path, app_id):
    root = Path(root)
    seed = 2603000 if level == 3 else 2604000
    run_name = f"duo-strong-search2048-level{level}-{seed}"
    assert root.name == run_name
    arm = root / "search-2048"
    top, finished, hardware = [
        read(root / f"{name}.json") for name in ("protocol", "finished", "hardware")
    ]
    app = next(a for a in read(apps_path) if a["app_id"] == app_id)
    assert app["state"] == "stopped" and int(app["tasks"]) == 0
    assert hardware["gpu"] == "NVIDIA L40S"
    assert hardware["cpu_cap"] == 2 and hardware["ram_gib_cap"] == 8
    assert 0 < hardware["seconds"] < 7200
    protocol, completed = read(arm / "protocol.json"), read(arm / "completed.json")
    shared = {
        "first_level": level,
        "last_level": level,
        "opening_pairs": 100,
        "seed": seed,
        "device": "cuda",
        "precision": "bfloat16",
    }
    assert top["shared_config"] == shared and top["budgets"] == [2048]
    assert finished["budgets"] == [2048] and finished["games"] == 200
    assert top["model_sha256"] == finished["model_sha256"] == MODEL_SHA
    assert protocol["config"] == shared | {"simulations": 2048}
    assert protocol["checkpoint_sha256"] == digest(arm / "model.pt") == MODEL_SHA
    assert protocol["openings"] == opening_book(100, seed=seed)
    assert protocol["pentobi_binary_sha256"] == BINARY_SHA
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert completed["protocol_sha256"] == fingerprint
    assert completed["levels"] == [str(level)] and completed["games"] == 200
    entries = Path(f"runs/modal-level{level}-cost-ledger.jsonl").read_text().splitlines()
    assert len(entries) == 1
    ledger = json.loads(entries[0])
    assert top["execution"] == ledger and ledger["study"] == f"level{level}"
    assert ledger["run_name"] == run_name
    runner = Path(f"runs/{run_name}-runner.py")
    assert digest(runner) == ledger["runner_sha256"]
    committed = subprocess.check_output(
        ["git", "show", ledger["source_commit"] + ":cloud/modal_search_sweep.py"]
    )
    assert hashlib.sha256(committed).hexdigest() == ledger["runner_sha256"]
    assert (
        digest("runs/duo-strong-pentobi-steps-2599000-module.py")
        == top["evaluator_sha256"]
        == ledger["evaluator_sha256"]
        == EVALUATOR_SHA
    )
    directory = arm / f"level-{level:02}"
    records = [read(p) for p in sorted(directory.glob("pair-*.json"))]
    assert [r["index"] for r in records] == list(range(100))
    replayed = roots = 0
    for row in records:
        assert row["level"] == level and row["protocol_sha256"] == fingerprint
        assert [g["seat"] for g in row["games"]] == [0, 1]
        for detail in row["games"]:
            index, seat = row["index"], detail["seat"]
            assert detail["seed"] == seed + index + seat * 1000000
            assert detail["actions"][:4] == protocol["openings"][index]
            game, decisions = DuoGame(), 0
            for ply, action in enumerate(detail["actions"]):
                decisions += ply >= 4 and game.current_player == seat
                game.step(action)
            assert game.terminated and list(game.scores) == detail["scores"]
            assert game.scores[seat] - game.scores[1 - seat] == detail["margin"]
            assert detail["policy_metrics"]["simulations"] == 2048 * decisions
            replayed += 1
            roots += decisions
    summary = summarize(records, 100)
    summary["win_rate"] = summary["wins"] / summary["games"]
    assert summary == read(directory / "summary.json")
    assert read(arm / "results.json") == {str(level): summary}
    assert read(root / "results.json") == {"2048": summary}
    assert replayed == 200 and not (root / "failure.json").exists()
    preserved = read("runs/search-preserved-checkpoints.json") | {
        "runs/duo-deep-local-764/conv.pt": "0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86",
        "runs/duo-deep-study-764-download-20261026T0634/duo-deep-study-764/conv/resume.pt": "2df38e9b7585c23fe0b8159cc4511db1811c0b745534d8a9223e885dec1db52d",
        "runs/duo-continuation-final-20261026T2007/duo-continuation-865/frozen-standard256.pt": "4cbeb255cb5688a6b931ac4ac6964f21e3c6430be2e6dd558e6d7c9a14d43b19",
        "runs/duo-continuation-final-20261026T2007/duo-continuation-865/frozen-strong1024.pt": MODEL_SHA,
    }
    for path, expected in preserved.items():
        assert digest(path) == expected
    return {
        "run_name": run_name,
        "level": level,
        "protocol_config": protocol["config"],
        "simulations_per_move": 2048,
        "games_replayed": replayed,
        "searched_decisions": roots,
        "actual_simulations": 2048 * roots,
        "checkpoint_sha256": MODEL_SHA,
        "protocol_sha256": fingerprint,
        "certified_checkpoints_unchanged": True,
        "preserved": preserved,
        "stopped_app": app,
        "hardware": hardware,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--apps", type=Path, required=True)
    parser.add_argument("--app-id", required=True)
    args = parser.parse_args()
    record = audit(args.run, args.level, args.apps, args.app_id)
    (args.run.parent / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    manifest = {
        str(p.relative_to(args.run)): digest(p) for p in sorted(args.run.rglob("*")) if p.is_file()
    }
    (args.run.parent / "download-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Written only after every trace and preservation check above has passed.
    if args.level == 4:
        Path("runs/pentobi-level4-verified.json").write_text(json.dumps(record, indent=2) + "\n")
    compact = record | {
        "summary": {k: v for k, v in record["summary"].items() if not k.startswith("opening_pair")}
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
