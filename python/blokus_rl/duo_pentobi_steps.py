"""Sequential fixed-policy Pentobi test; stop only after a complete losing level."""

import hashlib
import json
import time
from pathlib import Path

from blokus_rl.duo_crossplay import opening_book
from blokus_rl.duo_deep import atomic_json
from blokus_rl.duo_deep_evaluate import IndexedSearchPolicy
from blokus_rl.duo_pentobi import Pentobi, match
from blokus_rl.duo_pentobi_ladder import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, checkpoint, binary, out, *, commit=None, resume=False, play=None):
    """Use the same book at each level; preserve whole pairs on interruption.

    `play` is an evaluation-only test hook returning one complete opening pair.
    A resume must retain the exact model, binary, config and saved records.
    """
    from blokus_rl.duo_learning import load_duo
    from blokus_rl.learning import configure_torch

    checkpoint, binary, out = map(Path, (checkpoint, binary, out))
    if out.exists() and not resume:
        raise ValueError("existing output; diagnose before explicit exact-config resume")
    if not 1 <= config["first_level"] <= config["last_level"] <= 9:
        raise ValueError("Pentobi levels must be within 1–9")
    if config["opening_pairs"] < 1 or config["simulations"] < 1:
        raise ValueError("positive pair and search budgets required")
    book = opening_book(config["opening_pairs"], seed=config["seed"])
    protocol = {
        "config": config,
        "checkpoint_sha256": digest(checkpoint),
        "pentobi_binary_sha256": digest(binary),
        "pentobi": "31.0; Duo; one thread; no book or resignation; default symmetric draw",
        "openings": book,
        "stop_rule": "after a complete level: (wins + draws/2)/games < 0.5; equality advances",
        "selection": "frozen before this fresh book; no learning or checkpoint selection",
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if out.exists() and json.loads((out / "protocol.json").read_text()) != protocol:
        raise ValueError("model/binary/protocol changed during recovery")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "protocol.json", protocol)
    if not (out / "model.pt").exists():
        (out / "model.pt").write_bytes(checkpoint.read_bytes())
    if digest(out / "model.pt") != protocol["checkpoint_sha256"]:
        raise ValueError("saved frozen model changed")
    if play is None:
        configure_torch(config["seed"])
        model = load_duo(checkpoint).to(config["device"]).eval()
        policy = IndexedSearchPolicy(model, config["simulations"], config["precision"])

        def play(level, index, opening):
            details = []
            with Pentobi(binary=binary, level=level, seed=config["seed"], timeout=600) as engine:
                for seat in (0, 1):
                    before = dict(policy.metrics)
                    started = time.monotonic()
                    detail = match(
                        engine, policy, seat, config["seed"] + index + seat * 1000000, opening
                    )
                    detail |= {
                        "seconds": time.monotonic() - started,
                        "policy_metrics": {
                            k: v - before.get(k, 0) for k, v in policy.metrics.items()
                        },
                    }
                    details.append(detail)
            return {"level": level, "index": index, "games": details}

    results = {}
    if commit:
        commit()
    for level in range(config["first_level"], config["last_level"] + 1):
        directory = out / f"level-{level:02}"
        directory.mkdir(exist_ok=True)
        records = []
        for index, opening in enumerate(book):
            path = directory / f"pair-{index:03}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["protocol_sha256"] != fingerprint:
                    raise ValueError("cached pair protocol changed")
            else:
                row = play(level, index, opening)
                row["protocol_sha256"] = fingerprint
                atomic_json(path, row)
            if (
                row["level"] != level
                or row["index"] != index
                or [g["seat"] for g in row["games"]] != [0, 1]
            ):
                raise ValueError("invalid pair routing or swapped seats")
            if [g["seed"] for g in row["games"]] != [
                config["seed"] + index,
                config["seed"] + index + 1000000,
            ]:
                raise ValueError("cached pair seeds changed")
            if any(g["actions"][: len(opening)] != opening for g in row["games"]):
                raise ValueError("cached opening changed")
            records.append(row)
            if (index + 1) % 10 == 0 or index + 1 == len(book):
                atomic_json(
                    out / "status.json",
                    {"level": level, "pairs": index + 1, "pairs_total": len(book)},
                )
                if commit:
                    commit()
                print(json.dumps({"level": level, "pairs": index + 1}), flush=True)
        result = summarize(records, len(book))
        result["win_rate"] = result["wins"] / result["games"]
        atomic_json(directory / "summary.json", result)
        results[str(level)] = result
        atomic_json(out / "results.json", results)
        if commit:
            commit()
        print(json.dumps({"level": level, "summary": result}), flush=True)
        if result["score_rate"] < 0.5:
            reason = "first_complete_level_below_half_score"
            break
    else:
        reason = "highest_pentobi_level_reached"
    atomic_json(
        out / "completed.json",
        {
            "protocol_sha256": fingerprint,
            "levels": list(results),
            "reason": reason,
            "games": sum(r["games"] for r in results.values()),
        },
    )
    if commit:
        commit()
    return results
