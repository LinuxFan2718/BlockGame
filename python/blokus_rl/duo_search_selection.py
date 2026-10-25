"""Bounded follow-up: select the deployed searched agent on validation games.

Only existing frozen warm-start checkpoints are compared. No additional training.
The final test uses a fresh opening seed, after choosing once on validation.
"""

import hashlib
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .duo_crossplay import evaluate_pair, opening_book
from .duo_learning import load_duo
from .duo_mcts import MctsPolicy
from .learning import configure_torch


def trial(task):
    checkpoint, reference, book, seed = task
    configure_torch(seed)
    return evaluate_pair(
        MctsPolicy(load_duo(checkpoint), 128), MctsPolicy(load_duo(reference), 128), book, seed
    )


def main():
    configure_torch(260643)
    source = Path("runs/duo-az-warm-643")
    if not (source / "completed.json").exists():
        raise ValueError("training must finish first")
    out = Path("runs/duo-az-warm-search-selection-2559000")
    out.mkdir(exist_ok=False)
    rounds = list(range(4, 33, 4))
    sources = {"reference": source / "initial.pt"} | {
        str(r): source / f"round-{r:03d}.pt" for r in rounds
    }
    paths = {name: out / (name + ".pt") for name in sources}
    for name, path in sources.items():
        shutil.copy2(path, paths[name])
    validation = opening_book(40, seed=2543000)
    protocol = {
        "hypothesis": "Raw-policy checkpoint selection may discard a stronger learned search evaluator.",
        "trigger": "Original fixed-budget study: warm raw49% vs initial; searched warm61.25% vs equally searched initial.",
        "selection": "Highest score vs searched frozen initial, 128 simulations each, validation seed2543000,40 opening pairs; initial50% retained unless exceeded; strict ties keep earliest round.",
        "source_training_games": 4096,
        "source_training_seed": 643,
        "new_training_games": 0,
        "rounds": rounds,
        "validation_openings": validation,
        "fresh_test_seed": 2559000,
        "fresh_test_opening_pairs": 100,
        "success": "Fresh-test searched-vs-searched paired95% lower bound >50%. Report even if selection fails.",
        "checkpoint_sha256": {
            k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()
        },
        "limitations": "Adaptive follow-up prompted by previous held-out results; fresh test never used for selection. One training seed; not a replication.",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tasks = [(paths[str(r)], paths["reference"], validation, 2543000) for r in rounds]
    selected, best = 0, 0.5
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for round_, result in zip(rounds, pool.map(trial, tasks), strict=True):
            (out / f"validation-{round_:03d}.json").write_text(json.dumps(result, indent=2) + "\n")
            if result["score_rate"] > best:
                selected, best = round_, result["score_rate"]
            print("validation", round_, result["score_rate"], flush=True)
    chosen = paths[str(selected)] if selected else paths["reference"]
    shutil.copy2(chosen, out / "selected.pt")
    selection = {
        "selected_round": selected,
        "validation_score": best,
        "selected_sha256": hashlib.sha256(chosen.read_bytes()).hexdigest(),
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    # Commit the choice to disk before constructing or evaluating the new test book.
    book = opening_book(100, seed=2559000)
    selected_model = load_duo(out / "selected.pt")
    reference_model = load_duo(paths["reference"])
    results = {}
    for name, a, b in [
        ("search_vs_search", MctsPolicy(selected_model, 128), MctsPolicy(reference_model, 128)),
        ("raw_vs_raw", selected_model, reference_model),
    ]:
        start = time.monotonic()
        result = evaluate_pair(a, b, book, 2559000)
        result["seconds"] = time.monotonic() - start
        results[name] = result
        (out / "test-results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            "test",
            name,
            {
                k: result[k]
                for k in [
                    "wins",
                    "draws",
                    "losses",
                    "score_rate",
                    "score_rate_95pct_cluster_bootstrap",
                ]
            },
            flush=True,
        )
    (out / "completed.json").write_text(json.dumps(selection) + "\n")


if __name__ == "__main__":
    main()
