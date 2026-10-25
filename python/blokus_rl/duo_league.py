"""A matched experiment extending the existing PPO loop to frozen neural rivals."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .duo_crossplay import evaluate_pair, opening_book
from .duo_learning import (
    DuoConfig,
    collect_duo,
    evaluate_duo,
    load_duo,
    make_duo_loss,
    optimize_duo,
    save_duo,
)
from .learning import configure_torch


def train_frozen_league(
    parent,
    training_paths,
    validation_path,
    held_out_path,
    out,
    arm="mixed",
    seed=442,
    updates=300,
    episodes=32,
    validation_seed=1890000,
    test_seed=2290000,
    validation_games=30,
    validation_openings=20,
    test_games=200,
    test_openings=100,
    eval_every=25,
):
    if arm not in ("control", "mixed"):
        raise ValueError("arm must be control or mixed")
    if not training_paths:
        raise ValueError("a frozen training pool is required")
    all_paths = [
        Path(parent),
        *map(Path, training_paths),
        Path(validation_path),
        Path(held_out_path),
    ]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_paths}
    held_out_hash = hashes[str(Path(held_out_path).resolve())]
    training_hashes = {hashes[str(Path(p).resolve())] for p in training_paths}
    if (
        held_out_hash in training_hashes
        or held_out_hash == hashes[str(Path(validation_path).resolve())]
    ):
        raise ValueError("held-out model must not be used in training or validation")
    if test_seed == validation_seed:
        raise ValueError("test and validation opening seeds must differ")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    configure_torch(seed)
    model = load_duo(parent).train()
    # Only training and validation models are loaded before selection. Held-out
    # model bytes are hashed for the leakage guard, then loaded after training.
    opponents = [load_duo(p).requires_grad_(False) for p in training_paths]
    validation = load_duo(validation_path).requires_grad_(False)
    config = DuoConfig(
        seed=seed,
        updates=updates,
        episodes_per_update=episodes,
        opponent="frontier",
        reward="margin",
        selection="frontier",
        validation_seed=validation_seed,
        test_seed=test_seed,
        hidden_size=model.hidden_size,
        architecture=getattr(model, "architecture", "additive"),
    )
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_module = make_duo_loss(model, config)
    parent_meta = {
        "path": str(Path(parent).resolve()),
        "sha256": hashes[str(Path(parent).resolve())],
    }
    protocol = {
        "arm": arm,
        "ppo_config": asdict(config),
        "checkpoint_sha256": hashes,
        "training_opponents": [str(Path(p).resolve()) for p in training_paths],
        "validation_opponent": str(Path(validation_path).resolve()),
        "held_out_opponent": str(Path(held_out_path).resolve()),
        "mixture": {"random": 0.1, "largest": 0.2, "frontier": 0.2, "frozen_neural": 0.5}
        if arm == "mixed"
        else {"frontier": 1.0},
        "neural_training_mode": "greedy",
        "validation_games_per_seat": validation_games,
        "validation_opening_pairs": validation_openings,
        "test_games_per_seat": test_games,
        "test_opening_pairs": test_openings,
        "eval_every": eval_every,
        "selection": "mean largest/frontier/known-neural score rates; tie known-neural",
        "opening_plies": 4,
        "parent": parent_meta,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("duo_learning.py"),
                Path(__file__).with_name("duo_crossplay.py"),
            ]
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    book = opening_book(validation_openings, seed=validation_seed)
    (out / "validation-openings.json").write_text(json.dumps(book) + "\n")

    def checkpoint(name, update):
        save_duo(
            out / name,
            model,
            config,
            update,
            parent=parent_meta,
            metadata={
                "experiment": "frozen_neural_curriculum",
                "arm": arm,
                "training_opponents": protocol["training_opponents"],
                "mixture": protocol["mixture"],
                "neural_training_mode": "greedy",
                "selection": protocol["selection"],
            },
        )

    checkpoint("initial.pt", 0)

    def selector(generator):
        draw = generator.random()
        if draw < 0.1:
            return "random"
        if draw < 0.3:
            return "largest"
        if draw < 0.5:
            return "frontier"
        return opponents[int(generator.integers(len(opponents)))]

    start = time.monotonic()
    best_score, best_update, rows = (-1, -1), 0, []
    for update in range(updates + 1):
        row = {"update": update, "games": update * episodes}
        if update:
            records, reward = collect_duo(
                model,
                config,
                rng,
                [],
                opponent_selector=selector if arm == "mixed" else None,
                sample_opponent=False,
            )
            row.update(optimize_duo(model, loss_module, optimizer, records, config))
            row.update(training_return=reward, transitions=len(records))
        if update % eval_every == 0 or update == updates:
            validation_results = {
                baseline: evaluate_duo(model, baseline, validation_games, validation_seed)
                for baseline in ("random", "largest", "frontier")
            }
            neural = evaluate_pair(model, validation, book, validation_seed, bootstrap=False)
            validation_results["known_neural"] = neural
            for key, result in validation_results.items():
                row[f"{key}_score_rate"] = result["score_rate"]
            score = (
                (row["largest_score_rate"] + row["frontier_score_rate"] + neural["score_rate"]) / 3,
                neural["score_rate"],
            )
            if score > best_score:
                best_score, best_update = score, update
                checkpoint("best.pt", update)
                (out / "best-validation.json").write_text(
                    json.dumps(validation_results, indent=2) + "\n"
                )
            print(
                f"{arm} update {update} games {row['games']} "
                f"largest {row['largest_score_rate']:.1%} frontier {row['frontier_score_rate']:.1%} "
                f"known neural {neural['score_rate']:.1%}",
                flush=True,
            )
        row["elapsed_seconds"] = time.monotonic() - start
        rows.append(row)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        checkpoint("latest.pt", update)
    checkpoint("final.pt", updates)
    selected, initial = load_duo(out / "best.pt"), load_duo(out / "initial.pt")
    # Selection is complete before loading/evaluating this reference opponent.
    held_out = load_duo(held_out_path).requires_grad_(False)
    test_book = opening_book(test_openings, seed=test_seed)
    (out / "test-openings.json").write_text(json.dumps(test_book) + "\n")
    tests = {}
    for label, policy in (("initial", initial), ("selected", selected)):
        tests[label] = {
            baseline: evaluate_duo(policy, baseline, test_games, test_seed)
            for baseline in ("random", "largest", "frontier")
        }
        tests[label]["known_neural"] = evaluate_pair(policy, validation, test_book, test_seed)
        tests[label]["held_out_neural"] = evaluate_pair(policy, held_out, test_book, test_seed)
        # Additional pool members are evaluated explicitly rather than hidden
        # inside a pooled average that can obscure a single exploitable rival.
        tests[label]["training_pool"] = [
            evaluate_pair(policy, rival, test_book, test_seed) for rival in opponents
        ]
    summary = {
        "arm": arm,
        "best_update": best_update,
        "training_games": updates * episodes,
        "held_out": tests,
        "elapsed_seconds": time.monotonic() - start,
        "test_opening_sha256": hashlib.sha256(json.dumps(test_book).encode()).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Finished {arm}, selected {best_update}: {out / 'summary.json'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--train-opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-opponent", type=Path, required=True)
    parser.add_argument("--held-out-opponent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "mixed"), required=True)
    parser.add_argument("--seed", type=int, default=442)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    train_frozen_league(
        args.parent,
        args.train_opponents,
        args.validation_opponent,
        args.held_out_opponent,
        args.out,
        args.arm,
        args.seed,
        args.updates,
        args.episodes,
    )


if __name__ == "__main__":
    main()
