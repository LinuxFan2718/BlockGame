"""Finish the prespecified local comparisons after downloading a completed cloud study."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .duo_crossplay import opening_book, paired_interval
from .duo_deep import atomic_json
from .duo_deep_evaluate import evaluate_many
from .duo_learning import load_duo
from .learning import configure_torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_pairs(chunks):
    """Keep both seats together when computing the interval across saved chunks."""
    pairs = np.asarray([pair for chunk in chunks for pair in chunk["opening_pair_scores"]])
    games = len(pairs) * 2
    metrics = {}
    for chunk in chunks:
        for key, value in chunk["metrics"].items():
            metrics[key] = metrics.get(key, 0) + value
    return {
        "games": games,
        "opening_pairs": len(pairs),
        "wins": int(np.sum(pairs == 1)),
        "draws": int(np.sum(pairs == 0.5)),
        "losses": int(np.sum(pairs == 0)),
        "score_rate": float(pairs.mean()),
        "score_rate_95pct_cluster_bootstrap": paired_interval(pairs),
        "opening_pair_scores": pairs.tolist(),
        "mean_margin": sum(c["mean_margin"] * c["games"] for c in chunks) / games,
        "learner_simulations": chunks[0]["learner_simulations"],
        "opponent_simulations": chunks[0]["opponent_simulations"],
        "metrics": metrics,
        "seconds": sum(c["seconds"] for c in chunks),
    }


def paired_case(policy, opponent, book, out, settings, *, chunk_pairs=10):
    """Atomic whole-pair progress; refuse to reuse results from different settings."""
    out = Path(out)
    if out.exists():
        result = json.loads(out.read_text())
        if result["settings"] != settings:
            raise ValueError("completed local evaluation settings changed")
        return result
    partial = out.with_suffix(".partial.json")
    chunks = []
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved["settings"] != settings or saved["chunk_pairs"] != chunk_pairs:
            raise ValueError("partial local evaluation settings changed")
        chunks = saved["chunks"]
    offset = sum(c["opening_pairs"] for c in chunks)
    if offset > len(book) or (offset < len(book) and offset % chunk_pairs):
        raise ValueError("partial evaluation has an invalid opening-pair boundary")
    while offset < len(book):
        subset = book[offset : offset + chunk_pairs]
        result = evaluate_many(
            policy,
            opponent,
            subset,
            simulations=settings["simulations"],
            opponent_simulations=settings["simulations"],
            live_games=chunk_pairs * 2,
            seed=settings["seed"] + offset,
            precision="float32",
        )
        chunks.append(result)
        offset += len(subset)
        atomic_json(partial, {"settings": settings, "chunk_pairs": chunk_pairs, "chunks": chunks})
        print(f"{out.stem}: {offset}/{len(book)} opening pairs", flush=True)
    result = merge_pairs(chunks) | {"settings": settings}
    atomic_json(out, result)
    return result


def finish(run, out, warm, *, threads=1, resume=False):
    run, out, warm = Path(run), Path(out), Path(warm)
    if not (run / "finished.json").exists():
        raise ValueError("download a completed cloud study before local evaluation")
    if threads < 1:
        raise ValueError("threads must be positive")
    if out.exists() and not resume:
        raise ValueError("use an unused output directory or explicitly --resume")
    cloud_protocol = json.loads((run / "heldout-protocol.json").read_text())
    sources = {a: run / f"frozen-{a}.pt" for a in ("residual", "conv")}
    for architecture, path in sources.items():
        if digest(path) != cloud_protocol["checkpoint_sha256"][architecture]:
            raise ValueError(f"frozen {architecture} hash mismatch")
    sources |= {f"initial-{a}": run / a / "initial.pt" for a in ("residual", "conv")}
    sources["warm-round8"] = warm
    book = opening_book(cloud_protocol["opening_pairs"], seed=cloud_protocol["seed"])
    protocol = {
        "purpose": "prespecified descriptive local comparisons; no selection or training",
        "cloud_run": str(run.resolve()),
        "seed": cloud_protocol["seed"],
        "openings": book,
        "checkpoint_sha256": {name: digest(path) for name, path in sources.items()},
        "warm_source": str(warm.resolve()),
        "precision": "CPU float32; cloud comparisons use GPU bfloat16 autocast",
        "threads": threads,
        "torch": str(torch.__version__),
        "chunk_pairs": 10,
        "search": "128 simulations/move on both sides; raw cases use zero",
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = out / "protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("local evaluation protocol/checkpoints changed")
        else:
            for name, path in sources.items():
                (out / f"{name}.pt").write_bytes(path.read_bytes())
            atomic_json(protocol_path, protocol)
        for name, expected in protocol["checkpoint_sha256"].items():
            if digest(out / f"{name}.pt") != expected:
                raise ValueError(f"local frozen {name} hash mismatch")
        configure_torch(protocol["seed"])
        torch.set_num_threads(threads)
        models = {name: load_duo(out / f"{name}.pt").eval() for name in sources}
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        cases = []
        for architecture in ("residual", "conv"):
            cases.extend(
                [
                    (f"{architecture}_raw_vs_initial", architecture, f"initial-{architecture}", 0),
                    (f"{architecture}_raw_vs_largest", architecture, "largest", 0),
                    (f"{architecture}_raw_vs_warm_round8", architecture, "warm-round8", 0),
                ]
            )
        cases += [
            (f"{a}_search128_vs_warm_round8", a, "warm-round8", 128) for a in ("residual", "conv")
        ]
        for name, left, right, budget in cases:
            settings = {
                "protocol_sha256": fingerprint,
                "case": name,
                "simulations": budget,
                "seed": protocol["seed"],
            }
            paired_case(
                models[left], models.get(right, right), book, out / f"{name}.json", settings
            )
        atomic_json(
            out / "completed.json", {"protocol_sha256": fingerprint, "cases": [c[0] for c in cases]}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm", type=Path, default=Path("runs/duo-az-warm-643/best.pt"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    finish(args.run, args.out, args.warm, threads=args.threads, resume=args.resume)


if __name__ == "__main__":
    main()
