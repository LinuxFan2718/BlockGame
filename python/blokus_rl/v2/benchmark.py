"""Measure self-play throughput of the Rust engine (with or without a network)."""

import argparse
import json
import time

import numpy as np
import torch

from . import engine as v2
from .model import DuoNetV2


class UniformEvaluator:
    """Free stand-in network: uniform priors, zero values. Isolates CPU cost."""

    def __init__(self):
        self.positions = self.batches = 0
        self.sizes = []

    def submit(self, batch):
        counts = np.frombuffer(batch[3], np.uint32)
        if len(counts) == 0:
            return None
        self.positions += len(counts)
        self.batches += 1
        self.sizes.append(len(counts))
        return (
            np.repeat(1.0 / counts, counts).astype(np.float32),
            np.zeros(len(counts), np.float32),
        )

    def collect(self, handle):
        return handle


def measure(evaluator, games, slots, agent, seed=1):
    engine = v2.make_engine([agent, agent], slots, games, seed=seed)
    sizes = []
    original = evaluator.submit

    def submit(batch):
        n = len(np.frombuffer(batch[3], np.uint32))
        if n:
            sizes.append(n)
        return original(batch)

    evaluator.submit = submit
    started = time.perf_counter()
    records = []
    timings = {}
    runner = v2.run_threaded if getattr(evaluator, "device", None) is not None and evaluator.device.type == "cuda" else v2.run
    runner(engine, evaluator, on_finished=records.extend, timings=timings)
    seconds = time.perf_counter() - started
    evaluator.submit = original
    counters = engine.counters()
    sizes = np.array(sizes)
    return {
        "games": len(records),
        "seconds": round(seconds, 2),
        "games_per_second": round(len(records) / seconds, 3),
        "evaluations_per_second": round(counters["evaluations"] / seconds),
        "simulations_per_second": round(counters["simulations"] / seconds),
        "mean_batch": round(float(sizes.mean()), 1),
        "median_batch": float(np.median(sizes)),
        "recorded_positions": int(sum(r["records"] for r in records)),
        "timings": {k: round(v, 2) for k, v in timings.items()},
        **counters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--slots", type=int, default=256)
    parser.add_argument("--network", choices=("uniform", "v2"), default="uniform")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--device", default=v2.default_device())
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--fast", type=int, default=128)
    parser.add_argument("--full", type=int, default=1024)
    parser.add_argument("--full-probability", type=float, default=0.25)
    parser.add_argument("--leaf-batch", type=int, default=4)
    parser.add_argument("--solver-max-moves", type=int, default=0)
    parser.add_argument("--solver-root-nodes", type=int, default=0)
    parser.add_argument("--solver-leaf-nodes", type=int, default=0)
    args = parser.parse_args()
    agent = v2.self_play_agent(
        fast_simulations=args.fast,
        full_simulations=args.full,
        full_probability=args.full_probability,
        leaf_batch=args.leaf_batch,
        solver_max_moves=args.solver_max_moves,
        solver_root_nodes=args.solver_root_nodes,
        solver_leaf_nodes=args.solver_leaf_nodes,
    )
    if args.network == "uniform":
        evaluator = UniformEvaluator()
    else:
        model = DuoNetV2(args.blocks, args.channels).to(args.device).eval()
        evaluator = v2.Evaluator(model, args.device, precision=args.precision)
        torch.set_grad_enabled(False)
    result = measure(evaluator, args.games, args.slots, agent)
    result.update(vars(args), torch_threads=torch.get_num_threads())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
