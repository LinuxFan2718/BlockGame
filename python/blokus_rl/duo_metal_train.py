"""Continue a saved Duo CNN locally on Metal, with exact-config round recovery."""

import argparse
from pathlib import Path

from .duo_continuation import initialize_branch
from .duo_deep import DeepConfig, train

ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = ROOT / "runs/duo-continuation-final-20261026T2007/duo-continuation-865"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=PARENT_ROOT / "frozen-strong1024.pt")
    parser.add_argument("--source-state", type=Path, default=PARENT_ROOT / "strong1024/resume.pt")
    parser.add_argument("--resume", action="store_true")
    defaults = dict(seed=1090, rounds=64, games_per_round=256, workers=4, live_games=32,
                    full_simulations=1024, fast_simulations=32, inference_batch=256,
                    leaf_batch=8, minibatch_size=256, replay_positions=100000,
                    eval_every=16, validation_openings=32, validation_simulations=512,
                    validation_seed=2610300, max_seconds=6000)
    for name, value in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    config = DeepConfig(
        **{name: getattr(args, name) for name in defaults},
        architecture="conv", device="mps", precision="float32", batch_wait_ms=0,
    )
    if not args.resume:
        initialize_branch(config, args.out, args.parent, args.source_state, selected_best=True)
    train(config, args.out, args.parent, resume=True)


if __name__ == "__main__":
    main()
