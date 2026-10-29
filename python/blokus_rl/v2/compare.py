"""Paired comparison of two ladder runs that used the same openings (same seed).

For each opening pair, take each run's mean score over its two seat-swapped
games; report the mean difference with a bootstrap interval over openings.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def pair_scores(path):
    games = json.loads(Path(path).read_text())
    pairs = {}
    for g in games:
        seat = g["seats"].index(0)
        margin = g["scores"][seat] - g["scores"][1 - seat]
        pairs.setdefault(g["index"] // 2, []).append(1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)
    return {k: float(np.mean(v)) for k, v in pairs.items() if len(v) == 2}


def compare(a, b, seed=17, resamples=10000):
    sa, sb = pair_scores(a), pair_scores(b)
    keys = sorted(set(sa) & set(sb))
    diff = np.array([sa[k] - sb[k] for k in keys])
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(len(diff), size=(resamples, len(diff)))].mean(1)
    return {
        "pairs": len(keys),
        "a": round(float(np.mean([sa[k] for k in keys])), 4),
        "b": round(float(np.mean([sb[k] for k in keys])), 4),
        "difference": round(float(diff.mean()), 4),
        "interval": [round(float(x), 4) for x in np.quantile(boot, [0.025, 0.975])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="level-N-games.json of run A")
    parser.add_argument("b", type=Path, help="level-N-games.json of run B")
    args = parser.parse_args()
    print(json.dumps(compare(args.a, args.b)))


if __name__ == "__main__":
    main()
