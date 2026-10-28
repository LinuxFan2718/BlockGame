"""Compare the image's original and isolated optimized native extensions."""

import importlib.util
import json
import os
import sys
import sysconfig
import types
from pathlib import Path

import numpy as np
from blokus_rl._native import DuoGame, DuoSearch

parent = types.ModuleType("original_probe")
parent.__path__ = []
sys.modules[parent.__name__] = parent
baseline_path = (
    Path(os.environ["BLOKUS_BASELINE_EXTENSION"])
    if os.environ.get("BLOKUS_BASELINE_EXTENSION")
    else next((Path(sysconfig.get_paths()["purelib"]) / "blokus_rl").glob("_native.*.so"))
)
spec = importlib.util.spec_from_file_location("original_probe._native", baseline_path)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
rng = np.random.default_rng(1069)
positions = roots = simulations = 0
for seed in range(32):
    old, new = baseline.DuoGame(), DuoGame()
    plies = 0
    while not old.terminated:
        assert old.legal_actions() == new.legal_actions()
        assert old.scores == new.scores and old.board == new.board
        if seed < 3 and plies in (0, 8, 16, 24):
            a, b = baseline.DuoSearch(old), DuoSearch(new)
            while a.completed < 256:
                left, right = (
                    a.request_decisions(min(8, 256 - a.completed)),
                    b.request_decisions(min(8, 256 - b.completed)),
                )
                assert len(left) == len(right)
                for (ta, da), (tb, db) in zip(left, right, strict=True):
                    assert ta == tb and all(
                        bytes(x) == bytes(y) for x, y in zip(da, db, strict=True)
                    )
                    count = len(da[1]) // 4
                    priors = rng.random(count).tolist()
                    value = float(rng.uniform(-1, 1))
                    a.expand(ta, priors, value)
                    b.expand(tb, priors, value)
            assert a.stats() == b.stats()
            roots += 1
            simulations += 256
        action = int(rng.choice(old.legal_actions()))
        old.step(action)
        new.step(action)
        positions += 1
        plies += 1
    assert old.terminated == new.terminated and old.scores == new.scores
    assert old.legal_actions() == new.legal_actions() == []
result = {
    "random_games": 32,
    "positions": positions,
    "arbitrary_prior_roots": roots,
    "simulations_per_implementation": simulations,
    "exact": True,
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
