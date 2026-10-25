"""Frozen opponent snapshots for the PPO self-play experiment."""

from copy import deepcopy

import numpy as np
import torch

from .learning import observation_tensors


class OpponentPool:
    def __init__(self, current, seed, current_fraction=0.8, max_snapshots=32):
        if not 0 <= current_fraction <= 1:
            raise ValueError("current_fraction must be between 0 and 1")
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be positive")
        self.current = current
        self.current_fraction = current_fraction
        self.max_snapshots = max_snapshots
        self.rng = np.random.default_rng(seed)
        self.snapshots = []
        self.selected = current
        self.selected_label = "current"
        self.episode_counts = {"current": 0, "past": 0}
        self.add_snapshot(0)

    def add_snapshot(self, update):
        snapshot = deepcopy(self.current).eval().requires_grad_(False)
        self.snapshots.append((update, snapshot))
        self.snapshots = self.snapshots[-self.max_snapshots :]

    def begin_episode(self):
        if self.rng.random() < self.current_fraction:
            self.selected = self.current
            self.selected_label = "current"
            self.episode_counts["current"] += 1
        else:
            update, self.selected = self.snapshots[int(self.rng.integers(len(self.snapshots)))]
            self.selected_label = f"past-{update}"
            self.episode_counts["past"] += 1

    @torch.no_grad()
    def __call__(self, observation, rng):
        # Called only during frozen rollout collection. The opponent stays fixed
        # for the WHOLE episode, including its opening move when the learner is O.
        distribution, _ = self.selected.distribution(*observation_tensors(observation))
        return int(distribution.sample())
