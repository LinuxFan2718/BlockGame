"""Drive the Rust engine with batched network inference.

The engine splits its games into two groups. While the accelerator evaluates
one group's batch, Rust searches the other group on all CPU cores:

    submit(prepare(B))   # CPU: search group B; GPU: still running group A
    apply(A, collect(A)) # wait for A, back up its results

Network calls never sit inside the per-position search loop.
"""

import contextlib
import time

import numpy as np
import torch

from .._native import DuoEngine
from .model import MARGIN_SCALE, PLANES, SIZE


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Evaluator:
    """Batched inference for one or more networks (keyed by engine net id)."""

    def __init__(self, models, device, *, precision="float32", score_weight=0.0, score_scale=20.0):
        self.models = models if isinstance(models, dict) else {0: models}
        self.device = torch.device(device)
        self.precision = precision
        self.score_weight, self.score_scale = score_weight, score_scale
        self.positions = 0
        self.batches = 0
        self.seconds = 0.0
        for model in self.models.values():
            model.eval()

    def autocast(self):
        if self.precision == "bfloat16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _to(self, array):
        tensor = torch.from_numpy(array)
        if self.device.type == "cuda":
            return tensor.pin_memory().to(self.device, non_blocking=True)
        return tensor.to(self.device)

    @torch.no_grad()
    def submit(self, batch):
        """Launch evaluation of an engine batch; returns a handle or None if empty."""
        planes, scalars, legal, counts, nets = batch
        counts = np.frombuffer(counts, np.uint32)
        n = len(counts)
        if n == 0:
            return None
        started = time.perf_counter()
        planes = self._to(np.frombuffer(planes, np.uint8).reshape(n, PLANES, SIZE, SIZE))
        scalars = self._to(np.frombuffer(scalars, np.float32).reshape(n, -1))
        counts64 = counts.astype(np.int64)
        rows_np = np.repeat(np.arange(n), counts64)
        starts = np.concatenate([[0], np.cumsum(counts64)[:-1]])
        cols_np = np.arange(len(rows_np)) - np.repeat(starts, counts64)
        width = int(counts.max())
        legal = self._to(np.frombuffer(legal, np.uint16).astype(np.int64))
        rows, cols = self._to(rows_np), self._to(cols_np)
        nets = np.frombuffer(nets, np.uint8)
        x = planes.float()
        if len(self.models) == 1:
            groups = [(next(iter(self.models.values())), None)]
        else:
            groups = [
                (model, self._to(np.flatnonzero(nets == key)))
                for key, model in self.models.items()
                if (nets == key).any()
            ]
        if len(groups) > 1:
            logits = torch.empty(n, 17837, device=self.device)
            values = torch.empty(n, device=self.device)
        with self.autocast():
            for model, index in groups:
                xs, ss = (x, scalars) if index is None else (x[index], scalars[index])
                policy, outcome, margin, _ = model(xs, ss)
                wdl = torch.softmax(outcome.float(), 1)
                value = wdl[:, 0] - wdl[:, 2]
                if self.score_weight:
                    value = value + self.score_weight * torch.tanh(
                        margin.float() * MARGIN_SCALE / self.score_scale
                    )
                if index is None:
                    logits, values = policy.float(), value
                else:
                    logits[index], values[index] = policy.float(), value
        # Softmax over each position's legal moves via a padded (n, width)
        # layout; segment reductions (index_add_) are very slow on Metal.
        padded = torch.full((n, width), -torch.inf, device=self.device)
        padded[rows, cols] = logits[rows, legal]
        probabilities = torch.softmax(padded, 1)[rows, cols]
        self.positions += n
        self.batches += 1
        self.seconds += time.perf_counter() - started
        return probabilities, values

    def collect(self, handle):
        probabilities, values = handle
        return (
            probabilities.float().cpu().numpy(),
            values.float().cpu().numpy().astype(np.float32),
        )


def run(engine, evaluator, *, on_finished=None, external=None, replay=None, poll=None):
    """Play until the engine has finished all games.

    on_finished(list_of_game_dicts) is called as games end; external(engine)
    supplies moves for external seats; poll() runs once per iteration and may
    return True to stop early.
    """
    handles = {0: evaluator.submit(engine.prepare(0)), 1: None}
    current = 0
    while True:
        following = 1 - current
        handles[following] = evaluator.submit(engine.prepare(following))
        if handles[current] is not None:
            engine.apply(current, *evaluator.collect(handles[current]))
            handles[current] = None
        finished = engine.drain(replay)
        if finished and on_finished:
            on_finished(finished)
        if external is not None:
            external(engine)
        if poll is not None and poll():
            return  # Early stop requested; unfinished games are discarded.
        current = following
        if engine.done() and handles[0] is None and handles[1] is None:
            break
    finished = engine.drain(replay)
    if finished and on_finished:
        on_finished(finished)


def self_play_agent(**overrides):
    agent = dict(
        full_simulations=1024,
        fast_simulations=128,
        full_probability=0.25,
        noise_weight=0.25,
        noise_mass=10.0,
        temperature_plies=12,
        record=True,
        reuse_tree=True,
        leaf_batch=4,
        c_puct=1.5,
        fpu_reduction=0.25,
        fpu_root=0.0,
    )
    agent.update(overrides)
    return agent


def play_agent(simulations, **overrides):
    """Deterministic evaluation settings: full search every move, no noise."""
    agent = self_play_agent(
        full_simulations=simulations,
        full_probability=1.0,
        noise_weight=0.0,
        temperature_plies=0,
        record=False,
    )
    agent.update(overrides)
    return agent


def make_engine(agents, slots, games, **kwargs):
    return DuoEngine(agents, slots, games, **kwargs)
