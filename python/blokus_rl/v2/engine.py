"""Drive the Rust engine with batched network inference.

The engine splits its games into two groups. While the accelerator evaluates
one group's batch, Rust searches the other group on all CPU cores:

    submit(prepare(B))   # CPU: search group B; GPU: still running group A
    apply(A, collect(A)) # wait for A, back up its results

Network calls never sit inside the per-position search loop.
"""

import contextlib
import threading
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
        self.channels_last = self.device.type == "cuda"
        if self.device.type == "cuda":
            # Batches are padded to a few fixed sizes, so autotuning pays off.
            torch.backends.cudnn.benchmark = True
        for model in self.models.values():
            model.eval()
            if self.channels_last:
                # NHWC lets cuDNN use tensor cores for bf16 convolutions.
                model.to(memory_format=torch.channels_last)

    def autocast(self):
        if self.precision == "bfloat16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _to(self, array):
        # Plain copies: allocating pinned staging buffers every batch cost
        # more than the transfers themselves.
        return torch.from_numpy(array).to(self.device)

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
        # Policy indices fit in int16 (max 17,836); widen on the device.
        legal = self._to(np.frombuffer(legal, np.int16)).long()
        width = int(counts.max())
        if self.device.type == "cuda":
            lengths = self._to(counts.astype(np.int64))
            rows = torch.repeat_interleave(torch.arange(n, device=self.device), lengths)
            starts = torch.cumsum(lengths, 0) - lengths
            cols = torch.arange(len(legal), device=self.device) - starts[rows]
        else:
            # repeat_interleave is slow on Metal; build the indices on the CPU.
            counts64 = counts.astype(np.int64)
            rows_np = np.repeat(np.arange(n), counts64)
            starts = np.concatenate([[0], np.cumsum(counts64)[:-1]])
            rows = self._to(rows_np)
            cols = self._to(np.arange(len(rows_np)) - np.repeat(starts, counts64))
        nets = np.frombuffer(nets, np.uint8)
        x = planes.float()
        if self.device.type == "cuda" and len(self.models) == 1:
            bucket = -(-n // 256) * 256
            if bucket != n:
                x = torch.cat([x, x.new_zeros(bucket - n, *x.shape[1:])])
                scalars = torch.cat([scalars, scalars.new_zeros(bucket - n, scalars.shape[1])])
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
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
                    logits, values = policy[:n], value[:n]  # gather first, widen after
                else:
                    logits[index], values[index] = policy.float(), value
        # Softmax over each position's legal moves via a padded (n, width)
        # layout; segment reductions (index_add_) are very slow on Metal.
        padded = torch.full((n, width), -torch.inf, device=self.device)
        padded[rows, cols] = logits[rows, legal].float()
        probabilities = torch.softmax(padded, 1)[rows, cols]
        self.positions += n
        self.batches += 1
        self.seconds += time.perf_counter() - started
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        return probabilities, values, event

    def collect(self, handle):
        probabilities, values, event = handle
        if event is not None:
            # Poll instead of blocking so other threads keep the GIL meanwhile.
            while not event.query():
                time.sleep(0.0002)
        return (
            probabilities.float().cpu().numpy(),
            values.float().cpu().numpy().astype(np.float32),
        )


def run(engine, evaluator, *, on_finished=None, external=None, replay=None, poll=None, timings=None):
    """Play until the engine has finished all games.

    on_finished(list_of_game_dicts) is called as games end; external(engine)
    supplies moves for external seats; poll() runs once per iteration and may
    return True to stop early. `timings`, if a dict, accumulates seconds spent
    in each phase of the loop (overlapping GPU work is not counted twice).
    """
    clock = time.perf_counter
    spent = timings if timings is not None else {}
    for key in ("prepare", "submit", "collect", "apply", "callbacks"):
        spent.setdefault(key, 0.0)
    t = clock()
    handles = {0: evaluator.submit(engine.prepare(0)), 1: None}
    spent["submit"] += clock() - t
    current = 0
    while True:
        following = 1 - current
        t0 = clock()
        batch = engine.prepare(following)
        t1 = clock()
        handles[following] = evaluator.submit(batch)
        t2 = clock()
        spent["prepare"] += t1 - t0
        spent["submit"] += t2 - t1
        if handles[current] is not None:
            result = evaluator.collect(handles[current])
            t3 = clock()
            engine.apply(current, *result)
            spent["collect"] += t3 - t2
            spent["apply"] += clock() - t3
            handles[current] = None
        t4 = clock()
        finished = engine.drain(replay)
        if finished and on_finished:
            on_finished(finished)
        if external is not None:
            external(engine)
        stop = poll is not None and poll()
        spent["callbacks"] += clock() - t4
        if stop:
            return  # Early stop requested; unfinished games are discarded.
        current = following
        if engine.done() and handles[0] is None and handles[1] is None:
            break
    finished = engine.drain(replay)
    if finished and on_finished:
        on_finished(finished)


def run_threaded(engine, evaluator, *, on_finished=None, replay=None, poll=None, timings=None):
    """Like `run`, but each game group gets its own thread (prepare → evaluate →
    apply), so one group's Rust work overlaps the other's GPU submission. The
    main thread drains finished games and runs `poll` (e.g. training steps).
    No external seats."""
    stop = threading.Event()
    errors = []
    spent = timings if timings is not None else {}
    for key in ("prepare", "submit", "collect", "apply", "callbacks"):
        spent.setdefault(key, 0.0)
    lock = threading.Lock()

    def worker(group):
        clock = time.perf_counter
        local = dict.fromkeys(("prepare", "submit", "collect", "apply"), 0.0)
        try:
            while not stop.is_set():
                t0 = clock()
                batch = engine.prepare(group)
                t1 = clock()
                handle = evaluator.submit(batch)
                t2 = clock()
                local["prepare"] += t1 - t0
                local["submit"] += t2 - t1
                if handle is None:
                    if engine.done():
                        return
                    time.sleep(0.001)
                    continue
                result = evaluator.collect(handle)
                t3 = clock()
                engine.apply(group, *result)
                local["collect"] += t3 - t2
                local["apply"] += clock() - t3
        except BaseException as error:  # surface worker failures in the main thread
            errors.append(error)
            stop.set()
        finally:
            with lock:
                for key, value in local.items():
                    spent[key] += value

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in range(2)]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            t = time.perf_counter()
            finished = engine.drain(replay)
            if finished and on_finished:
                on_finished(finished)
            if poll is not None and poll():
                stop.set()
            spent["callbacks"] += time.perf_counter() - t
            time.sleep(0.01)
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    if errors:
        raise errors[0]
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
