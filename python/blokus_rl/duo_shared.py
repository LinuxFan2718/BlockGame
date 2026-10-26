"""CPU search workers sharing one batched neural evaluator.

Only the parent owns the neural model/GPU. Every collection round uses frozen
weights. Workers receive probabilities for their own ordered legal move lists;
game rules, search traversal, exploration noise and terminal targets stay in Rust
and the workers. Results are merged in worker order, not arrival order.
"""

import contextlib
import multiprocessing as mp
import queue
import time
import traceback

import numpy as np
import torch

from ._native import DuoGame
from .duo_mcts import search_games
from .duo_residual import candidate_indices_numpy
from .learning import configure_torch


def indexed(records):
    return [
        {
            "observation": row["observation"],
            "indices": row.get("indices")
            if "indices" in row
            else candidate_indices_numpy(row["candidates"]),
        }
        for row in records
    ]


def pack_indexed(records, device):
    count = max(len(row["indices"]) for row in records)
    indices = np.zeros((len(records), count), dtype=np.int64)
    mask = np.zeros((len(records), count), dtype=bool)
    for i, row in enumerate(records):
        n = len(row["indices"])
        indices[i, :n], mask[i, :n] = row["indices"], True
    return (
        torch.from_numpy(np.stack([r["observation"] for r in records])).to(device),
        torch.from_numpy(indices).to(device),
        torch.from_numpy(mask).to(device),
    )


def precision_context(model, precision):
    if precision == "bfloat16" and next(model.parameters()).device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_indexed(model, records, precision="float32"):
    observations, indices, mask = pack_indexed(records, next(model.parameters()).device)
    with precision_context(model, precision):
        scores, values = model(observations)
        logits = scores.gather(-1, indices).float().masked_fill(~mask, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        values = values.flatten().clamp(-1, 1).float().cpu().numpy()
    if not np.isfinite(probabilities).all() or not np.isfinite(values).all():
        raise RuntimeError("nonfinite batched inference")
    return probabilities, values


class RemoteEvaluator:
    def __init__(self, worker, requests, responses):
        self.worker, self.requests, self.responses = worker, requests, responses
        self.sequence = 0

    def __call__(self, records):
        sequence = self.sequence
        self.sequence += 1
        self.requests.put(("evaluate", self.worker, sequence, indexed(records)))
        reply_sequence, probabilities, values = self.responses.get(timeout=180)
        if reply_sequence != sequence:
            raise RuntimeError("inference response ordering mismatch")
        return probabilities, values


def collect(config, seed, count, evaluator):
    records, scores, metrics = [], [], {}
    for offset in range(0, count, config.live_games):
        n = min(config.live_games, count - offset)
        games = [DuoGame() for _ in range(n)]
        rngs = [np.random.default_rng(seed + offset + i) for i in range(n)]
        histories, plies = [[] for _ in games], [0] * n
        active = list(range(n))
        while active:
            groups = {False: [], True: []}
            for i in active:
                groups[rngs[i].random() >= config.fast_probability].append(i)
            for full, ids in groups.items():
                if not ids:
                    continue
                results = search_games(
                    None,
                    [games[i] for i in ids],
                    config.full_simulations if full else config.fast_simulations,
                    config.leaf_batch,
                    config.inference_batch,
                    rngs=[rngs[i] for i in ids] if full else None,
                    metrics=metrics,
                    evaluator=evaluator,
                )
                for i, result in zip(ids, results, strict=True):
                    if full:
                        histories[i].append(
                            {
                                "observation": result.observation,
                                "indices": candidate_indices_numpy(result.candidates).astype(
                                    np.uint16
                                ),
                                "visits": result.visits.astype(np.uint16),
                                "player": games[i].current_player,
                            }
                        )
                    if plies[i] < config.temperature_plies:
                        probabilities = result.policy.astype(np.float64)
                        probabilities /= probabilities.sum()
                        action = int(rngs[i].choice(result.actions, p=probabilities))
                    else:
                        action = result.action()
                    games[i].step(action)
                    plies[i] += 1
                    if games[i].terminated:
                        score = games[i].scores
                        winner = int(np.sign(score[0] - score[1]))
                        for row in histories[i]:
                            row["value_target"] = float(
                                winner if row.pop("player") == 0 else -winner
                            )
                        records.extend(histories[i])
                        scores.append(score)
            active = [i for i in active if not games[i].terminated]
    return records, scores, metrics


def worker_main(worker, config, tasks, requests, responses):
    configure_torch(config.seed + worker)
    evaluator = RemoteEvaluator(worker, requests, responses)
    try:
        while (task := tasks.get()) is not None:
            round_, count = task
            seed = config.seed + round_ * 1000000 + worker * 10000
            result = collect(config, seed, count, evaluator)
            requests.put(("done", worker, result))
    except BaseException:
        requests.put(("error", worker, traceback.format_exc()))


class SharedSearchPool:
    def __init__(self, config):
        self.config = config
        self.context = mp.get_context("spawn")
        self.requests = self.context.Queue(maxsize=config.workers * 2)
        self.tasks = [self.context.Queue(maxsize=1) for _ in range(config.workers)]
        self.responses = [self.context.Queue(maxsize=1) for _ in range(config.workers)]
        self.processes = []

    def __enter__(self):
        for worker in range(self.config.workers):
            process = self.context.Process(
                target=worker_main,
                args=(
                    worker,
                    self.config,
                    self.tasks[worker],
                    self.requests,
                    self.responses[worker],
                ),
            )
            process.start()
            self.processes.append(process)
        return self

    def __exit__(self, exception_type, *args):
        for task in self.tasks:
            try:
                task.put_nowait(None)
            except queue.Full:
                pass
        for process in self.processes:
            if exception_type and process.is_alive():
                process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for channel in [self.requests, *self.tasks, *self.responses]:
            channel.cancel_join_thread()
            channel.close()

    def collect(self, model, round_, count):
        config = self.config
        for worker, task in enumerate(self.tasks):
            n = count // config.workers + int(worker < count % config.workers)
            task.put((round_, n))
        completed, metrics = {}, {"gpu_batches": 0, "gpu_positions": 0, "inference_seconds": 0.0}
        started = time.monotonic()

        def receive(timeout):
            try:
                return self.requests.get(timeout=timeout)
            except queue.Empty:
                if time.monotonic() - started > config.round_timeout_seconds:
                    raise TimeoutError("self-play round exceeded its bounded deadline") from None
                if any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError("search worker exited before completing its round") from None
                return None

        def handle(message):
            if message[0] == "error":
                raise RuntimeError(f"worker {message[1]} failed:\n{message[2]}")
            if message[0] == "done":
                if message[1] in completed:
                    raise RuntimeError("duplicate worker completion")
                completed[message[1]] = message[2]
                return None
            if message[0] != "evaluate":
                raise RuntimeError("unknown worker message")
            return message

        while len(completed) < config.workers:
            if time.monotonic() - started > config.round_timeout_seconds:
                raise TimeoutError("self-play round exceeded its bounded deadline")
            message = receive(1.0)
            if message is None:
                continue
            request = handle(message)
            if request is None:
                continue
            batch_requests = [request]
            size = len(request[3])
            deadline = time.monotonic() + config.batch_wait_ms / 1000
            while size < config.inference_batch and time.monotonic() < deadline:
                message = receive(max(0.00001, deadline - time.monotonic()))
                if message is None:
                    break
                request = handle(message)
                if request is not None:
                    batch_requests.append(request)
                    size += len(request[3])
            rows = [row for request in batch_requests for row in request[3]]
            probabilities, values = [], []
            start = time.monotonic()
            for offset in range(0, len(rows), config.inference_batch):
                subset = rows[offset : offset + config.inference_batch]
                p, v = evaluate_indexed(model, subset, config.precision)
                probabilities.extend(
                    p[i, : len(row["indices"])].copy() for i, row in enumerate(subset)
                )
                values.extend(v.tolist())
                metrics["gpu_batches"] += 1
                metrics["gpu_positions"] += len(subset)
            metrics["inference_seconds"] += time.monotonic() - start
            offset = 0
            for _, worker, sequence, request_rows in batch_requests:
                end = offset + len(request_rows)
                width = max(len(p) for p in probabilities[offset:end])
                padded = np.zeros((end - offset, width), dtype=np.float32)
                for i, p in enumerate(probabilities[offset:end]):
                    padded[i, : len(p)] = p
                self.responses[worker].put(
                    (sequence, padded, np.asarray(values[offset:end], dtype=np.float32))
                )
                offset = end
        records, scores = [], []
        for worker in range(config.workers):
            rows, results, counters = completed[worker]
            records.extend(rows)
            scores.extend(results)
            for key, value in counters.items():
                metrics[key] = metrics.get(key, 0) + value
        if len(scores) != count:
            raise RuntimeError("collection game accounting mismatch")
        metrics["collect_seconds"] = time.monotonic() - started
        metrics["mean_gpu_batch"] = metrics["gpu_positions"] / max(metrics["gpu_batches"], 1)
        return records, scores, metrics
