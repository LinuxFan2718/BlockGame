"""Bounded inference buffers and a cache for one frozen model generation."""

from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from .duo_env import PASS, STATE_SIZE


@dataclass
class IndexedBlock:
    """Borrowed views of one outstanding worker request, valid until its reply."""

    observations: np.ndarray
    indices: np.ndarray
    counts: np.ndarray

    def __len__(self):
        return len(self.counts)

    @property
    def width(self):
        return int(self.counts.max())

    def slice(self, start, end):
        return IndexedBlock(
            self.observations[start:end], self.indices[start:end], self.counts[start:end]
        )

    def records(self):
        return [
            {"observation": obs, "indices": ids[: int(n)]}
            for obs, ids, n in zip(self.observations, self.indices, self.counts, strict=True)
        ]


def slice_blocks(blocks, start, end):
    result, offset = [], 0
    for block in blocks:
        lo, hi = max(0, start - offset), min(len(block), end - offset)
        if lo < hi:
            result.append(block.slice(lo, hi))
        offset += len(block)
    if sum(map(len, result)) != end - start:
        raise ValueError("block slice exceeds request")
    return result


def fill_blocks(blocks, observations, indices, mask):
    """Pack whole strided blocks; clear stale padding and keep dummy rows finite."""
    n = sum(map(len, blocks))
    if not blocks or not 0 < n <= len(observations):
        raise ValueError("invalid block batch length")
    width = max(block.width for block in blocks)
    if not 0 < width <= indices.shape[1]:
        raise ValueError("invalid block action width")
    observations.fill(0)
    indices.fill(0)
    mask.fill(False)
    mask[:, 0] = True
    offset = 0
    columns = np.arange(width)[None, :]
    for block in blocks:
        end = offset + len(block)
        if (
            block.observations.shape != (len(block), STATE_SIZE)
            or np.any(block.counts < 1)
            or np.any(block.counts > block.indices.shape[1])
        ):
            raise ValueError("invalid indexed block shape/counts")
        observations[offset:end] = block.observations
        valid = columns < block.counts[:, None]
        mask[offset:end, :width] = valid
        # Stale shared-memory padding is never interpreted as a policy index.
        block_width = block.width
        np.copyto(
            indices[offset:end, :block_width],
            block.indices[:, :block_width],
            casting="unsafe",
            where=valid[:, :block_width],
        )
        offset = end
    return n, width


class InferenceBuffer:
    """One outstanding request per worker; queues carry only sequence metadata.

    The parent owns/unlinks the allocation after workers exit. Responses are
    copied by the worker before it issues the next request, so a later batch
    cannot overwrite priors still being consumed by Rust.
    """

    def __init__(self, rows, *, name=None):
        self.rows = rows
        self.owner = name is None
        specs = {
            "observations": ((rows, STATE_SIZE), np.float32),
            "counts": ((rows,), np.uint16),
            "indices": ((rows, PASS + 1), np.uint16),
            "probabilities": ((rows, PASS + 1), np.float32),
            "values": ((rows,), np.float32),
        }
        offsets, size = {}, 0
        for key, (shape, dtype) in specs.items():
            size = (size + 63) // 64 * 64
            offsets[key] = size
            size += int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.memory = SharedMemory(name=name, create=self.owner, size=size if self.owner else 0)
        self.arrays = {
            key: np.ndarray(shape, dtype=dtype, buffer=self.memory.buf, offset=offsets[key])
            for key, (shape, dtype) in specs.items()
        }

    def descriptor(self):
        return {"rows": self.rows, "name": self.memory.name}

    def write_request(self, records):
        if not 0 < len(records) <= self.rows:
            raise ValueError("inference request exceeds shared buffer capacity")
        for i, row in enumerate(records):
            n = len(row["indices"])
            if not 0 < n <= PASS + 1:
                raise ValueError("invalid legal-action count")
            self.arrays["observations"][i] = row["observation"]
            self.arrays["counts"][i] = n
            self.arrays["indices"][i, :n] = row["indices"]

    def read_request(self, count):
        if not 0 < count <= self.rows:
            raise ValueError("invalid shared request count")
        return [
            {
                "observation": self.arrays["observations"][i],
                "indices": self.arrays["indices"][i, : int(self.arrays["counts"][i])],
            }
            for i in range(count)
        ]

    def read_block(self, count):
        if not 0 < count <= self.rows:
            raise ValueError("invalid shared request count")
        return IndexedBlock(
            self.arrays["observations"][:count],
            self.arrays["indices"][:count],
            self.arrays["counts"][:count],
        )

    def write_response(self, probabilities, values):
        n, width = probabilities.shape
        if not 0 < n <= self.rows or not 0 < width <= PASS + 1 or len(values) != n:
            raise ValueError("invalid shared response shape")
        self.arrays["probabilities"][:n, :width] = probabilities
        self.arrays["values"][:n] = values
        return width

    def read_response(self, count, width):
        if not 0 < count <= self.rows or not 0 < width <= PASS + 1:
            raise ValueError("invalid shared response metadata")
        return (
            self.arrays["probabilities"][:count, :width].copy(),
            self.arrays["values"][:count].copy(),
        )

    def close(self):
        self.arrays.clear()
        self.memory.close()
        if self.owner:
            self.memory.unlink()


class FrozenEvaluationCache:
    """LRU plus within-batch deduplication; create anew after every weight update.

    Keys contain the entire neural observation and ordered legal indices.
    Outputs precede root exploration noise. This caches no visits or search
    statistics, and never equates positions using board occupancy alone.
    """

    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("nonnegative cache capacity required")
        self.capacity = capacity
        self.entries = OrderedDict()

    def evaluate(self, records, evaluator):
        if not self.capacity:
            p, v = evaluator(records)
            return p, v, {"cache_hits": 0, "cache_misses": len(records), "cache_evictions": 0}
        missing, keys, waiting, resolved = [], [], {}, {}
        hits = evictions = 0
        for row in records:
            key = (
                np.asarray(row["observation"], dtype=np.float32).tobytes(),
                np.asarray(row["indices"], dtype=np.uint16).tobytes(),
            )
            keys.append(key)
            if key in self.entries:
                resolved[key] = self.entries[key]
                self.entries.move_to_end(key)
                hits += 1
            elif key in waiting:
                hits += 1
            else:
                waiting[key] = len(missing)
                missing.append(row)
        if missing:
            probabilities, values = evaluator(missing)
            for key, i in waiting.items():
                n = len(missing[i]["indices"])
                result = (probabilities[i, :n].copy(), float(values[i]))
                resolved[key] = self.entries[key] = result
                if len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
                    evictions += 1
        width = max(len(row["indices"]) for row in records)
        probabilities = np.zeros((len(records), width), dtype=np.float32)
        values = np.empty(len(records), dtype=np.float32)
        for i, key in enumerate(keys):
            p, values[i] = resolved[key]
            probabilities[i, : len(p)] = p
        return (
            probabilities,
            values,
            {
                "cache_hits": hits,
                "cache_misses": len(missing),
                "cache_evictions": evictions,
            },
        )
