"""Lossless packed replay checkpoints; legacy list checkpoints remain readable."""

import numpy as np
import torch

from .duo_env import PASS, STATE_SIZE

FORMAT = "indexed-replay-v1"
BINARY_FEATURES = 631


def replay_length(saved):
    if isinstance(saved, dict):
        if saved.get("format") != FORMAT:
            raise ValueError("unknown replay format")
        return len(saved["value_targets"])
    return len(saved)


def pack_replay(records):
    """Bitpack binary planes/inventories; retain score features and targets exactly.

    Flat tensors replace hundreds of thousands of separately pickled NumPy
    objects. This changes storage, not sample order, values or RNG state.
    """
    records = list(records)
    if not records:
        observations = np.empty((0, STATE_SIZE), dtype=np.float32)
    else:
        observations = np.stack([r["observation"] for r in records])
    if observations.dtype != np.float32 or observations.shape != (len(records), STATE_SIZE):
        raise ValueError("replay requires canonical float32 Duo observations")
    binary = observations[:, :BINARY_FEATURES]
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError("nonbinary feature in binary replay region")
    counts = np.fromiter((len(r["indices"]) for r in records), dtype=np.int64)
    if np.any(counts <= 0):
        raise ValueError("empty replay legal-action list")
    indices = np.concatenate([r["indices"] for r in records]) if records else np.empty(0, np.uint16)
    visits = np.concatenate([r["visits"] for r in records]) if records else np.empty(0, np.uint16)
    if len(indices) != len(visits) or any(len(r["indices"]) != len(r["visits"]) for r in records):
        raise ValueError("replay indices and visits disagree")
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or not np.issubdtype(visits.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices > PASS)
        or np.any(visits < 0)
        or np.any(visits > 65535)
    ):
        raise ValueError("replay integer outside lossless uint16 range")
    values = np.fromiter((r["value_target"] for r in records), dtype=np.float64)
    if not np.isfinite(observations).all() or not np.isfinite(values).all():
        raise ValueError("nonfinite replay")
    return {
        "format": FORMAT,
        "observation_bits": torch.from_numpy(
            np.packbits(binary.astype(np.uint8), axis=1, bitorder="little")
        ),
        "observation_tail": torch.from_numpy(observations[:, BINARY_FEATURES:].copy()),
        "offsets": torch.from_numpy(np.concatenate(([0], np.cumsum(counts)))),
        "indices": torch.from_numpy(indices.astype(np.uint16, copy=False)),
        "visits": torch.from_numpy(visits.astype(np.uint16, copy=False)),
        "value_targets": torch.from_numpy(values),
    }


def replay_records(saved):
    if not isinstance(saved, dict):
        return saved  # Original atomic checkpoints store ordered lists of records.
    n = replay_length(saved)
    bits, tail, offsets, indices, visits, values = (
        saved[key].cpu().numpy()
        for key in (
            "observation_bits",
            "observation_tail",
            "offsets",
            "indices",
            "visits",
            "value_targets",
        )
    )
    if (
        bits.shape != (n, (BINARY_FEATURES + 7) // 8)
        or bits.dtype != np.uint8
        or tail.shape != (n, STATE_SIZE - BINARY_FEATURES)
        or tail.dtype != np.float32
        or offsets.shape != (n + 1,)
        or offsets.dtype != np.int64
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or offsets[-1] != len(indices)
        or indices.dtype != np.uint16
        or visits.dtype != np.uint16
        or len(visits) != len(indices)
        or values.shape != (n,)
        or not np.isfinite(tail).all()
        or not np.isfinite(values).all()
        or np.any(indices > PASS)
    ):
        raise ValueError("invalid packed replay")
    observations = np.empty((n, STATE_SIZE), dtype=np.float32)
    observations[:, :BINARY_FEATURES] = np.unpackbits(
        bits, axis=1, count=BINARY_FEATURES, bitorder="little"
    )
    observations[:, BINARY_FEATURES:] = tail
    return [
        {
            "observation": observations[i],
            "indices": indices[offsets[i] : offsets[i + 1]],
            "visits": visits[offsets[i] : offsets[i + 1]],
            "value_target": float(values[i]),
        }
        for i in range(n)
    ]
