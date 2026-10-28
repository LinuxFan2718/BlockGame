"""Explicit training-device validation and accelerator checkpoint state."""

import torch


def validate_training_precision(device, precision):
    if device not in ("cpu", "cuda", "mps") or precision not in ("float32", "bfloat16"):
        raise ValueError("unsupported training device/precision")
    if device == "mps" and precision != "float32":
        raise ValueError("Metal training currently requires precision=float32")


def prepare_training_device(device, precision, *backends):
    validate_training_precision(device, precision)
    if any(backend not in ("eager", "cuda_graphs") for backend in backends):
        raise ValueError("unknown training inference backend")
    if device != "cuda" and "cuda_graphs" in backends:
        raise ValueError("CUDA Graphs require a CUDA device; use eager on Metal/CPU")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal requested but unavailable; run outside the sandbox on Apple silicon")
    # MPS gather backward uses scatter_reduce, which has no strict deterministic
    # kernel. Keep warnings enabled and record this mode in Metal run metadata.
    torch.use_deterministic_algorithms(True, warn_only=device == "mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_float32_matmul_precision("high")


def metal_rng_state(device):
    if device != "mps":
        return None
    torch.mps.synchronize()
    return torch.mps.get_rng_state().clone()


def restore_metal_rng(device, saved):
    if device == "mps":
        state = saved.get("mps_rng")
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("Metal recovery requires a saved mps_rng state")
        torch.mps.set_rng_state(state.cpu())


def synchronize_device(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
