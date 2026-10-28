"""Device selection for local inference; deliberately separate from training."""

import torch


def inference_device(requested="auto"):
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("inference device must be auto, cpu, mps, or cuda")
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Metal is unavailable in this process. Use --device cpu or run in Terminal.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu, or mps on Apple silicon.")
    return torch.device(requested)


def available_devices():
    return ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (
        ["cuda"] if torch.cuda.is_available() else []
    )
