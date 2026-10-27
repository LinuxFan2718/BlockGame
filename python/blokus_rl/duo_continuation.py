"""Explicitly fork a completed self-play state into a new controlled experiment.

This is different from crash recovery: both branches inherit identical neural
weights, optimizer moments and replay, then use a declared new game/RNG seed.
The old study and its exact-config resume mechanism are left intact.
"""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .duo_deep import atomic_json
from .duo_learning import load_duo
from .learning import configure_torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize_branch(config, out, reference, source_resume, *, selected_best=False):
    """Create round zero; require the reference to equal the source's latest model.

    With selected_best, require the source's saved best weights instead and reset
    AdamW: the source optimizer belongs to its latest, possibly different model.
    Both branches can still inherit exactly the same off-policy replay.
    The unchanged parent is a selection candidate with score 0.5 against itself.
    New checkpoints must improve on it on validation. Counters start at zero for
    additional training; the parent's games/round are recorded separately.
    """
    out, reference, source_resume = map(Path, (out, reference, source_resume))
    if out.exists():
        raise ValueError("branch directory already exists; use exact-config resume")
    saved = torch.load(source_resume, map_location="cpu", weights_only=False)
    parent = load_duo(reference)
    parent_config = saved["config"]
    for key in ("architecture", "hidden_size", "learning_rate"):
        if parent_config[key] != getattr(config, key):
            raise ValueError(f"fork changes unsupported setting: {key}")
    if config.architecture == "residual":
        for key in ("channels", "blocks"):
            if parent_config[key] != getattr(config, key):
                raise ValueError(f"fork changes unsupported setting: {key}")
    weights = parent.state_dict()
    expected = saved["best_checkpoint"]["model_state"] if selected_best else saved["model_state"]
    if weights.keys() != expected.keys() or any(
        not torch.equal(value, expected[key]) for key, value in weights.items()
    ):
        raise ValueError("source resume weights do not match the frozen reference")
    if len(saved["replay"]) > config.replay_positions:
        raise ValueError("fork would silently truncate inherited replay")
    if any(group["lr"] != config.learning_rate for group in saved["optimizer"]["param_groups"]):
        raise ValueError("fork learning rate does not match inherited optimizer")
    reference_hash, source_hash = sha256(reference), sha256(source_resume)
    configure_torch(config.seed)
    state = dict(saved)
    if selected_best:
        state["model_state"] = weights
        state["optimizer"] = torch.optim.AdamW(
            parent.parameters(), lr=config.learning_rate, weight_decay=1e-4
        ).state_dict()
    state.update(
        config=asdict(config),
        reference_sha256=reference_hash,
        round=0,
        games=0,
        best=0.5,
        elapsed_seconds=0.0,
        numpy_rng=np.random.default_rng(config.seed).bit_generator.state,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
        best_checkpoint=torch.load(reference, map_location="cpu", weights_only=True),
    )
    # All checks precede creation, so rejected forks leave no misleading run.
    out.mkdir(parents=True)
    for name in ("initial.pt", "latest.pt", "best.pt"):
        (out / name).write_bytes(reference.read_bytes())
    atomic_json(
        out / "protocol.json",
        {
            "config": asdict(config),
            "reference_sha256": reference_hash,
            "source_resume_sha256": source_hash,
            "parent_round": saved["round"],
            "parent_games": saved["games"],
            "inherited_replay_positions": len(saved["replay"]),
            "selected_parent_update": torch.load(reference, map_location="cpu", weights_only=True)[
                "update"
            ],
            "optimizer_reset": selected_best,
            "initialization": (
                "saved best neural weights; fresh AdamW; inherited latest replay; fresh declared RNG seed"
                if selected_best
                else "parent neural weights, AdamW moments and replay; fresh declared RNG seed"
            ),
            "training": "additional pure self-play; no expert labels; full-search visit CE + terminal WDL MSE",
            "selection": "parent included at 0.5; searched validation vs frozen parent, equal simulations",
            "parameters": sum(p.numel() for p in parent.parameters()),
        },
    )
    torch.save(state, out / "resume.tmp")
    (out / "resume.tmp").replace(out / "resume.pt")
