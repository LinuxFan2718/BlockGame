"""Residual spatial policy conditioned on inventory, with batch-independent norms.

The Rust engine owns legality. This network predicts oriented-placement logits
and a terminal WDL value from the observer's normalized board coordinates.
"""

import numpy as np
import torch
from torch import nn
from torchrl.modules import MaskedCategorical

from .duo_conv import DuoConvActorCritic, install_orientation_lookup
from .duo_env import CELLS, PASS, STATE_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x, scale, shift):
        residual = torch.relu(self.norm1(self.conv1(x)))
        residual = self.norm2(self.conv2(residual))
        residual = residual * (1 + scale[..., None, None]) + shift[..., None, None]
        return torch.relu(x + residual)


class DuoResidualActorCritic(nn.Module):
    architecture = "residual"

    def __init__(self, hidden_size=128, channels=64, blocks=10):
        super().__init__()
        if channels < 8 or channels % 8 or blocks < 1:
            raise ValueError("positive blocks and channels divisible by eight required")
        self.hidden_size, self.channels, self.blocks = hidden_size, channels, blocks
        self.state_size = STATE_SIZE
        self.stem = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
        )
        self.trunk = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.context = nn.Sequential(nn.Linear(46, hidden_size), nn.ReLU())
        self.modulation = nn.Linear(hidden_size, (blocks + 2) * 2 * channels)
        self.policy_features = nn.Conv2d(channels, channels, 1)
        self.policy_head = nn.Conv2d(channels, 91, 1)
        self.value_features = nn.Conv2d(channels, channels, 1)
        self.value_head = nn.Sequential(
            nn.Linear(channels + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )
        self.pass_head = nn.Linear(hidden_size, 1)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.modulation.weight, std=1e-3)
        for module in (self.policy_head, self.pass_head, self.value_head[-2]):
            nn.init.normal_(module.weight, std=0.01)
        coords = torch.linspace(0, 1, 14)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy])[None])
        install_orientation_lookup(self)

    def forward(self, observation):
        shape = observation.shape[:-1]
        flat = observation.reshape(-1, STATE_SIZE)
        board = torch.cat(
            [flat[:, :392].reshape(-1, 2, 14, 14), flat[:, 434:630].reshape(-1, 1, 14, 14)],
            dim=1,
        )
        context = self.context(torch.cat([flat[:, 392:434], flat[:, 630:]], dim=-1))
        modulation = self.modulation(context).reshape(-1, self.blocks + 2, 2, self.channels)
        x = self.stem(torch.cat([board, self.coordinates.expand(len(flat), -1, -1, -1)], 1))
        for i, block in enumerate(self.trunk):
            x = block(x, modulation[:, i, 0], modulation[:, i, 1])

        def modulate(features, index):
            scale, shift = modulation[:, index, 0], modulation[:, index, 1]
            return torch.relu(features * (1 + scale[..., None, None]) + shift[..., None, None])

        policy = self.policy_head(modulate(self.policy_features(x), self.blocks))
        value = modulate(self.value_features(x), self.blocks + 1).mean((-2, -1))
        scores = torch.cat([policy.flatten(1), self.pass_head(context)], -1)
        return scores.reshape(*shape, PASS + 1), self.value_head(
            torch.cat([value, context], -1)
        ).reshape(*shape, 1)

    candidate_indices = DuoConvActorCritic.candidate_indices

    def logits(self, observation, candidates):
        return self(observation)[0].gather(-1, self.candidate_indices(candidates))

    def distribution(self, observation, candidates, mask=None):
        if mask is None:
            mask = torch.ones(candidates.shape[:-1], dtype=torch.bool, device=candidates.device)
        if not mask.any(-1).all():
            raise ValueError("empty candidate set")
        scores, value = self(observation)
        return MaskedCategorical(
            logits=scores.gather(-1, self.candidate_indices(candidates)), mask=mask
        ), value


_LOOKUP = None


def candidate_indices_numpy(candidates):
    """Encode canonical placements on the worker, before crossing the GPU boundary."""
    global _LOOKUP
    if _LOOKUP is None:
        # Construct only the fixed geometry lookup, not a neural network.
        holder = nn.Module()
        install_orientation_lookup(holder)
        _LOOKUP = (holder.shape_codes.numpy(), holder.shape_ids.numpy())
    cells = candidates[:, 1:].astype(np.int64)
    valid = cells < CELLS
    rows, cols = cells // 14, cells % 14
    rmin = np.where(valid, rows, 14).min(-1)
    cmin = np.where(valid, cols, 14).min(-1)
    dr = np.clip(rows - rmin[:, None], 0, 4)
    dc = np.clip(cols - cmin[:, None], 0, 4)
    codes = ((np.int64(1) << (dr * 5 + dc)) * valid).sum(-1)
    positions = np.searchsorted(_LOOKUP[0], codes)
    if np.any(positions >= len(_LOOKUP[0])) or not np.array_equal(_LOOKUP[0][positions], codes):
        raise ValueError("unknown candidate geometry")
    indices = _LOOKUP[1][positions] * CELLS + rmin * 14 + cmin
    return np.where(candidates[:, 0] == 21, PASS, indices).astype(np.int32)
