"""Residual policy/value network with global pooling and auxiliary heads.

Inputs are the Rust v2 planes (rules-derived, mover's frame) and scalars
(inventories, scores, mobility). Outputs:

- policy: 91 oriented-shape logits at each of 196 anchors, plus pass (17,837)
- outcome: win/draw/loss logits for the player to move
- margin: final score margin (own minus opponent), scaled by MARGIN_SCALE
- ownership: final owner of each cell (empty/own/opponent), auxiliary only

Every block mixes in a global summary (mean and max over the board, plus the
scalars), so distant regions and the remaining pieces inform local features,
in the style of KataGo's global pooling.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .._native import duo_v2_shapes

PLANES, SCALARS, POLICY_SIZE, SYMMETRIES = duo_v2_shapes()
ORIENTATIONS, CELLS, SIZE = 91, 196, 14
MARGIN_SCALE = 20.0


def pool(x):
    """Mean and max over the board. Flattening first is ~20x faster on Metal
    than reducing over two dimensions."""
    flat = x.flatten(2)
    return torch.cat([flat.mean(2), flat.amax(2)], 1)


class GlobalBias(nn.Module):
    """Per-channel bias from the board's mean/max features and the scalars."""

    def __init__(self, channels, context):
        super().__init__()
        self.linear = nn.Linear(2 * channels + context, channels)

    def forward(self, x, context):
        pooled = torch.cat([pool(x), context], 1)
        return x + self.linear(pooled)[:, :, None, None]


class Block(nn.Module):
    def __init__(self, channels, context):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.pool = GlobalBias(channels, context)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.norm2.weight)  # Each block starts as the identity.

    def forward(self, x, context):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.pool(h, context)
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class DuoNetV2(nn.Module):
    def __init__(self, blocks=6, channels=64, context=64):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels, "context": context}
        self.context = nn.Sequential(nn.Linear(SCALARS, context), nn.ReLU())
        self.stem = nn.Conv2d(PLANES, channels, 3, padding=1, bias=False)
        self.stem_scalars = nn.Linear(context, channels)
        self.stem_norm = nn.BatchNorm2d(channels)
        self.trunk = nn.ModuleList(Block(channels, context) for _ in range(blocks))
        head = channels // 2
        self.policy_conv = nn.Conv2d(channels, head, 1)
        self.policy_pool = GlobalBias(head, context)
        self.policy_out = nn.Conv2d(head, ORIENTATIONS, 1)
        self.policy_shape = nn.Linear(2 * head + context, ORIENTATIONS + 1)
        self.value_conv = nn.Conv2d(channels, head, 1)
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * head + context, 128), nn.ReLU(), nn.Linear(128, 4)
        )
        self.ownership_out = nn.Conv2d(channels, 3, 1)

    def forward(self, planes, scalars, ownership=False):
        """planes: (N, PLANES, 14, 14) float; scalars: (N, SCALARS)."""
        context = self.context(scalars)
        x = self.stem(planes) + self.stem_scalars(context)[:, :, None, None]
        x = F.relu(self.stem_norm(x))
        for block in self.trunk:
            x = block(x, context)
        p = F.relu(self.policy_conv(x))
        p = self.policy_pool(p, context)
        pooled = torch.cat([pool(p), context], 1)
        shape = self.policy_shape(pooled)
        logits = self.policy_out(p) + shape[:, :ORIENTATIONS, None, None]
        policy = torch.cat([logits.flatten(1), shape[:, ORIENTATIONS:]], 1)
        v = F.relu(self.value_conv(x))
        v = self.value_mlp(torch.cat([pool(v), context], 1))
        outcome, margin = v[:, :3], v[:, 3]
        owner = self.ownership_out(x) if ownership else None
        return policy, outcome, margin, owner


def build(config):
    return DuoNetV2(**config)


def save(path, model, extra=None):
    torch.save(
        {
            "format": "duo-v2",
            "config": model.config,
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            **(extra or {}),
        },
        path,
    )


def load(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") != "duo-v2":
        raise ValueError(f"{path} is not a v2 checkpoint")
    model = build(data["config"])
    model.load_state_dict(data["state"])
    return model.to(device).eval(), data
