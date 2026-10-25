"""Spatial policy with one output channel per oriented polyomino.

Unlike the additive actor, an oriented shape at an anchor has its own learned
score. Convolutions share geometric features across board locations. The legal
candidate list is still generated exclusively by Rust.
"""

import math

import torch
from torch import nn
from torchrl.modules import MaskedCategorical

from ._native import DuoGame
from .duo_env import CELLS, PASS, STATE_SIZE


class DuoConvActorCritic(nn.Module):
    architecture = "conv"

    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size, self.state_size = hidden_size, STATE_SIZE
        channels = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.context = nn.Sequential(nn.Linear(channels + 46, hidden_size), nn.Tanh())
        self.modulation = nn.Linear(hidden_size, channels * 2)
        self.orientation_head = nn.Conv2d(channels, 91, 1)
        self.piece_head = nn.Linear(hidden_size, 21)
        self.pass_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        for layer in (self.orientation_head, self.piece_head, self.pass_head):
            nn.init.orthogonal_(layer.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1)
        # Stable Rust orientation ids are derived from normalized geometry.
        # This also handles the observer's 180-degree coordinate rotation.
        game = DuoGame()
        codes, pieces = [], []
        for orientation in range(91):
            piece, _, _, cells = game.describe(orientation * CELLS)
            codes.append(sum(1 << ((cell // 14) * 5 + cell % 14) for cell in cells))
            pieces.append(piece)
        ordered = sorted((code, i) for i, code in enumerate(codes))
        ordered.insert(0, (0, 91))  # padded/forced-pass candidate
        self.register_buffer("shape_codes", torch.tensor([x[0] for x in ordered]))
        self.register_buffer("shape_ids", torch.tensor([x[1] for x in ordered]))
        self.register_buffer("orientation_pieces", torch.tensor(pieces))

    def forward(self, observation):
        batch_shape = observation.shape[:-1]
        flat = observation.reshape(-1, STATE_SIZE)
        board = flat[:, :392].reshape(-1, 2, 14, 14)
        starts = flat[:, 434:630].reshape(-1, 1, 14, 14)
        local = self.encoder(torch.cat([board, starts], dim=1))
        inventory_and_scores = torch.cat([flat[:, 392:434], flat[:, 630:]], dim=-1)
        context = self.context(torch.cat([local.mean((-2, -1)), inventory_and_scores], dim=-1))
        scale, shift = self.modulation(context).chunk(2, dim=-1)
        local = local * (1 + 0.1 * scale[..., None, None]) + 0.1 * shift[..., None, None]
        orientation_logits = self.orientation_head(local)
        orientation_logits = (
            orientation_logits + self.piece_head(context)[:, self.orientation_pieces, None, None]
        )
        scores = torch.cat([orientation_logits.flatten(1), self.pass_head(context)], dim=-1)
        return scores.reshape(*batch_shape, PASS + 1), self.value_head(context).reshape(
            *batch_shape, 1
        )

    def candidate_indices(self, candidates):
        cells = candidates[..., 1:].long()
        valid = cells < CELLS
        rows, cols = cells // 14, cells % 14
        rmin = torch.where(valid, rows, 14).min(-1).values
        cmin = torch.where(valid, cols, 14).min(-1).values
        dr = (rows - rmin.unsqueeze(-1)).clamp(0, 4)
        dc = (cols - cmin.unsqueeze(-1)).clamp(0, 4)
        bits = torch.bitwise_left_shift(torch.ones_like(cells), dr * 5 + dc)
        codes = (bits * valid).sum(-1)
        positions = torch.searchsorted(self.shape_codes, codes.contiguous())
        orientation = self.shape_ids[positions]
        indices = orientation * CELLS + rmin * 14 + cmin
        return torch.where(candidates[..., 0] == 21, PASS, indices)

    def logits(self, observation, candidates):
        scores, _ = self(observation)
        return scores.gather(-1, self.candidate_indices(candidates))

    def distribution(self, observation, candidates, mask=None):
        if mask is None:
            mask = torch.ones(candidates.shape[:-1], dtype=torch.bool)
        if not mask.any(-1).all():
            raise ValueError("empty candidate set")
        scores, value = self(observation)
        logits = scores.gather(-1, self.candidate_indices(candidates))
        return MaskedCategorical(logits=logits, mask=mask), value
