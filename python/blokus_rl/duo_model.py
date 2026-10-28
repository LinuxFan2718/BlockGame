"""A compositional move scorer: board-square scores + a piece score per candidate."""

import math

import torch
from torch import nn
from torchrl.modules import MaskedCategorical

from .duo_env import CELLS, PIECES, STATE_SIZE


class DuoActorCritic(nn.Module):
    def __init__(self, hidden_size=128, state_size=STATE_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, CELLS + PIECES + 1)
        self.value_head = nn.Linear(hidden_size, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, observation):
        hidden = self.trunk(observation)
        return self.policy_head(hidden), self.value_head(hidden)

    def logits(self, observation, candidates):
        scores, _ = self(observation)
        # Work with arbitrary batch dimensions, including a single live decision.
        square_scores = torch.cat([scores[..., :CELLS], scores[..., :1] * 0], dim=-1)
        cells = candidates[..., 1:].long()
        shape = cells.shape
        square_term = square_scores.gather(-1, cells.flatten(-2)).reshape(shape).sum(-1)
        piece_term = scores[..., CELLS:].gather(-1, candidates[..., 0].long())
        return square_term + piece_term

    def distribution(self, observation, candidates, mask=None):
        if mask is None:
            mask = torch.ones(candidates.shape[:-1], dtype=torch.bool, device=candidates.device)
        if not mask.any(-1).all():
            raise ValueError("empty candidate set")
        return MaskedCategorical(logits=self.logits(observation, candidates), mask=mask), self(
            observation
        )[1]


class DuoPolicyModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observation, candidates):
        return self.model.logits(observation, candidates)


class DuoValueModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observation):
        return self.model(observation)[1]
