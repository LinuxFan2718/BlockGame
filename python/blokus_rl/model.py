"""The model is ordinary PyTorch. PPO will decide how to train its weights."""

import math

import torch
from torch import nn
from torchrl.modules import MaskedCategorical


class ActorCritic(nn.Module):
    """A shared MLP with a move-score head (actor) and value head (critic)."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, 9)
        self.value_head = nn.Linear(hidden_size, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        # Begin close to a uniform policy. These are initialization choices,
        # not game-specific strategy; every weight will be learned.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, observation: torch.Tensor):
        features = self.trunk(observation)
        return self.policy_head(features), self.value_head(features)

    def distribution(self, observation: torch.Tensor, action_mask: torch.Tensor):
        if not action_mask.any(dim=-1).all():
            raise ValueError("cannot choose an action from a terminal/all-false mask")
        logits, value = self(observation)
        return MaskedCategorical(logits=logits, mask=action_mask.bool()), value


class PolicyModule(nn.Module):
    """Present the actor branch to TorchRL without copying model parameters."""

    def __init__(self, model: ActorCritic):
        super().__init__()
        self.model = model

    def forward(self, observation):
        return self.model(observation)[0]


class ValueModule(nn.Module):
    def __init__(self, model: ActorCritic):
        super().__init__()
        self.model = model

    def forward(self, observation):
        return self.model(observation)[1]
