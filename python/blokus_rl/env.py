"""Environment adapters. These call Rust; they do not implement game rules."""

from collections.abc import Callable
from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from ._native import Game

Observation = dict[str, np.ndarray]
Opponent = Callable[[Observation, np.random.Generator], int]


class TicTacToeEnv(AECEnv):
    """PettingZoo's Agent Environment Cycle: one Rust move per AEC step."""

    metadata = {
        "name": "rust_tic_tac_toe_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
    }
    possible_agents = ["X", "O"]

    def __init__(self, render_mode: str | None = None):
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, human, or ansi")
        self.render_mode = render_mode
        self.game = Game()

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return spaces.Dict(
            observation=spaces.Box(-1, 1, shape=(9,), dtype=np.float32),
            action_mask=spaces.MultiBinary(9),
        )

    @lru_cache(maxsize=None)
    def action_space(self, agent: str):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return spaces.Discrete(9)

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.agents = self.possible_agents.copy()
        self.agent_selection = "X"
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = self.rewards.copy()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        if seed is not None:
            for offset, agent in enumerate(self.agents):
                self.action_space(agent).seed(seed + offset)

    def observe(self, agent: str) -> Observation:
        cells, mask, _ = self.game.observe(agent)
        # PettingZoo exposes a mask for each player; only the acting player can act.
        if agent != self.agent_selection:
            mask = [False] * 9
        return {
            "observation": np.asarray(cells, dtype=np.float32),
            "action_mask": np.asarray(mask, dtype=np.int8),
        }

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            return self._was_dead_step(action)
        if action is None:
            raise ValueError("a live player must choose an action")
        if not self.action_space(agent).contains(action):
            raise ValueError("action must be an integer from 0 to 8")

        # Validate in Rust before changing Python bookkeeping.
        x_reward, o_reward, terminated = self.game.step(int(action))
        self._cumulative_rewards[agent] = 0.0
        self.rewards = {"X": x_reward, "O": o_reward}
        self.terminations = {name: terminated for name in self.agents}
        self.agent_selection = "O" if agent == "X" else "X"
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self):
        board = self.game.render()
        if self.render_mode == "human":
            print(board)
            return None
        if self.render_mode == "ansi":
            return board
        return None

    def close(self):
        pass


def make_aec_env(render_mode=None):
    """Standard PettingZoo wrappers, without changing Rust's illegal-move rules."""
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(TicTacToeEnv(render_mode))
    )


class FixedOpponentEnv(gym.Env):
    """One step = learner move + (if needed) frozen opponent response.

    PPO always sees successive decisions from the SAME player's perspective.
    Each reset alternates seats, unless options={"learner": "X" or "O"} is given.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, opponent: Opponent | None = None, render_mode=None):
        self.aec = TicTacToeEnv(render_mode=render_mode)
        self.observation_space = self.aec.observation_space("X")
        self.action_space = spaces.Discrete(9)
        self.render_mode = render_mode
        self.opponent = opponent
        self._next_learner = "X"
        self._finished = True

    def _opponent_move(self):
        observation = self.aec.observe(self.aec.agent_selection)
        if self.opponent is None:
            action = int(self.np_random.choice(np.flatnonzero(observation["action_mask"])))
        else:
            action = self.opponent(observation, self.np_random)
        self.aec.step(action)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # A seeded reset is reproducible even after prior episodes.
        if seed is not None:
            self._next_learner = "X"
            self.action_space.seed(seed)
        self.learner = (options or {}).get("learner", self._next_learner)
        if self.learner not in ("X", "O"):
            raise ValueError("learner must be X or O")
        self._next_learner = "O" if self.learner == "X" else "X"
        self.aec.reset(seed=seed)
        if self.learner == "O":
            self._opponent_move()  # X opens before O's first decision.
        self._finished = False
        return self.aec.observe(self.learner), self._info()

    def _info(self):
        return {"learner": self.learner, "outcome": self.aec.game.outcome}

    def step(self, action):
        if self._finished:
            raise RuntimeError("call reset before stepping a finished environment")
        if self.aec.agent_selection != self.learner:
            raise RuntimeError("internal turn mismatch")
        self.aec.step(action)
        if not self.aec.game.terminated:
            self._opponent_move()
        self._finished = self.aec.game.terminated
        reward = self.aec.rewards[self.learner]
        return (
            self.aec.observe(self.learner),
            reward,
            self._finished,
            False,  # At most nine marks: no time-limit truncations.
            self._info(),
        )

    def render(self):
        return self.aec.render()

    def close(self):
        self.aec.close()
