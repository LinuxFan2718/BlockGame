"""Blokus Duo observations and PettingZoo adapter. Rust owns all game rules."""

from functools import lru_cache

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from ._native import DuoGame

SIZE, CELLS, PIECES = 14, 196, 21
PASS = 91 * CELLS
ACTION_COUNT = PASS + 1
LEGACY_STATE_SIZE = 2 * CELLS + 2 * PIECES + CELLS + 1
STATE_SIZE = LEGACY_STATE_SIZE + 3


def state_vector(game: DuoGame, player: int) -> np.ndarray:
    """Own/opponent planes, inventories, active start squares, first-player flag.

    Player 1's coordinates rotate 180 degrees. Colors are always relative to the
    observer. Inventory and board together are essential: the same occupied
    squares can leave different pieces available in future positions.
    """
    board = np.frombuffer(game.board, dtype=np.uint8)
    starts = np.zeros(CELLS, dtype=np.float32)
    if game.current_player == player:
        starts[game.required_starts()] = 1
    own = (board == player + 1).astype(np.float32)
    other = (board == 2 - player).astype(np.float32)
    if player == 1:
        own, other, starts = own[::-1], other[::-1], starts[::-1]
    scores = game.scores
    return np.concatenate(
        [
            own,
            other,
            game.remaining(player),
            game.remaining(1 - player),
            starts,
            [float(player == 0)],
            [
                (scores[player] + 89) / 109,
                (scores[1 - player] + 89) / 109,
                game.consecutive_passes / 2,
            ],
        ]
    ).astype(np.float32)


def decision(game: DuoGame):
    """Stable engine actions plus compact candidate encodings for the network.

    Candidate = [piece_id, cell0, ..., cell4]. Cell 196 means padding;
    piece 21 means a forced pass. These are geometry, not heuristic move scores.
    """
    return decode_decision(game.decision_data())


def decode_decision(buffers):
    observation, actions, candidates = buffers
    return (
        np.frombuffer(observation, dtype="<f4"),
        np.frombuffer(actions, dtype="<i4"),
        np.frombuffer(candidates, dtype="<i2").reshape(-1, 6),
    )


def decode_indexed_decision(buffers):
    observation, actions, indices = buffers
    return (
        np.frombuffer(observation, dtype="<f4"),
        np.frombuffer(actions, dtype="<i4"),
        np.frombuffer(indices, dtype="<u2"),
    )


def legacy_decision(game: DuoGame):
    """Independent reference encoding retained for exact compatibility checks."""
    if game.terminated:
        raise ValueError("terminal positions have no decision")
    player = game.current_player
    actions, candidates = game.candidates()
    candidates = np.asarray(candidates, dtype=np.int16)
    if player == 1:
        cells = candidates[:, 1:]
        cells[:] = np.where(cells < CELLS, CELLS - 1 - cells, CELLS)
    return state_vector(game, player), np.asarray(actions, dtype=np.int32), candidates


class DuoEnv(AECEnv):
    """One turn per step, including forced passes. Rewards: terminal +/-1 or 0."""

    metadata = {
        "name": "rust_blokus_duo_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
    }
    possible_agents = ["player_0", "player_1"]

    def __init__(self, render_mode=None):
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("invalid render mode")
        self.render_mode = render_mode
        self.game = DuoGame()

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return spaces.Dict(
            observation=spaces.Box(0, 1, shape=(STATE_SIZE,), dtype=np.float32),
            action_mask=spaces.MultiBinary(ACTION_COUNT),
        )

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return spaces.Discrete(ACTION_COUNT)

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.agents = self.possible_agents.copy()
        self.agent_selection = self.agents[0]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = self.rewards.copy()
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = self.terminations.copy()
        self.infos = {a: {} for a in self.agents}
        if seed is not None:
            for i, agent in enumerate(self.agents):
                self.action_space(agent).seed(seed + i)

    def observe(self, agent):
        player = self.possible_agents.index(agent)
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        if agent == self.agent_selection and not self.game.terminated:
            mask[self.game.legal_actions()] = 1
        return {"observation": state_vector(self.game, player), "action_mask": mask}

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            return self._was_dead_step(action)
        if action is None or not self.action_space(agent).contains(action):
            raise ValueError("a live player must choose an integer action")
        r0, r1, terminal = self.game.step(int(action))
        self._cumulative_rewards[agent] = 0.0
        self.rewards = dict(zip(self.possible_agents, (r0, r1), strict=True))
        self.terminations = dict.fromkeys(self.agents, terminal)
        self.agent_selection = self.possible_agents[1 - self.possible_agents.index(agent)]
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode == "human":
            print(self.game.render())
        elif self.render_mode == "ansi":
            return self.game.render()

    def close(self):
        pass
