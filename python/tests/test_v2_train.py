import numpy as np
import torch

from blokus_rl._native import DuoGame, DuoReplay
from blokus_rl.v2 import train as trainer
from blokus_rl.v2.model import DuoNetV2


def random_games(count, seed):
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(count):
        game, moves = DuoGame(), []
        while not game.terminated:
            action = int(rng.choice(game.legal_actions()))
            game.step(action)
            moves.append(action)
        games.append(moves)
    return games


def test_losses_fall_when_fitting_a_fixed_batch():
    torch.manual_seed(0)
    replay = DuoReplay(10_000)
    for moves in random_games(8, 1):
        replay.add_moves(moves)
    config = trainer.TrainConfig(blocks=1, channels=16, device="cpu", learning_rate=3e-3)
    learner = trainer.Learner(config, DuoNetV2(1, 16, 16))
    sample = replay.sample(64, len(replay), 3, True)
    learner.step(sample)
    first = learner.drain_metrics()
    for _ in range(60):
        learner.step(sample)
    last = learner.drain_metrics()
    for _ in range(5):
        learner.step(sample)
    final = learner.drain_metrics()
    assert final["policy"] < first["policy"] * 0.7
    assert final["value"] < first["value"]
    assert final["ownership"] < first["ownership"]
    assert np.isfinite(last["grad_norm"])


def test_checkpoint_round_trip(tmp_path):
    config = trainer.TrainConfig(blocks=1, channels=16, device="cpu")
    learner = trainer.Learner(config, DuoNetV2(1, 16, 64))
    replay = DuoReplay(1000)
    for moves in random_games(2, 2):
        replay.add_moves(moves)
    learner.step(replay.sample(16, len(replay), 1, True))
    trainer.save_checkpoint(tmp_path, learner, replay, {"games": 2, "elapsed": 1.0}, config)
    state = torch.load(tmp_path / "state.pt", weights_only=False)
    other = trainer.Learner(config, DuoNetV2(1, 16, 64))
    other.restore(state["learner"])
    assert other.steps == 1
    restored = DuoReplay.from_bytes((tmp_path / "replay.bin").read_bytes())
    assert len(restored) == len(replay)
    for a, b in zip(other.ema.state_dict().values(), learner.ema.state_dict().values()):
        assert torch.equal(a, b)
