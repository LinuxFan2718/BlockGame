import numpy as np
import pytest
from blokus_rl._native import Game
from blokus_rl.env import FixedOpponentEnv, make_aec_env
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test, seed_test


def scripted_opponent(actions):
    actions = iter(actions)
    return lambda observation, rng: next(actions)


def test_native_binding_preserves_errors_and_observation_perspectives():
    game = Game()
    assert game.current_player == "X"
    assert game.step(0) == (0.0, 0.0, False)
    before = game.observe("X")
    with pytest.raises(ValueError, match="occupied"):
        game.step(0)
    with pytest.raises(ValueError, match="outside"):
        game.step(9)
    with pytest.raises((OverflowError, ValueError)):
        game.step(-1)
    assert game.observe("X") == before
    assert game.current_player == "O"
    assert game.observe("X")[0][0] == 1
    assert game.observe("O")[0][0] == -1
    with pytest.raises(ValueError, match="X or O"):
        game.observe("Z")
    game.reset()
    assert game.legal_actions() == list(range(9))


def test_pettingzoo_api_and_seed_contracts():
    api_test(make_aec_env(), num_cycles=100)
    seed_test(make_aec_env, num_cycles=50)


def test_gymnasium_contract():
    check_env(FixedOpponentEnv(), skip_render_check=True)


def test_torchrl_can_wrap_the_aec_environment_and_collect_a_rollout():
    from torchrl.envs import PettingZooWrapper
    from torchrl.envs.utils import check_env_specs

    wrapped = PettingZooWrapper(make_aec_env(), use_mask=True)
    check_env_specs(wrapped)
    rollout = wrapped.rollout(10)
    assert rollout.numel() > 0
    wrapped.close()


def test_aec_delivers_terminal_feedback_to_both_players():
    env = make_aec_env()
    env.reset()
    for action in [0, 3, 1, 4, 2]:
        env.step(action)
    terminal_rewards = {}
    for player in env.agent_iter():
        observation, reward, terminated, truncated, _ = env.last()
        assert terminated and not truncated
        assert not observation["action_mask"].any()
        terminal_rewards[player] = reward
        env.step(None)
    assert terminal_rewards == {"O": -1.0, "X": 1.0}
    assert env.agents == []


def test_opponents_turn_has_an_empty_action_mask():
    env = make_aec_env()
    env.reset()
    assert env.observe("X")["action_mask"].all()
    assert not env.observe("O")["action_mask"].any()
    env.step(4)
    assert env.observe("O")["action_mask"].sum() == 8
    assert not env.observe("X")["action_mask"].any()


def test_fixed_opponent_reports_loss_on_the_opponents_reply():
    env = FixedOpponentEnv(opponent=scripted_opponent([3, 4, 5]))
    env.reset(seed=42, options={"learner": "X"})
    for action in [0, 1]:
        observation, reward, terminated, _, _ = env.step(action)
        assert reward == 0 and not terminated
        assert env.aec.agent_selection == "X"
    observation, reward, terminated, truncated, info = env.step(8)
    assert (reward, terminated, truncated) == (-1.0, True, False)
    assert info == {"learner": "X", "outcome": "O"}
    assert not observation["action_mask"].any()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(6)


def test_fixed_opponent_stops_immediately_after_learner_wins():
    # There is intentionally no third opponent action in the iterator.
    env = FixedOpponentEnv(opponent=scripted_opponent([3, 4]))
    env.reset(options={"learner": "X"})
    env.step(0)
    env.step(1)
    _, reward, terminated, _, _ = env.step(2)
    assert reward == 1 and terminated


def test_o_observes_the_opening_x_move_and_can_win():
    env = FixedOpponentEnv(opponent=scripted_opponent([0, 1, 8]))
    observation, info = env.reset(seed=1, options={"learner": "O"})
    assert observation["observation"][0] == -1
    assert observation["action_mask"][0] == 0
    assert info["learner"] == "O"
    env.step(3)
    env.step(4)
    observation, reward, terminated, _, info = env.step(5)
    assert reward == 1 and terminated and info["outcome"] == "O"
    assert not observation["action_mask"].any()


def test_fixed_opponent_draw_and_alternating_seats():
    env = FixedOpponentEnv(opponent=scripted_opponent([1, 4, 5, 6]))
    env.reset(options={"learner": "X"})
    for action in [0, 2, 3, 7, 8]:
        _, reward, terminated, _, info = env.step(action)
    assert reward == 0 and terminated and info["outcome"] == "draw"
    env = FixedOpponentEnv()
    seats = [env.reset()[1]["learner"] for _ in range(4)]
    assert seats == ["X", "O", "X", "O"]
    first, _ = env.reset(seed=5, options={"learner": "O"})
    second, _ = env.reset(seed=5, options={"learner": "O"})
    np.testing.assert_array_equal(first["observation"], second["observation"])


def test_invalid_action_does_not_advance_the_python_environment():
    env = FixedOpponentEnv(opponent=scripted_opponent([4]))
    env.reset(options={"learner": "X"})
    env.step(0)
    before = env.aec.game.observe("X")
    for action in [0, 9, -1, 0.5, None]:
        with pytest.raises(ValueError):
            env.step(action)
        assert env.aec.game.observe("X") == before
        assert env.aec.agent_selection == "X"
