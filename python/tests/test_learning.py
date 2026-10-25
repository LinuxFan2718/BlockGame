import csv
import math

import pytest
import torch
from blokus_rl.env import FixedOpponentEnv
from blokus_rl.learning import (
    TrainConfig,
    collect_batch,
    configure_torch,
    discounted_returns,
    evaluate,
    load_checkpoint,
    make_ppo_loss,
    optimize_batch,
    save_checkpoint,
    train,
)
from blokus_rl.model import ActorCritic
from tensordict import TensorDict


@pytest.fixture(autouse=True)
def seeded_torch():
    configure_torch(42)


def test_model_shapes_masks_and_parameter_count():
    model = ActorCritic()
    cells = torch.zeros(3, 9)
    mask = torch.tensor([[True, False, True, False, False, False, False, False, False]] * 3)
    distribution, values = model.distribution(cells, mask)
    assert distribution.probs.shape == (3, 9)
    assert values.shape == (3, 1)
    assert torch.equal(distribution.probs[:, 1], torch.zeros(3))
    assert torch.allclose(distribution.probs.sum(-1), torch.ones(3))
    assert sum(parameter.numel() for parameter in model.parameters()) == 5450
    samples = distribution.sample((100,))
    assert ((samples == 0) | (samples == 2)).all()
    with pytest.raises(ValueError, match="terminal"):
        model.distribution(cells, torch.zeros_like(mask))


def test_returns_discount_future_rewards_without_crossing_episode_boundaries():
    assert discounted_returns([0, 0, -1], 1) == [-1, -1, -1]
    assert discounted_returns([0, 0, 1], 0.5) == [0.25, 0.5, 1]
    assert discounted_returns([0, 0], 1) == [0, 0]
    assert discounted_returns([], 1) == []


def test_rollouts_record_detached_legal_behavior_probabilities():
    model = ActorCritic()
    env = FixedOpponentEnv()
    env.reset(seed=42)
    batch, _ = collect_batch(model, env, TrainConfig(episodes_per_update=16))
    assert batch.batch_size == (len(batch),)
    assert not batch["action_log_prob"].requires_grad
    assert not batch["advantage"].requires_grad
    assert batch["action_mask"].gather(1, batch["action"].unsqueeze(-1)).all()
    distribution, _ = model.distribution(batch["observation"], batch["action_mask"])
    assert torch.allclose(
        distribution.log_prob(batch["action"]), batch["action_log_prob"], atol=1e-6
    )
    assert set(batch["value_target"].flatten().tolist()) <= {-1, 0, 1}


@pytest.mark.parametrize("advantage, expected", [(1.0, -1.2), (-1.0, 1.8)])
def test_ppo_clipping_matches_a_hand_calculated_probability_ratio(advantage, expected):
    model = ActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_head.bias[0] = math.log(9)  # New P(action 0) = 0.9 among two legal moves.
    batch = TensorDict(
        {
            "observation": torch.zeros(1, 9),
            "action_mask": torch.tensor(
                [[True, True, False, False, False, False, False, False, False]]
            ),
            "action": torch.tensor([0]),
            "action_log_prob": torch.tensor([math.log(0.5)]),  # Behavior P = 0.5, ratio = 1.8.
            "advantage": torch.tensor([[advantage]]),
            "value_target": torch.zeros(1, 1),
        },
        batch_size=[1],
    )
    loss = make_ppo_loss(model, TrainConfig())(batch)["loss_objective"]
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    if advantage > 0:
        assert model.policy_head.bias.grad[0].item() == pytest.approx(0)
    else:
        assert model.policy_head.bias.grad[0].abs() > 0


def test_ppo_updates_shared_layers_and_both_heads():
    model = ActorCritic()
    config = TrainConfig(episodes_per_update=16, epochs=2, minibatch_size=32)
    env = FixedOpponentEnv()
    env.reset(seed=42)
    batch, _ = collect_batch(model, env, config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    metrics = optimize_batch(
        model,
        make_ppo_loss(model, config),
        torch.optim.Adam(model.parameters(), lr=0.001),
        batch,
        config,
    )
    assert all(math.isfinite(value) for value in metrics.values())
    for name in ("trunk.0.weight", "policy_head.weight", "value_head.weight"):
        assert not torch.equal(model.state_dict()[name], before[name])


def test_evaluation_freezes_parameters_mode_and_rng():
    model = ActorCritic().train()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    first = evaluate(model, games_per_seat=10, seed=500)
    second = evaluate(model, games_per_seat=10, seed=500)
    assert first == second
    assert first["overall"]["games"] == 20
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name])


def test_seeded_training_updates_are_repeatable():
    def run():
        configure_torch(123)
        model = ActorCritic(16)
        config = TrainConfig(episodes_per_update=8, epochs=1, minibatch_size=16)
        env = FixedOpponentEnv()
        env.reset(seed=123)
        loss_module = make_ppo_loss(model, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for _ in range(2):
            batch, _ = collect_batch(model, env, config)
            optimize_batch(model, loss_module, optimizer, batch, config)
        return model.state_dict()

    first, second = run(), run()
    for key in first:
        assert torch.equal(first[key], second[key])


def test_checkpoint_round_trip_preserves_outputs_and_rng(tmp_path):
    model = ActorCritic(32)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, TrainConfig(hidden_size=32))
    rng = torch.get_rng_state().clone()
    loaded = load_checkpoint(path)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not loaded.training
    observations = torch.randn(10, 9)
    for actual, expected in zip(loaded(observations), model(observations), strict=True):
        assert torch.equal(actual, expected)


def test_training_smoke_saves_a_usable_model_and_metrics(tmp_path):
    out = tmp_path / "smoke"
    config = TrainConfig(
        updates=2,
        episodes_per_update=8,
        epochs=1,
        minibatch_size=16,
        eval_every=1,
        eval_games_per_seat=3,
        final_games_per_seat=5,
    )
    summary = train(config, out)
    assert summary["transitions"] >= 16
    assert summary["trained_policy"]["overall"]["games"] == 10
    assert load_checkpoint(out / "model.pt").hidden_size == config.hidden_size
    for filename in ("summary.json", "metrics.csv", "learning_curve.png", "learning_curve.svg"):
        assert (out / filename).stat().st_size > 0
    with (out / "metrics.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 2
    with pytest.raises(FileExistsError):
        train(config, out)


@pytest.mark.parametrize(
    "kwargs", [{"updates": 0}, {"epochs": 0}, {"seed": -1}, {"gamma": 1.1}, {"learning_rate": 0}]
)
def test_invalid_training_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
