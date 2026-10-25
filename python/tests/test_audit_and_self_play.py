import numpy as np
import pytest
import torch
from blokus_rl._native import Game
from blokus_rl.audit import (
    EMPTY,
    NeuralPolicy,
    audit_policy,
    game_graph,
    versus_frozen_policy,
)
from blokus_rl.env import FixedOpponentEnv
from blokus_rl.experiments import study
from blokus_rl.learning import TrainConfig, collect_batch, configure_torch
from blokus_rl.model import ActorCritic
from blokus_rl.search import SearchPolicy, collect_search_batch, optimize_search_batch
from blokus_rl.self_play import OpponentPool


@pytest.fixture(autouse=True)
def seeded_torch():
    configure_torch(42)


class PerfectPolicy:
    def __init__(self):
        self.graph = game_graph()
        self.predictions = {}
        for board, node in self.graph.positions.items():
            if node.children:
                actions = self.graph.optimal_actions(board)
                probabilities = np.zeros(9)
                probabilities[list(actions)] = 1 / len(actions)
                self.predictions[board] = (probabilities, 0)

    def action(self, board):
        return self.graph.optimal_actions(board)[0]


def test_rust_graph_and_perfect_policy_have_expected_certificates():
    graph = game_graph()
    assert len(graph.positions) == 5478
    assert graph.minimax(EMPTY, "X") == graph.minimax(EMPTY, "O") == 0
    result = audit_policy(PerfectPolicy())
    assert result["never_loses_both_seats"]
    for seat in ("X", "O"):
        assert result["by_seat"][seat]["vs_perfect"] == {"wins": 0, "draws": 1, "losses": 0}
        assert result["by_seat"][seat]["vs_random"]["losses"] == 0
        assert result["by_seat"][seat]["avoidable_action_errors"] == 0


def test_audit_witness_replays_to_a_real_rust_loss():
    class FirstLegal:
        def action(self, board):
            return min(game_graph().positions[board].children)

    result = audit_policy(FirstLegal())
    assert not result["never_loses_both_seats"]
    for seat in ("X", "O"):
        witness = result["by_seat"][seat]["losing_example_squares"]
        assert witness
        game = Game()
        for square in witness:
            game.step(square - 1)
        assert game.terminated
        assert game.outcome == ("O" if seat == "X" else "X")


def test_frozen_policy_evaluation_preserves_probability_mass():
    perfect = PerfectPolicy()
    result = versus_frozen_policy(perfect, perfect)
    for seat in ("X", "O"):
        assert result[seat]["draws"] == pytest.approx(1)
        assert result[seat]["wins"] == result[seat]["losses"] == 0


def test_cached_policy_matches_live_inference_without_changing_model_or_rng():
    model = ActorCritic().train()
    before = torch.get_rng_state().clone()
    policy = NeuralPolicy(model)
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    for board in list(policy.predictions)[::25]:
        node = policy.graph.positions[board]
        cells = torch.tensor(board, dtype=torch.float32) * (1 if node.player == "X" else -1)
        mask = torch.tensor([a in node.children for a in range(9)])
        with torch.no_grad():
            distribution, _ = model.distribution(cells, mask)
        assert policy.action(board) == int(distribution.mode)


def test_opponent_snapshots_are_immutable_and_pool_is_bounded():
    model = ActorCritic()
    pool = OpponentPool(model, seed=1, current_fraction=0, max_snapshots=2)
    original = pool.snapshots[0][1].policy_head.bias.clone()
    with torch.no_grad():
        model.policy_head.bias.add_(1)
    assert torch.equal(pool.snapshots[0][1].policy_head.bias, original)
    assert all(not parameter.requires_grad for parameter in pool.snapshots[0][1].parameters())
    pool.add_snapshot(10)
    pool.add_snapshot(20)
    assert [update for update, _ in pool.snapshots] == [10, 20]
    pool.begin_episode()
    assert pool.selected is not model
    assert pool.episode_counts == {"current": 0, "past": 1}


def test_self_play_selects_one_opponent_per_episode_and_records_on_policy_actions():
    model = ActorCritic()
    pool = OpponentPool(model, seed=1, current_fraction=1)
    env = FixedOpponentEnv(opponent=pool)
    env.reset(seed=42)
    batch, _ = collect_batch(
        model, env, TrainConfig(episodes_per_update=8), before_episode=pool.begin_episode
    )
    assert pool.episode_counts == {"current": 8, "past": 0}
    assert pool.selected is model
    with torch.no_grad():
        distribution, _ = model.distribution(batch["observation"], batch["action_mask"])
    assert torch.allclose(
        distribution.log_prob(batch["action"]), batch["action_log_prob"], atol=1e-6
    )


def test_search_finds_a_terminal_block_and_is_deterministic():
    model = ActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    policy = SearchPolicy(NeuralPolicy(model), simulations=32)
    # O must play middle-left (action 3); bottom-right permits an immediate X win.
    board = (1, 1, -1, 0, -1, 1, 1, -1, 0)
    first = policy.distribution(board)
    second = policy.distribution(board)
    np.testing.assert_array_equal(first, second)
    assert policy.action(board) == 3
    assert first[3] > first[8]
    assert first.sum() == pytest.approx(1)
    assert np.count_nonzero(first) == 2


def test_search_and_its_training_targets_do_not_use_the_evaluation_oracle(monkeypatch):
    graph = game_graph()

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not use minimax")

    monkeypatch.setattr(graph, "minimax", forbidden)
    monkeypatch.setattr(graph, "optimal_actions", forbidden)
    model = ActorCritic()
    config = TrainConfig(episodes_per_update=4, epochs=1, minibatch_size=16)
    batch, counts = collect_search_batch(model, config, graph, np.random.default_rng(42), 8)
    assert torch.allclose(batch["policy_target"].sum(-1), torch.ones(len(batch["policy_target"])))
    assert (batch["policy_target"][~batch["action_mask"]] == 0).all()
    assert set(batch["value_target"].flatten().tolist()) <= {-1, 0, 1}
    assert counts["search_simulations"] == 8 * len(batch["observation"])
    before = model.policy_head.weight.detach().clone()
    result = optimize_search_batch(
        model, torch.optim.Adam(model.parameters(), lr=0.001), batch, config
    )
    assert result["optimizer_steps"] > 0
    assert not torch.equal(before, model.policy_head.weight)


def test_three_arm_smoke_experiment_saves_reports_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "study"
    runs = study(out, seeds=[7], updates=2, episodes_per_update=4, simulations=4, audit_every=1)
    assert len(runs) == 3
    for arm in ("random", "selfplay", "alphazero"):
        directory = out / f"{arm}-seed-7"
        assert (directory / "best.pt").exists()
        assert (directory / "final.pt").exists()
        assert (directory / "summary.json").exists()
    for name in ("protocol.json", "comparison.json", "report.md", "comparison.png"):
        assert (out / name).stat().st_size > 0
    assert (
        runs[1]["opponent_episode_counts"]["current"] + runs[1]["opponent_episode_counts"]["past"]
        == 8
    )
    assert runs[2]["initial_with_search"] is not None
