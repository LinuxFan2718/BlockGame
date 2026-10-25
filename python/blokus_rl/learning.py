"""An explicit on-policy PPO loop, using TorchRL's clipped PPO implementation.

We collect COMPLETE episodes. This lets us start with Monte Carlo returns instead
of introducing GAE and truncated-rollout bootstrapping at the same time as PPO.
"""

import csv
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchrl
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MaskedCategorical, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss

from .env import FixedOpponentEnv
from .model import ActorCritic, PolicyModule, ValueModule


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    updates: int = 300
    episodes_per_update: int = 128
    epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 64
    learning_rate: float = 0.0005
    gamma: float = 1.0
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    eval_every: int = 10
    eval_games_per_seat: int = 100
    final_games_per_seat: int = 1000

    def __post_init__(self):
        for name in (
            "updates",
            "episodes_per_update",
            "epochs",
            "minibatch_size",
            "hidden_size",
            "eval_every",
            "eval_games_per_seat",
            "final_games_per_seat",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def configure_torch(seed: int):
    # Small networks are faster with one CPU thread. This also makes seeded runs
    # reproducible on the same hardware and locked dependency versions.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def observation_tensors(observation):
    return (
        torch.as_tensor(observation["observation"], dtype=torch.float32),
        torch.as_tensor(observation["action_mask"], dtype=torch.bool),
    )


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Work backward through ONE episode, never across an episode boundary."""
    result = [0.0] * len(rewards)
    future_return = 0.0
    for index in reversed(range(len(rewards))):
        future_return = rewards[index] + gamma * future_return
        result[index] = future_return
    return result


@torch.no_grad()
def collect_batch(
    model: ActorCritic, env: FixedOpponentEnv, config: TrainConfig, *, before_episode=None
):
    observations, masks, actions, log_probs, values, targets = [], [], [], [], [], []
    episode_rewards = []
    for _ in range(config.episodes_per_update):
        if before_episode is not None:
            before_episode()
        observation, _ = env.reset()
        rewards = []
        while True:
            cells, mask = observation_tensors(observation)
            distribution, value = model.distribution(cells, mask)
            action = distribution.sample()
            observations.append(cells)
            masks.append(mask)
            actions.append(action)
            # Freeze the BEHAVIOR policy's log probability and value estimate.
            # PPO compares future model outputs with these recorded numbers.
            log_probs.append(distribution.log_prob(action))
            values.append(value)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            rewards.append(reward)
            if terminated or truncated:
                if truncated:
                    raise RuntimeError("this collector expects naturally completed games")
                break
        targets.extend(discounted_returns(rewards, config.gamma))
        episode_rewards.append(sum(rewards))

    value_target = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    advantage = value_target - torch.stack(values)
    # Normalize once over the rollout; do not change targets on later PPO epochs.
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    batch = TensorDict(
        {
            "observation": torch.stack(observations),
            "action_mask": torch.stack(masks),
            "action": torch.stack(actions),
            "action_log_prob": torch.stack(log_probs),
            "advantage": advantage,
            "value_target": value_target,
        },
        batch_size=[len(actions)],
    )
    return batch, float(np.mean(episode_rewards))


def make_ppo_loss(model: ActorCritic, config: TrainConfig):
    # These adapters reference the SAME model. The optimizer below updates each
    # shared parameter once. TorchRL implements the PPO objective, not our rules.
    actor = ProbabilisticActor(
        module=TensorDictModule(PolicyModule(model), in_keys=["observation"], out_keys=["logits"]),
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
        log_prob_key="action_log_prob",
    )
    critic = TensorDictModule(ValueModule(model), in_keys=["observation"], out_keys=["state_value"])
    loss = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=config.entropy_coeff,
        critic_coeff=0.5,
        loss_critic_type="l2",
        normalize_advantage=False,
        functional=False,
    )
    loss.set_keys(sample_log_prob="action_log_prob")
    return loss


def optimize_batch(model, loss_module, optimizer, batch, config):
    totals = {}
    minibatches = 0
    for _ in range(config.epochs):
        permutation = torch.randperm(len(batch))
        for indices in permutation.split(config.minibatch_size):
            losses = loss_module(batch[indices])
            objective = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
            if not torch.isfinite(objective):
                raise RuntimeError("PPO produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            # Gradient clipping and PPO probability-ratio clipping are different:
            # this one limits the gradient norm before the optimizer step.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True
            )
            optimizer.step()
            for key in ("loss_objective", "loss_critic", "entropy", "clip_fraction", "kl_approx"):
                totals[key] = totals.get(key, 0.0) + float(losses[key].detach())
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm)
            minibatches += 1
    return {key: value / minibatches for key, value in totals.items()}


def _summarize(counts):
    total = sum(counts.values())
    return {
        **counts,
        "games": total,
        "win_rate": counts["wins"] / total,
        "draw_rate": counts["draws"] / total,
        "loss_rate": counts["losses"] / total,
        "mean_return": (counts["wins"] - counts["losses"]) / total,
    }


@torch.no_grad()
def evaluate(model: ActorCritic | None, games_per_seat=1000, seed=30000, sample=False):
    """Frozen evaluation versus random. None means a random learner baseline.

    Separate seeded RNGs and fork_rng keep evaluation out of training's RNG stream.
    The default policy uses the most probable LEGAL action (greedy evaluation).
    """
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be positive")
    learner_rng = np.random.default_rng(seed + 2_000_000)
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    by_seat = {}
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for seat_index, seat in enumerate(("X", "O")):
                counts = {"wins": 0, "draws": 0, "losses": 0}
                env = FixedOpponentEnv()
                for episode in range(games_per_seat):
                    observation, _ = env.reset(
                        seed=seed + seat_index * 1_000_000 + episode,
                        options={"learner": seat},
                    )
                    while True:
                        if model is None:
                            action = int(
                                learner_rng.choice(np.flatnonzero(observation["action_mask"]))
                            )
                        else:
                            cells, mask = observation_tensors(observation)
                            distribution, _ = model.distribution(cells, mask)
                            action = int(distribution.sample() if sample else distribution.mode)
                        observation, reward, terminated, _, _ = env.step(action)
                        if terminated:
                            counts[{1.0: "wins", 0.0: "draws", -1.0: "losses"}[reward]] += 1
                            break
                env.close()
                by_seat[seat] = _summarize(counts)
    finally:
        if model is not None:
            model.train(was_training)
    overall = _summarize(
        {key: sum(by_seat[seat][key] for seat in ("X", "O")) for key in ("wins", "draws", "losses")}
    )
    return {
        "mode": "random" if model is None else ("sample" if sample else "greedy"),
        "seed": seed,
        "by_seat": by_seat,
        "overall": overall,
    }


def save_checkpoint(path: Path, model: ActorCritic, config: TrainConfig, *, metadata=None):
    torch.save(
        {
            "format_version": 1,
            "hidden_size": model.hidden_size,
            "model_state": model.state_dict(),
            "config": asdict(config),
            "torch_version": str(torch.__version__),
            "torchrl_version": str(torchrl.__version__),
            "training_metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    # Constructing layers consumes randomness. Keep loading separate from the
    # caller's RNG state just as evaluation is separate from training.
    with torch.random.fork_rng(devices=[]):
        model = ActorCritic(checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def print_evaluation(label, result):
    print(label, flush=True)
    for seat, numbers in result["by_seat"].items():
        print(
            f"  {seat}: {numbers['wins']} wins, {numbers['draws']} draws, "
            f"{numbers['losses']} losses / {numbers['games']} games "
            f"(return {numbers['mean_return']:+.3f})",
            flush=True,
        )


def plot_history(history, initial_evaluation, out):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blokus-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(
        [row["episodes"] for row in history],
        [row["training_return"] for row in history],
        label="Training batch",
        alpha=0.7,
    )
    evaluations = [row for row in history if row.get("eval_return") is not None]
    x = [0] + [row["episodes"] for row in evaluations]
    axes[0].plot(
        x,
        [initial_evaluation["overall"]["mean_return"]]
        + [row["eval_return"] for row in evaluations],
        label="Greedy evaluation",
        marker=".",
    )
    axes[0].set(title="Reward learned from game outcomes", ylabel="Mean return", ylim=(-1, 1))
    for metric, label in (("win_rate", "Wins"), ("draw_rate", "Draws"), ("loss_rate", "Losses")):
        axes[1].plot(
            x,
            [initial_evaluation["overall"][metric]]
            + [row[f"eval_{metric}"] for row in evaluations],
            label=label,
            marker=".",
        )
    axes[1].set(title="Held-out games against random", ylabel="Fraction of games", ylim=(0, 1))
    for axis in axes:
        axis.set_xlabel("Training episodes")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(out / "learning_curve.png", dpi=160)
    fig.savefig(out / "learning_curve.svg")
    plt.close(fig)


def train(config: TrainConfig, out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    configure_torch(config.seed)
    model = ActorCritic(config.hidden_size)
    # No dropout or batch normalization: eval mode preserves gradients and keeps
    # inference behavior identical during rollout collection and PPO optimization.
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_module = make_ppo_loss(model, config)
    env = FixedOpponentEnv()
    env.reset(seed=config.seed)
    start = time.monotonic()
    print(
        f"PyTorch {torch.__version__}; TorchRL {torchrl.__version__}; CPU; seed {config.seed}",
        flush=True,
    )
    print(
        f"Model: 9 -> {config.hidden_size} -> {config.hidden_size} -> (9 logits, 1 value); "
        f"{sum(p.numel() for p in model.parameters()):,} parameters",
        flush=True,
    )
    random_baseline = evaluate(None, config.final_games_per_seat, seed=30000)
    initial = evaluate(model, config.final_games_per_seat, seed=30000)
    print_evaluation("Random baseline:", random_baseline)
    print_evaluation("Untrained policy (greedy):", initial)
    history = []
    transitions = 0
    with (out / "metrics.csv").open("w", newline="") as file:
        writer = None
        for update in range(1, config.updates + 1):
            # We finish collecting before modifying weights: this is on-policy PPO.
            batch, training_return = collect_batch(model, env, config)
            transitions += len(batch)
            row = {
                "update": update,
                "episodes": update * config.episodes_per_update,
                "transitions": transitions,
                "training_return": training_return,
                **optimize_batch(model, loss_module, optimizer, batch, config),
                "eval_return": None,
                "eval_win_rate": None,
                "eval_draw_rate": None,
                "eval_loss_rate": None,
            }
            if update % config.eval_every == 0 or update == config.updates:
                result = evaluate(model, config.eval_games_per_seat, seed=20000)["overall"]
                row.update(
                    eval_return=result["mean_return"],
                    eval_win_rate=result["win_rate"],
                    eval_draw_rate=result["draw_rate"],
                    eval_loss_rate=result["loss_rate"],
                )
                print(
                    f"Update {update:3}/{config.updates}: {row['episodes']:6} episodes, "
                    f"train return {training_return:+.3f}, "
                    f"eval W/D/L {result['win_rate']:.1%}/{result['draw_rate']:.1%}/{result['loss_rate']:.1%}, "
                    f"entropy {row['entropy']:.3f}",
                    flush=True,
                )
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            history.append(row)
    env.close()
    final = evaluate(model, config.final_games_per_seat, seed=30000)
    print_evaluation("Trained policy (greedy, final evaluation):", final)
    save_checkpoint(out / "model.pt", model, config)
    summary = {
        "config": asdict(config),
        "versions": {"torch": str(torch.__version__), "torchrl": str(torchrl.__version__)},
        "random_baseline": random_baseline,
        "untrained_policy": initial,
        "trained_policy": final,
        "transitions": transitions,
        "training_and_evaluation_seconds": time.monotonic() - start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_history(history, initial, out)
    print(f"Saved checkpoint, metrics, summary, and learning curve to {out}", flush=True)
    return summary
