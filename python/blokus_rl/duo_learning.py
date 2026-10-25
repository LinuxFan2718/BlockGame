"""Complete-episode PPO for Blokus Duo using PyTorch and TorchRL.

Legal moves are ragged: each minibatch pads candidates, and TorchRL masks the
padding. A policy action is an index in that decision's saved candidate list.
The Rust action id is used only when applying the chosen move to the engine.
"""

import copy
import csv
import hashlib
import json
import math
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

from ._native import DuoGame
from .duo_env import CELLS, PASS, STATE_SIZE, decision
from .duo_model import DuoActorCritic, DuoPolicyModule, DuoValueModule
from .learning import configure_torch

BASELINES = ("random", "largest", "frontier")


@dataclass(frozen=True)
class DuoConfig:
    seed: int = 42
    updates: int = 100
    episodes_per_update: int = 16
    epochs: int = 3
    minibatch_size: int = 64
    hidden_size: int = 128
    learning_rate: float = 0.0003
    entropy_coeff: float = 0.02
    eval_every: int = 10
    eval_games_per_seat: int = 20
    final_games_per_seat: int = 100
    opponent: str = "random"
    reward: str = "win"
    validation_seed: int = 70000
    test_seed: int = 90000
    selection: str = "random"
    architecture: str = "additive"

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
        if (
            min(self.seed, self.validation_seed, self.test_seed) < 0
            or self.learning_rate <= 0
            or self.entropy_coeff < 0
        ):
            raise ValueError("invalid seed, learning rate or entropy coefficient")
        if self.opponent not in (
            "random",
            "league",
            "mixed",
            "largest",
            "frontier",
        ) or self.reward not in (
            "win",
            "margin",
        ):
            raise ValueError("invalid opponent or reward")
        if self.selection not in ("random", "balanced", "frontier"):
            raise ValueError("invalid selection criterion")
        if self.architecture not in ("additive", "conv"):
            raise ValueError("invalid model architecture")


def pack(records, indices=None):
    """Pad only the current minibatch. Padding is never a selectable move."""
    rows = records if indices is None else [records[int(i)] for i in indices]
    count = max(len(r["candidates"]) for r in rows)
    candidates = torch.full((len(rows), count, 6), CELLS, dtype=torch.long)
    candidates[..., 0] = 21
    mask = torch.zeros((len(rows), count), dtype=torch.bool)
    for i, r in enumerate(rows):
        n = len(r["candidates"])
        candidates[i, :n] = torch.from_numpy(r["candidates"].astype(np.int64))
        mask[i, :n] = True
    result = {
        "observation": torch.from_numpy(np.stack([r["observation"] for r in rows])),
        "candidates": candidates,
        "action_mask": mask,
    }
    for key in ("action", "action_log_prob", "value_target", "advantage"):
        if key in rows[0]:
            result[key] = torch.tensor(
                [r[key] for r in rows], dtype=(torch.long if key == "action" else torch.float32)
            )
            if key in ("value_target", "advantage"):
                result[key] = result[key].unsqueeze(-1)
    return TensorDict(result, batch_size=[len(rows)])


@torch.no_grad()
def choose(game, policy, rng, *, sample=False):
    """Uniform random, largest-piece baseline, or neural policy. No search/oracle."""
    if isinstance(policy, str):
        actions, candidates = game.candidates()
        if policy == "random":
            return int(rng.choice(actions))
        if policy in ("largest", "frontier"):
            areas = np.count_nonzero(np.asarray(candidates)[:, 1:] < CELLS, axis=1)
            choices = np.flatnonzero(areas == areas.max())
            if policy == "frontier" and len(choices) > 1:
                player = game.current_player
                scores = []
                for index in choices:
                    successor = game.copy()
                    successor.step(int(actions[int(index)]))
                    anchors = successor.frontier_counts
                    scores.append(anchors[player] - anchors[1 - player])
                choices = choices[np.asarray(scores) == max(scores)]
            return int(actions[int(rng.choice(choices))])
        raise ValueError(f"unknown baseline: {policy}")
    state, actions, candidates = decision(game)
    distribution, _ = policy.distribution(torch.from_numpy(state), torch.from_numpy(candidates))
    index = int(distribution.sample() if sample else distribution.mode)
    return int(actions[index])


def wilson(wins, games):
    # Score rate treats a draw as half a win; the binomial interval below is for
    # the STRICT win rate only. Report draw counts separately.
    z = 1.96
    p = wins / games
    center = (p + z * z / (2 * games)) / (1 + z * z / games)
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / (1 + z * z / games)
    return [center - half, center + half]


@torch.no_grad()
def evaluate_duo(policy, opponent="random", games_per_seat=100, seed=90000):
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be positive")
    seats = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for seat in (0, 1):
            wins = draws = losses = 0
            margins, own_scores = [], []
            for episode in range(games_per_seat):
                # Independent streams prevent the policy's own random choices
                # from altering the opponent's random number sequence.
                opponent_rng = np.random.default_rng(seed + seat * 1_000_000 + episode)
                learner_rng = np.random.default_rng(seed + seat * 1_000_000 + episode + 2_000_000)
                game = DuoGame()
                while not game.terminated:
                    player = game.current_player
                    action = choose(
                        game,
                        policy if player == seat else opponent,
                        learner_rng if player == seat else opponent_rng,
                        sample=player != seat and not isinstance(opponent, str),
                    )
                    game.step(action)
                score = game.scores
                margin = score[seat] - score[1 - seat]
                wins += margin > 0
                draws += margin == 0
                losses += margin < 0
                margins.append(margin)
                own_scores.append(score[seat])
            seats[str(seat)] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "games": games_per_seat,
                "score_rate": (wins + draws / 2) / games_per_seat,
                "win_rate": wins / games_per_seat,
                "mean_margin": float(np.mean(margins)),
                "mean_score": float(np.mean(own_scores)),
                "win_rate_95pct_wilson": wilson(wins, games_per_seat),
            }
    wins = sum(s["wins"] for s in seats.values())
    draws = sum(s["draws"] for s in seats.values())
    return {
        "seed": seed,
        "opponent": opponent if isinstance(opponent, str) else "frozen_neural",
        "opponent_mode": "baseline" if isinstance(opponent, str) else "sample",
        "learner_mode": "greedy" if not isinstance(policy, str) else "baseline",
        "by_seat": seats,
        "games": 2 * games_per_seat,
        "wins": wins,
        "draws": draws,
        "losses": 2 * games_per_seat - wins - draws,
        "win_rate": wins / (2 * games_per_seat),
        "score_rate": (wins + draws / 2) / (2 * games_per_seat),
        "mean_margin": sum(s["mean_margin"] for s in seats.values()) / 2,
        "win_rate_95pct_wilson": wilson(wins, 2 * games_per_seat),
    }


@torch.no_grad()
def collect_duo(model, config, rng, snapshots, *, opponent_selector=None, sample_opponent=True):
    records, results = [], []
    for episode in range(config.episodes_per_update):
        seat = episode % 2
        # Every opponent is fixed throughout a game and throughout collection.
        # Forty percent random anchors the league to a stationary task.
        opponent = "random"
        if opponent_selector is not None:
            opponent = opponent_selector(rng)
        elif config.opponent in ("largest", "frontier"):
            opponent = config.opponent
        elif config.opponent == "league" and snapshots and rng.random() >= 0.4:
            opponent = snapshots[int(rng.integers(len(snapshots)))]
        elif config.opponent == "mixed":
            draw = rng.random()
            if draw >= 0.5 and snapshots:
                opponent = snapshots[int(rng.integers(len(snapshots)))]
            elif draw >= 0.25:
                opponent = "largest"
        game = DuoGame()
        episode_records = []
        while not game.terminated:
            if game.current_player != seat:
                game.step(choose(game, opponent, rng, sample=sample_opponent))
                continue
            state, actions, candidates = decision(game)
            distribution, value = model.distribution(
                torch.from_numpy(state), torch.from_numpy(candidates)
            )
            action = distribution.sample()
            episode_records.append(
                {
                    "observation": state,
                    "candidates": candidates,
                    "action": int(action),
                    "action_log_prob": float(distribution.log_prob(action)),
                    "value": value.item(),
                }
            )
            game.step(int(actions[int(action)]))
        score = game.scores
        margin = score[seat] - score[1 - seat]
        target = float(np.sign(margin)) if config.reward == "win" else margin / 109
        for r in episode_records:
            r["value_target"] = target
            r["advantage"] = target - r.pop("value")
        records.extend(episode_records)
        results.append(float(np.sign(margin)))
    advantages = np.array([r["advantage"] for r in records])
    for r in records:
        r["advantage"] = (r["advantage"] - advantages.mean()) / max(advantages.std(), 1e-8)
    return records, float(np.mean(results))


def make_duo_loss(model, config):
    actor = ProbabilisticActor(
        module=TensorDictModule(
            DuoPolicyModule(model), in_keys=["observation", "candidates"], out_keys=["logits"]
        ),
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
        log_prob_key="action_log_prob",
    )
    critic = TensorDictModule(
        DuoValueModule(model), in_keys=["observation"], out_keys=["state_value"]
    )
    loss = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=config.entropy_coeff,
        critic_coeff=0.5,
        loss_critic_type="l2",
        normalize_advantage=False,
        functional=False,
    )
    loss.set_keys(sample_log_prob="action_log_prob")
    return loss


def optimize_duo(model, loss_module, optimizer, records, config):
    totals, count = {}, 0
    for _ in range(config.epochs):
        for indices in torch.randperm(len(records)).split(config.minibatch_size):
            losses = loss_module(pack(records, indices))
            objective = sum(losses[k] for k in ("loss_objective", "loss_critic", "loss_entropy"))
            if not torch.isfinite(objective):
                raise RuntimeError("nonfinite Blokus PPO loss")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            for key in ("loss_objective", "loss_critic", "entropy", "clip_fraction", "kl_approx"):
                totals[key] = totals.get(key, 0.0) + float(losses[key].detach())
            count += 1
    return {k: v / count for k, v in totals.items()}


def save_duo(path, model, config, update, *, parent=None, metadata=None):
    # Save to a neighboring file and rename so a wake-up never reads half a checkpoint.
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "game": "blokus_duo",
            "format_version": 3,
            "architecture": getattr(model, "architecture", "additive"),
            "state_size": model.state_size,
            "observation_version": 2,
            "hidden_size": model.hidden_size,
            "model_state": model.state_dict(),
            "config": asdict(config),
            "update": update,
            "parent_checkpoint": parent,
            "training_metadata": metadata or {},
            "torch_version": str(torch.__version__),
            "torchrl_version": str(torchrl.__version__),
        },
        temporary,
    )
    temporary.replace(path)


def load_duo(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("game") != "blokus_duo" or checkpoint.get("format_version") not in (1, 2, 3):
        raise ValueError("not a supported Blokus Duo checkpoint")
    from .duo_conv import DuoConvActorCritic

    architecture = checkpoint.get("architecture", "additive")
    if architecture not in ("additive", "conv"):
        raise ValueError("unsupported model architecture")
    with torch.random.fork_rng(devices=[]):
        model = (DuoConvActorCritic if architecture == "conv" else DuoActorCritic)(
            checkpoint["hidden_size"]
        )
    weights = dict(checkpoint["model_state"])
    first = weights.get("trunk.0.weight")
    if first is not None and first.shape[1] < STATE_SIZE:
        # Version 1 omitted completion scores/pass count. Zero new input weights
        # preserve its policy while allowing subsequent training to learn them.
        weights["trunk.0.weight"] = torch.nn.functional.pad(first, (0, STATE_SIZE - first.shape[1]))
    model.load_state_dict(weights)
    return model.eval()


def plot_duo(rows, out):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blokus-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for opponent in ("random", "largest", "frontier"):
        selected = [r for r in rows if f"{opponent}_score_rate" in r]
        if not selected:
            continue
        ax.plot(
            [r["games"] for r in selected],
            [r[f"{opponent}_score_rate"] for r in selected],
            "o-",
            label=opponent,
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Training games",
        ylabel="Validation score rate (win + ½ draw)",
        ylim=(0, 1),
        title="Blokus Duo: greedy policy, equal games in both seats",
    )
    ax.legend(title="Opponent")
    fig.savefig(out / "learning_curve.png", dpi=160)
    fig.savefig(out / "learning_curve.svg")
    plt.close(fig)


def train_duo(config, out, initial=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    configure_torch(config.seed)
    from .duo_conv import DuoConvActorCritic

    factory = DuoConvActorCritic if config.architecture == "conv" else DuoActorCritic
    model = load_duo(initial) if initial else factory(config.hidden_size)
    if getattr(model, "architecture", "additive") != config.architecture:
        raise ValueError("architecture must match the initial checkpoint")
    if model.hidden_size != config.hidden_size:
        raise ValueError("hidden size must match initial checkpoint")
    parent = (
        None
        if initial is None
        else {
            "path": str(Path(initial).resolve()),
            "sha256": hashlib.sha256(Path(initial).read_bytes()).hexdigest(),
        }
    )
    # --init is a warm start with new optimizer/RNG, not exact training resumption.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_module = make_duo_loss(model, config)
    rng = np.random.default_rng(config.seed)
    snapshots = [copy.deepcopy(model).eval().requires_grad_(False)]
    save_duo(out / "initial.pt", model, config, 0, parent=parent)
    protocol = {
        "config": asdict(config),
        "parent": parent,
        "validation_seed": config.validation_seed,
        "held_out_test_seed": config.test_seed,
        "selection": config.selection,
        "reward": "terminal win/draw/loss"
        if config.reward == "win"
        else "terminal score margin / 109",
        "parameters": sum(p.numel() for p in model.parameters()),
        "observation_size": STATE_SIZE,
        "architecture": getattr(model, "architecture", "additive"),
        "notes": "No human moves, heuristic targets, minimax or search used in training.",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("duo_env.py"),
                Path(__file__).with_name("duo_model.py"),
                Path(__file__).with_name("duo_conv.py"),
                Path(__file__).parents[2] / "src" / "duo.rs",
            )
            if path.exists()
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start = time.monotonic()
    rows, best_score, best_update = [], (-1, -1), 0
    baselines = (
        ("random", "largest", "frontier")
        if config.selection == "frontier"
        else ("random", "largest")
    )
    for update in range(config.updates + 1):
        row = {"update": update, "games": update * config.episodes_per_update}
        if update:
            records, reward = collect_duo(model, config, rng, snapshots)
            row.update(optimize_duo(model, loss_module, optimizer, records, config))
            row.update(training_return=reward, transitions=len(records))
            if config.opponent in ("league", "mixed") and update % 10 == 0:
                snapshots.append(copy.deepcopy(model).eval().requires_grad_(False))
                snapshots = snapshots[-16:]
        if update % config.eval_every == 0 or update == config.updates:
            evaluations = {}
            for opponent in baselines:
                result = evaluate_duo(
                    model, opponent, config.eval_games_per_seat, config.validation_seed
                )
                evaluations[opponent] = result
                row[f"{opponent}_score_rate"] = result["score_rate"]
                row[f"{opponent}_margin"] = result["mean_margin"]
            score = (row["random_score_rate"], row["largest_score_rate"])
            if config.selection == "balanced":
                score = (sum(score) / 2, row["largest_score_rate"])
            elif config.selection == "frontier":
                score = (
                    (row["largest_score_rate"] + row["frontier_score_rate"]) / 2,
                    row["frontier_score_rate"],
                )
            if score > best_score:
                best_score, best_update = score, update
                save_duo(out / "best.pt", model, config, update, parent=parent)
                (out / "best-validation.json").write_text(json.dumps(evaluations, indent=2) + "\n")
            print(
                f"update {update:4d} games {row['games']:6d} "
                f"vs random {row['random_score_rate']:.1%} "
                f"vs largest {row['largest_score_rate']:.1%} "
                + (
                    f"vs frontier {row['frontier_score_rate']:.1%} "
                    if "frontier_score_rate" in row
                    else ""
                )
                + f"elapsed {time.monotonic() - start:.1f}s",
                flush=True,
            )
        row["elapsed_seconds"] = time.monotonic() - start
        rows.append(row)
        # Append-only JSONL allows inspection while a run is still active.
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        save_duo(out / "latest.pt", model, config, update, parent=parent)
    save_duo(out / "final.pt", model, config, config.updates, parent=parent)
    best = load_duo(out / "best.pt")
    initial_model = load_duo(out / "initial.pt")
    held_out = {}
    for label, policy in (("initial", initial_model), ("selected", best)):
        held_out[label] = {
            opponent: evaluate_duo(policy, opponent, config.final_games_per_seat, config.test_seed)
            for opponent in baselines
        }
    held_out["selected_vs_initial"] = evaluate_duo(
        best, initial_model, config.final_games_per_seat, config.test_seed
    )
    summary = {
        "config": asdict(config),
        "best_update": best_update,
        "training_games": config.updates * config.episodes_per_update,
        "elapsed_seconds": time.monotonic() - start,
        "held_out": held_out,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    plot_duo(rows, out)
    print(f"Finished; selected update {best_update}. Results: {out / 'summary.json'}", flush=True)
    return summary


def play_duo(policy, human=0, seed=42):
    """Human terminal play. List candidates by piece; enter a stable engine move id."""
    rng = np.random.default_rng(seed)
    game = DuoGame()
    print("You are " + ("X" if human == 0 else "O") + ". q quits; ? lists pieces.")
    print("Enter a piece number 0..20 to list legal placements; then m MOVE_ID to play.")
    from ._native import duo_pieces

    pieces = duo_pieces()
    while not game.terminated:
        print(game.render())
        if game.current_player != human:
            action = choose(game, policy, rng)
            print(f"Bot plays {action}" + (" (forced pass)" if action == PASS else ""))
            game.step(action)
            continue
        actions = game.legal_actions()
        if actions == [PASS]:
            print("You have no placement; passing.")
            game.step(PASS)
            continue
        try:
            entry = input("Your move: ").strip()
        except EOFError:
            return
        if entry.lower() == "q":
            return
        if entry == "?":
            for i, shape in enumerate(pieces):
                if not game.remaining(human)[i]:
                    continue
                print(f"Piece {i}, area {len(shape)}:")
                for r in range(1 + max(p[0] for p in shape)):
                    print(
                        "  "
                        + "".join(
                            "##" if (r, c) in shape else ".."
                            for c in range(1 + max(p[1] for p in shape))
                        )
                    )
            continue
        try:
            if entry.startswith("m "):
                game.step(int(entry[2:]))
            else:
                piece = int(entry)
                for action in actions:
                    p, r, c, cells = game.describe(action)
                    if p == piece:
                        squares = " ".join(f"{chr(65 + s % 14)}{1 + s // 14}" for s in cells)
                        print(f"  m {action}: {squares}")
        except ValueError as error:
            print(error)
    print(game.render())
    print(
        "Draw"
        if game.scores[0] == game.scores[1]
        else ("X wins" if game.scores[0] > game.scores[1] else "O wins")
    )
