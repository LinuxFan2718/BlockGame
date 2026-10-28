"""Run the PyTorch lesson, train PPO, evaluate, or play a saved policy."""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from ._native import Game
from .audit import NeuralPolicy, audit_policy, print_audit
from .experiments import ARMS, study
from .learning import (
    TrainConfig,
    configure_torch,
    evaluate,
    load_checkpoint,
    print_evaluation,
    train,
)
from .model import ActorCritic
from .search import SearchPolicy


def inspect_model(seed: int):
    configure_torch(seed)
    game = Game()
    for action in [0, 4, 8, 1]:
        game.step(action)
    cells, mask, _ = game.observe("X")
    observation = torch.tensor(cells, dtype=torch.float32)
    action_mask = torch.tensor(mask, dtype=torch.bool)
    model = ActorCritic()
    distribution, value = model.distribution(observation, action_mask)
    action = distribution.mode.detach()
    before = float(distribution.probs[action].detach())
    print("1. The Rust engine supplies this board (X to move):")
    print(game.render())
    print("\n2. A plain PyTorch nn.Module supplies the predictions:")
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Input tensor (+1 mine, -1 theirs):", observation.tolist())
    print("Move probabilities (occupied squares have zero probability):")
    for index, probability in enumerate(distribution.probs.detach().tolist()):
        print(f"  square {index + 1}: {probability:.4f}")
    print(f"Untrained value estimate: {value.item():+.4f}")

    # A tiny synthetic exercise to expose autograd. These are made-up targets,
    # not feedback from the game. The actual PPO trainer uses recorded outcomes.
    print("\n3. One illustrative PyTorch update, using a made-up positive advantage +1")
    print("   and value target +1. This is an autograd lesson, not PPO training.")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = -distribution.log_prob(action) + 0.5 * (value.squeeze() - 1.0).square()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name, layer in (
        ("shared trunk", model.trunk),
        ("policy head", model.policy_head),
        ("value head", model.value_head),
    ):
        norm = (
            sum(
                float(parameter.grad.square().sum())
                for parameter in layer.parameters()
                if parameter.grad is not None
            )
            ** 0.5
        )
        print(f"  {name} gradient norm: {norm:.6f}")
    optimizer.step()
    with torch.no_grad():
        after, _ = model.distribution(observation, action_mask)
    print(
        f"Chosen square {int(action) + 1}: probability {before:.4f} -> {after.probs[action].item():.4f}"
    )
    print("No checkpoint was written; each inspect run starts with fresh seeded weights.")


def play(model: ActorCritic, human: str, seed: int, search_simulations=0):
    configure_torch(seed)
    bot_mark = "O" if human == "X" else "X"
    game = Game()
    search = SearchPolicy(NeuralPolicy(model), search_simulations) if search_simulations else None
    print(f"You are {human}; the trained bot is {bot_mark}. Enter 1-9 or q.")
    print(game.render())
    while not game.terminated:
        if game.current_player == human:
            try:
                entry = input(f"{human}, your move: ").strip()
            except EOFError:
                print("\nStopped.")
                return
            if entry.lower() == "q":
                return
            try:
                square = int(entry)
                if square not in range(1, 10):
                    raise ValueError("enter a square from 1 to 9")
                game.step(square - 1)
            except ValueError as error:
                print(error)
                continue
        else:
            if search:
                action = search.action(tuple(game.observe("X")[0]))
            else:
                cells, mask, _ = game.observe(bot_mark)
                with torch.no_grad():
                    distribution, _ = model.distribution(
                        torch.tensor(cells, dtype=torch.float32),
                        torch.tensor(mask, dtype=torch.bool),
                    )
                    action = int(distribution.mode)
            print(f"{bot_mark} (trained) chooses {action + 1}.")
            game.step(action)
        print(game.render())
    print("Draw." if game.outcome == "draw" else f"{game.outcome} wins.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="See a PyTorch forward pass and gradients")
    inspect_parser.add_argument("--seed", type=int, default=42)
    train_parser = commands.add_parser(
        "train", help="Train a neural policy with TorchRL's PPO loss"
    )
    train_parser.add_argument("--out", type=Path)
    for name in (
        "seed",
        "updates",
        "episodes_per_update",
        "epochs",
        "minibatch_size",
        "hidden_size",
        "eval_every",
        "eval_games_per_seat",
        "final_games_per_seat",
    ):
        train_parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(TrainConfig(), name)
        )
    train_parser.add_argument("--learning-rate", type=float, default=TrainConfig().learning_rate)
    eval_parser = commands.add_parser("evaluate", help="Evaluate a checkpoint against a random bot")
    eval_parser.add_argument("checkpoint", type=Path)
    eval_parser.add_argument("--games-per-seat", type=int, default=1000)
    eval_parser.add_argument("--seed", type=int, default=40000)
    eval_parser.add_argument(
        "--sample", action="store_true", help="Sample moves instead of choosing the mode"
    )
    eval_parser.add_argument("--json", type=Path, help="Also save evaluation results")
    play_parser = commands.add_parser("play", help="Play a trained policy using the Rust rules")
    play_parser.add_argument("checkpoint", type=Path)
    play_parser.add_argument("--human", choices=("X", "O"), default="X")
    play_parser.add_argument("--seed", type=int, default=42)
    play_parser.add_argument("--search-simulations", type=int, default=0)
    audit_parser = commands.add_parser(
        "audit", help="Check all opponent responses and perfect play"
    )
    audit_parser.add_argument("checkpoint", type=Path)
    audit_parser.add_argument("--json", type=Path)
    audit_parser.add_argument("--search-simulations", type=int, default=0)
    experiment_parser = commands.add_parser(
        "experiment", help="Run the declared self-play comparison"
    )
    experiment_parser.add_argument("--out", type=Path, required=True)
    experiment_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    experiment_parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    experiment_parser.add_argument("--updates", type=int, default=300)
    experiment_parser.add_argument("--episodes-per-update", type=int, default=128)
    experiment_parser.add_argument("--search-simulations", type=int, default=32)
    experiment_parser.add_argument("--audit-every", type=int, default=25)
    experiment_parser.add_argument("--workers", type=int, default=1)
    duo_train = commands.add_parser("duo-train", help="Train Blokus Duo with TorchRL PPO")
    from .duo_learning import DuoConfig

    duo_train.add_argument("--out", type=Path, required=True)
    duo_train.add_argument("--init", type=Path, help="Warm start weights; new optimizer and RNG")
    for field, default in vars(DuoConfig()).items():
        duo_train.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
    duo_eval = commands.add_parser("duo-evaluate", help="Evaluate a Blokus Duo bot in both seats")
    duo_eval.add_argument("checkpoint", help="Checkpoint path, random, largest, or frontier")
    duo_eval.add_argument(
        "--opponent", default="random", help="random, largest, frontier, or checkpoint path"
    )
    duo_eval.add_argument("--games-per-seat", type=int, default=100)
    duo_eval.add_argument("--seed", type=int, default=90000)
    duo_eval.add_argument("--json", type=Path)
    duo_eval.add_argument(
        "--simulations", type=int, default=0, help="PUCT simulations per move; 0 plays raw policy"
    )
    duo_play = commands.add_parser("duo-play", help="Play Blokus Duo in the terminal")
    duo_play.add_argument("checkpoint", help="Checkpoint path, random, largest, or frontier")
    duo_play.add_argument("--human", type=int, choices=(0, 1), default=0)
    duo_play.add_argument("--seed", type=int, default=42)
    duo_play.add_argument(
        "--simulations", type=int, default=0, help="PUCT simulations per bot move"
    )
    duo_play.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    duo_eval.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    duo_web = commands.add_parser("duo-web", help="Play a local browser game against a saved bot")
    duo_web.add_argument("checkpoint", nargs="?", type=Path)
    duo_web.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    duo_web.add_argument("--port", type=int, default=8765)
    duo_web.add_argument("--no-browser", action="store_true")
    commands.add_parser("duo-report", help="Rebuild the saved Blokus experiment report")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            inspect_model(args.seed)
        elif args.command == "train":
            config = replace(
                TrainConfig(),
                **{
                    field: getattr(args, field)
                    for field in vars(TrainConfig())
                    if hasattr(args, field)
                },
            )
            out = args.out or Path("runs") / datetime.now().strftime("ppo-%Y%m%d-%H%M%S")
            train(config, out)
        elif args.command == "evaluate":
            configure_torch(args.seed)
            result = evaluate(
                load_checkpoint(args.checkpoint), args.games_per_seat, args.seed, args.sample
            )
            print_evaluation("Saved policy versus random:", result)
            if args.json:
                args.json.write_text(json.dumps(result, indent=2) + "\n")
        elif args.command == "play":
            play(load_checkpoint(args.checkpoint), args.human, args.seed, args.search_simulations)
        elif args.command == "audit":
            configure_torch(42)
            policy = NeuralPolicy(load_checkpoint(args.checkpoint))
            if args.search_simulations:
                policy = SearchPolicy(policy, args.search_simulations)
            result = audit_policy(policy)
            print_audit(result)
            if args.json:
                args.json.write_text(json.dumps(result, indent=2) + "\n")
        elif args.command == "experiment":
            study(
                args.out,
                args.seeds,
                args.updates,
                args.episodes_per_update,
                args.search_simulations,
                args.arms,
                args.audit_every,
                args.workers,
            )
        elif args.command == "duo-train":
            from .duo_learning import train_duo

            config = DuoConfig(**{field: getattr(args, field) for field in vars(DuoConfig())})
            train_duo(config, args.out, args.init)
        elif args.command == "duo-web":
            from .duo_web import serve

            configure_torch(42)
            serve(args.checkpoint, args.device, args.port, not args.no_browser)
        elif args.command in ("duo-evaluate", "duo-play"):
            from .duo_learning import BASELINES, evaluate_duo, load_duo, play_duo
            from .duo_device import inference_device

            configure_torch(args.seed)
            policy = args.checkpoint if args.checkpoint in BASELINES else load_duo(args.checkpoint)
            device = inference_device(args.device)
            if not isinstance(policy, str):
                policy.to(device)
            if args.command == "duo-play":
                print(f"Inference: {device} (Metal GPU)" if device.type == "mps" else f"Inference: {device}")
            if args.simulations < 0:
                raise ValueError("simulations must be nonnegative")
            if args.simulations:
                if isinstance(policy, str):
                    raise ValueError("neural search requires a checkpoint")
                from .duo_mcts import MctsPolicy

                policy = MctsPolicy(policy, args.simulations)
            if args.command == "duo-play":
                play_duo(policy, args.human, args.seed)
            else:
                opponent = args.opponent if args.opponent in BASELINES else load_duo(args.opponent)
                if not isinstance(opponent, str):
                    opponent.to(device)
                result = evaluate_duo(policy, opponent, args.games_per_seat, args.seed)
                print(json.dumps(result, indent=2))
                if args.json:
                    args.json.write_text(json.dumps(result, indent=2) + "\n")
        elif args.command == "duo-report":
            from .duo_report import build_report

            build_report()
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
