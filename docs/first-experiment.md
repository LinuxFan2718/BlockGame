# First PPO experiment

We trained a 5,450-parameter PyTorch MLP using TorchRL's clipped PPO loss, playing
against a uniformly random legal-move opponent. The policy learned from terminal
win/draw/loss rewards. No solved policy or tactical rules were supplied.

The saved local run is [`runs/ppo-demo`](../runs/ppo-demo). Generated runs are
ignored by Git; source code, package lockfiles, and this experiment record are
kept separately.

## Configuration

- Seed: 42.
- PyTorch 2.14.1; TorchRL 0.14.1; Python 3.12.3; CPU.
- Architecture: 9 -> 64 -> 64, Tanh activations, 9-logit actor and 1-value critic.
- 300 updates, 128 complete games per update: **38,400 training games**.
- 132,195 learner transitions, each including the opponent response when needed.
- Four PPO epochs per rollout; minibatch size 256; Adam learning rate 0.0005.
- Gamma 1; clipping epsilon 0.2; entropy coefficient 0.01; value-loss coefficient 0.5.
- Learner seats alternate between X and O. Evaluation uses greedy legal moves.

To repeat the experiment into a new directory:

```sh
uv run blokus-rl train --updates 300 --seed 42 --out runs/reproduce-ppo-demo
```

## Before and after training

These comparisons use the same final-evaluation seed set (seed 30000), with 1,000
games per seat. Evaluation preserves the training RNG state and model weights.

| Policy | Seat | Wins | Draws | Losses |
| --- | --- | ---: | ---: | ---: |
| Untrained neural policy | X | 664 | 101 | 235 |
| Trained neural policy | X | 977 | 8 | 15 |
| Untrained neural policy | O | 304 | 101 | 595 |
| Trained neural policy | O | 822 | 28 | 150 |

The combined win rate increased from **48.4% to 89.95%**, and the combined loss
rate fell from **41.5% to 8.25%**. The untrained neural policy's greedy choices
are a different baseline from a uniformly random policy. Random-versus-random
on these seeds produced 850 wins, 244 draws, and 906 losses across both seats.

## Fresh opponent seeds

We then loaded the saved checkpoint and evaluated it on another seed set, with
no further training:

```sh
uv run blokus-rl evaluate runs/ppo-demo/model.pt \
  --games-per-seat 1000 --seed 40000 \
  --json runs/ppo-demo/fresh_evaluation.json
```

| Seat | Wins | Draws | Losses | Win rate |
| --- | ---: | ---: | ---: | ---: |
| X | 973 | 6 | 21 | 97.3% |
| O | 844 | 31 | 125 | 84.4% |

That is **1,817 wins, 37 draws, and 146 losses in 2,000 games** against random.
The agent still loses, particularly as O. This demonstrates learning against
the training opponent distribution; it does not establish optimal tic-tac-toe
play or strength against a strong human. Stronger evaluation opponents and
self-play are natural next experiments.

The [learning curve](../runs/ppo-demo/learning_curve.png) shows sampled training
returns and periodic greedy evaluation. Periodic evaluations use 100 games per
seat and a different seed set (20000), so their final plotted point differs from
the larger final evaluation in the summary.

## Inspect and play

```sh
uv run blokus-rl inspect
uv run blokus-rl play runs/ppo-demo/model.pt
uv run blokus-rl play runs/ppo-demo/model.pt --human O
```

See [the training walkthrough](training.md) for how observations, returns,
advantages, PPO clipping, gradients, and the optimizer fit together.
