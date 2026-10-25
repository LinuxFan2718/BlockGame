# Learning PyTorch and PPO with our Rust game

We now have a complete first experiment: a small PyTorch model learns from
tic-tac-toe outcomes against a random opponent, using TorchRL's clipped PPO loss.
The game is still implemented by the original Rust engine. Python supplies
environment adapters, the model, training, evaluation, and human-versus-model play.

The [first experiment record](first-experiment.md) contains the completed demo's
configuration, before/after results, and a fresh-seed evaluation. Its saved model
is available locally at `runs/ppo-demo/model.pt`.

## Start with the forward pass and gradients

From the project root:

```sh
# Creates .venv and builds the native Rust extension. Uses the exact uv.lock versions.
uv sync --locked

# A small, repeatable PyTorch lesson. No training checkpoint is needed.
uv run blokus-rl inspect
```

The project selects Python 3.12 through `.python-version`; the Rust tools must be
installed to build the binding. The training dependencies are isolated in `.venv`.
The Rust-only `cargo run` and `cargo test` commands still work independently.

`inspect` prints a board, the network, its input tensor, masked move probabilities,
and a value prediction. It then supplies a **made-up** positive advantage and
value target to demonstrate `loss.backward()` and `optimizer.step()`. It prints
gradient norms and the before/after probability of the selected move. This
exercise is deliberately separate from the real game-outcome-based PPO trainer.

Read these files in this order:

1. [`model.py`](../python/blokus_rl/model.py): a plain `torch.nn.Module`.
2. [`cli.py`, `inspect_model`](../python/blokus_rl/cli.py): tensors, a forward pass,
   a loss, gradients, and an optimizer step.
3. [`env.py`](../python/blokus_rl/env.py): observations and whose turn a step represents.
4. [`learning.py`](../python/blokus_rl/learning.py): collection, returns, PPO updates,
   evaluation, and saving the learned weights.

## The model: what we are learning

```text
9 board numbers: +1 = mine, -1 = theirs, 0 = empty
                  |
           Linear(9, 64) + Tanh
                  |
          Linear(64, 64) + Tanh
                  |
         shared learned features
            /             \
   Linear(64, 9)       Linear(64, 1)
   actor: logits       critic: value
          |
 legal-action mask
          |
  move probabilities
```

There are 5,450 trainable parameters, initialized from the chosen seed. No winning
patterns, blocking rules, expert moves, or solved-game tables are supplied to the
model. The engine supplies legality and terminal outcomes. The training objective
and collected experience determine how the weights change.

A **tensor** is a multidimensional array of numbers. Here a single input is a
float tensor with shape `[9]`; a batch of inputs has shape `[batch_size, 9]`.
`nn.Linear` learns a weight matrix and bias. `Tanh` supplies a nonlinearity: without
nonlinearities, a stack of linear layers would still express one linear mapping.

The actor's nine raw scores are called **logits**, just as a language model
produces logits over tokens. Masked softmax gives occupied squares zero
probability and renormalizes over the remaining legal actions. A move is sampled
from this distribution during training. Greedy evaluation chooses its mode.
The critic estimates the expected future reward under the current policy and
the training opponent. It is not a guaranteed assessment under optimal play.

The two heads share the hidden layers. Both policy and value losses can update
those shared features. The optimizer receives `model.parameters()` once, so
shared weights are not accidentally stepped twice.

This is a small **multilayer perceptron (MLP)** with no recurrent memory. The
current board, viewed on the player's turn, contains all decision-relevant state.
Feedforward evaluation and backward gradient computation are compatible:
`forward` produces predictions; autograd computes how a loss changes with weights.

## The environment: two meanings of step

The binding in [`bindings/python`](../bindings/python/src/lib.rs) exposes the Rust
`Game`. It calls `Game::step`; it does not recreate the rules in Python.

`TicTacToeEnv` implements PettingZoo's AEC interface:

- One step places one mark.
- X and O have separate rewards and terminal flags.
- Only the acting player's action mask has legal entries.
- When the game ends, both players receive terminal feedback. PettingZoo then
  drains terminal agents with `step(None)`.

The tests verify this adapter against PettingZoo's API and seed tests and against
TorchRL's `PettingZooWrapper`. This is our interface for future self-play and
multiplayer games.

The first PPO experiment uses `FixedOpponentEnv`, a Gymnasium adapter on top:

```text
learner observes -> learner moves -> opponent replies -> learner observes
```

One **training transition** covers a learner move plus an opponent reply. If
either move ends the game, it stops there. When the learner plays O, the random X
makes an opening move before the first observation is returned. Seats alternate
between episodes, so the same policy learns both roles. An explicit `learner`
option can choose a seat for evaluation.

This keeps consecutive value estimates in the same player's perspective. If X
moves and then O wins, the transition returns **-1 to X**, even though X did not
make the last move. Terminal observations have an all-false legal mask and are
never fed into an action distribution.

The training opponent samples uniformly from legal moves using a seeded NumPy
generator. The original Rust CLI's random bot uses SplitMix64. Both are random
policies; their seeded move sequences are different. No Python training action
can bypass the Rust engine's validation.

## PPO: what happens during one update

The implementation makes the learning loop explicit:

1. Freeze the current weights while collecting a batch of complete games.
2. Record observations, masks, sampled actions, their old log probabilities, and
   the critic's old value estimates.
3. Compute each decision's return by working backward within its episode.
4. Calculate advantage = return - old value estimate, then normalize advantages
   over the collected batch.
5. Shuffle the data into minibatches and run a few epochs of PPO optimization.
6. Discard that batch, collect fresh experience with the updated policy, and repeat.

This is **on-policy** learning: the data comes from the policy being improved.
Reusing it for a few PPO epochs is intentional; retaining an unlimited collection
of old-policy games would require different handling.

Our rewards are zero until the game ends: win +1, draw 0, loss -1. With the default
discount factor gamma = 1, every decision's Monte Carlo return is its player's
final result. With gamma below 1, more distant rewards are discounted. Returns
are computed separately for each episode; losses never leak across game resets.

We start with **complete-episode Monte Carlo returns** so there is no need to
bootstrap values at rollout cutoffs. PPO also supports other advantage estimators,
including GAE; introducing that estimator can be a later, separate lesson.

For an action selected during collection, define:

```text
ratio = new_policy_probability / recorded_old_policy_probability
      = exp(new_log_probability - recorded_old_log_probability)

policy_loss = -mean(min(ratio * advantage,
                        clamp(ratio, 0.8, 1.2) * advantage))
```

The 0.8 and 1.2 bounds come from the default clipping epsilon of 0.2. A positive
advantage encourages that action; a negative advantage discourages it. Clipping
removes some incentives for excessive changes. It is not a hard bound on every
probability change. A test compares TorchRL's result with hand-calculated cases.

The total loss combines:

- **Policy loss:** improve the decision distribution using the clipped objective.
- **Value loss:** make the critic's estimates closer to the observed returns.
- **Entropy bonus:** encourage exploration by rewarding less concentrated policies.

TorchRL's `ClipPPOLoss` computes those losses. PyTorch's `backward()` computes
gradients, and Adam adjusts the network weights. We also clip the **gradient
norm**, which is a separate operation from PPO's probability-ratio clipping.

## Train, evaluate, and play

```sh
# A new run directory is required; existing experiments are preserved.
uv run blokus-rl train --out runs/my-ppo-run

# A small end-to-end check, not enough experience to expect a strong policy.
uv run blokus-rl train --updates 2 --episodes-per-update 8 \
  --epochs 1 --eval-every 1 --eval-games-per-seat 5 \
  --final-games-per-seat 10 --out runs/smoke-run

# A shorter experiment using the same architecture and objective.
uv run blokus-rl train --updates 100 --out runs/shorter-ppo-run

# Fresh opponent seeds, 1,000 games as each mark, with fixed weights.
uv run blokus-rl evaluate runs/my-ppo-run/model.pt --games-per-seat 1000 --seed 40000

# Evaluate the stochastic policy, rather than always choosing its highest-probability move.
uv run blokus-rl evaluate runs/my-ppo-run/model.pt --sample --seed 40000

# Play either mark against a saved trained policy.
uv run blokus-rl play runs/my-ppo-run/model.pt
uv run blokus-rl play runs/my-ppo-run/model.pt --human O
```

Runs use the CPU and one PyTorch thread; the first network is small enough to
learn on this machine without GPU setup. Defaults are visible in `TrainConfig`
and `blokus-rl train --help`. Each update collects 128 games and optimizes for four
epochs. The default 300 updates collect 38,400 training games; 100 updates
collect 12,800 games.

Each run saves:

| File | What to inspect |
| --- | --- |
| `model.pt` | Learned weights, architecture size, training configuration, library versions |
| `metrics.csv` | Episodes, learner transitions, losses, entropy, clipping, evaluation metrics |
| `summary.json` | Random baseline, untrained policy, trained policy, per-seat results, configuration |
| `learning_curve.png`, `.svg` | Training returns and frozen-policy evaluation over time |

An **episode** is a complete game. A **transition** here is one learner decision
and its consequences. An **update** collects a batch and then trains on it. An
**epoch** makes one pass over that collected batch. A **minibatch** is the subset
used by one optimizer step. These are different counts in the training output.

The trainer evaluates random-versus-random, the initial neural policy, and the
final policy, with 1,000 games per seat by default. Smaller periodic evaluations
use a different seed set. Evaluation restores the training RNG state, preserves
model weights, and records X and O separately. The reported default policy is
greedy, while training samples moves; those results can differ.

The plot's training curve comes from sampled training games. Its evaluation curve
uses frozen weights and greedy moves against a separate seeded random opponent.
The initial point and periodic evaluations use different seed sets, as recorded
in `learning.py`; the final summary compares initial and final policies on the
same final-evaluation seeds. Changing the evaluation seed gives another sample.

Falling losses or entropy alone do not demonstrate better play. Inspect wins,
draws, and losses. These first evaluations measure performance against **random
opponents**; testing against stronger opponents is a separate experiment.

## Verification and extending the experiment

```sh
uv run pytest
uv run ruff check python
uv run ruff format --check python
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo clippy --manifest-path bindings/python/Cargo.toml -- -D warnings
```

The Python tests cover native error handling, observation perspectives, both
players' terminal rewards, opponent-opening moves, seeded environments, action
masks, Monte Carlo returns, a hand-calculated PPO ratio, gradient updates to both
heads, evaluation isolation, checkpoint round trips, and a small training run.
The Rust tests still check every legal tic-tac-toe game sequence.

Python edits are available immediately through the editable installation. After
changing Rust sources, rebuild the native extension:

```sh
uv sync --locked --reinstall-package blokus-rl
```

The [self-play experiment](self-play.md) now implements saved-policy opponents,
matched PPO controls, a small AlphaZero-style arm, and exact stronger-opponent
evaluation. A later experiment can compare Monte Carlo advantages with GAE. The
PettingZoo interface and PyTorch/TorchRL stack carry forward. Blokus will need a
new engine, observations that include remaining pieces, and a legal-move encoding
that describes piece, orientation, and placement. Four-player rewards and
opponent populations will require their own design.

References: [PyTorch model basics](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html),
[TorchRL's PPO tutorial](https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html),
[masked distributions](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.MaskedCategorical.html),
and [PettingZoo's turn-taking API](https://pettingzoo.farama.org/api/aec/).
