# Tic-tac-toe: a first environment for reinforcement learning

This project provides a Rust game engine, human and random players, and a
PyTorch neural agent trained with TorchRL's PPO loss. The Rust engine has no
external dependencies. Python training is an optional, separate package.

For the neural-network lesson, training, saved-model evaluation, and playing
against the trained bot, start with [the PyTorch and PPO walkthrough](docs/training.md).
For self-play, search-guided training, and exact never-lose checks, see the
[experiment guide](docs/self-play.md) and [declared evaluation protocol](docs/self-play-protocol.md).
The completed [results discussion](docs/self-play-results.md) includes two neural
policies verified never to lose from the normal start, playing either mark.

Blokus Duo now has a separate live Rust search and AlphaZero-style self-play
loop. Read the [rolling search experiment](docs/search-experiment.md) for the
method, held-out comparisons, Pentobi benchmark and hardware measurements.
Current jobs and follow-up steps are in [search status](docs/search-status.md).
The completed [search/self-play results and plots](docs/search-results.md)
include both successful and failed comparisons, plus an independent Pentobi benchmark.
The completed [Modal hardware comparison](docs/modal-results.md) measures complete
search throughput on the Mac, L4 and H100; [the assessment](docs/modal-experiment.md)
records costs and the decision to keep the current small network local.
The [AlphaBlokus source study](docs/alphablokus-study.md) explains its published
residual CNN, four-player search, self-play targets and GPU batching.
A shorter [saved explanation](docs/alphablokus-explanation.md) follows the learning
loop. The [deep CNN overnight study](docs/deep-self-play.md) and
[completed results and plot](docs/deep-results.md) cover two randomly initialized
CNNs trained through batched self-play. The smaller CNN won that comparison;
Pentobi level 1 beat both earlier models. Audit details are in [deep status](docs/deep-status.md).

The [completed Pentobi levels 1–9 benchmark](docs/pentobi-level-results.md)
tests that frozen small CNN with 128 search simulations over 1,800 games.
It scored 27.5% against level 1 and remains below that level under this protocol.
The [completed controlled continuation](docs/continuation-results.md) improved
the same small CNN through additional self-play. Stronger training search won
the primary comparison and its selected model scored 62.5% against Pentobi level 1
at 128 playing simulations, over 200 fresh held-out games. The subsequent
[sequential test stopped at level 2, scoring 31.5%](docs/strong-pentobi-levels.md).
The [frozen-model search sweep](docs/playing-search-sweep.md) then scored
42.0%, 62.75% and 75.0% against level 2 at 128, 512 and 2,048 simulations,
respectively, using the same fresh openings across budgets. Extra planning
improved strength without changing the network weights.
The subsequent [level 3 test at 2,048 simulations](docs/pentobi-level3.md)
scored **58.25%** (113 wins, 7 draws, 80 losses; nominal paired 95% interval
52.25%–64.25%) over 200 fresh games. The model remained frozen.
The [next test stopped at level 4](docs/pentobi-level4-5.md): 33.75% score
(63 wins, 9 draws, 128 losses; nominal paired 95% interval 27.75%–39.50%).
Level 5 was conditional on a win and was not run. The report recommends a
controlled self-play continuation using stronger training search.
The [new controlled training study](docs/teacher-search.md) is testing
1,024 versus 2,048 simulations for self-play targets. It was cancelled after
1,280 control games; the checkpoint is preserved, with no held-out result yet.
See the [protocol](docs/continuation-status.md) for settings, costs and limitations.
To play the selected agent at the fast 128-simulation budget:
`.venv/bin/blokus-rl duo-play runs/duo-continuation-final-20261026T2007/duo-continuation-865/frozen-strong1024.pt --simulations 128`.
Use `--simulations 2048` for the stronger playing budget from the sweep.
Earlier checkpoints remain preserved. Training entry point:
`.venv/bin/python -m blokus_rl.duo_az --help`.

```sh
uv sync --locked
uv run blokus-rl inspect
uv run blokus-rl train --out runs/my-ppo-run
uv run blokus-rl play runs/my-ppo-run/model.pt
```

## Play and inspect

Run these commands from this directory with Rust and Cargo installed. This uses
Rust's 2024 edition, supported by Rust 1.85 and later.

```sh
# You play X against a random bot. X always starts.
cargo run

# Two humans, or play O against a random X.
cargo run -- --x human --o human
cargo run -- --x random --o human

# Watch two random bots.
cargo run -- --x random --o random --seed 42

# Headless episodes for a reproducible baseline.
cargo run --release -- --x random --o random --games 10000 --seed 42 --quiet

# Record an episode as observations, actions, and final returns.
cargo run --example self_play

# Tests (including documentation examples), formatting, and lint checks.
cargo test --offline
cargo fmt --all -- --check
cargo clippy --offline --all-targets -- -D warnings
```

Humans enter squares **1–9** as displayed, or `q` to quit. End of input also exits
cleanly. Invalid input and occupied squares prompt again without losing a turn.
The library uses **0–8**, so models can index a nine-element output array:

```text
Library actions:       Human input:
 0 | 1 | 2              1 | 2 | 3
---+---+---            ---+---+---
 3 | 4 | 5              4 | 5 | 6
---+---+---            ---+---+---
 6 | 7 | 8              7 | 8 | 9
```

## How the pieces fit together

| File | Responsibility |
| --- | --- |
| `src/game.rs` | State, legal moves, turns, outcomes, observations, rewards |
| `src/agent.rs` | The `Agent` policy interface and seeded random baseline |
| `src/main.rs` | Human input, terminal rendering, and batch game totals |
| `examples/self_play.rs` | Record a complete episode without human input |
| `tests/engine.rs` | Rules, observations, rewards, reproducibility, exhaustive verification |
| `tests/cli.rs` | Human input, quitting, batch play, and argument errors |
| `bindings/python/` | Thin PyO3 binding to these same Rust rules |
| `python/blokus_rl/model.py` | A small PyTorch MLP with actor and critic heads |
| `python/blokus_rl/env.py` | PettingZoo turns and a fixed-opponent training adapter |
| `python/blokus_rl/learning.py` | Rollouts, returns, TorchRL PPO, evaluation, and checkpoints |
| `python/blokus_rl/audit.py` | Exact results, minimax evaluation, and never-lose certificates |
| `python/blokus_rl/self_play.py` | Current-policy and immutable past-policy opponents |
| `python/blokus_rl/search.py` | Small MCTS and search-guided policy/value learning |
| `python/blokus_rl/experiments.py` | Matched training runs, checkpoint selection, and reports |

The rules have no terminal input, timers, model code, or randomness. A human,
random bot, or trained bot chooses an action; `Game::step` applies the same rules
to all of them. This separation will let us change how a bot decides without
changing what counts as legal play.

`Game` owns a private nine-cell board. Each cell is `None`, `Some(Player::X)`, or
`Some(Player::O)`. Enums make these possibilities explicit, and private fields
prevent callers from creating impossible positions. `Result<Step, MoveError>`
requires callers to handle success or an invalid move explicitly. A copied game
is an independent snapshot, useful for future simulations and search.

## The agent/environment loop

In RL, an **environment** implements the world and its rules. An **agent** chooses
actions using a **policy**, a rule or model that maps observations to decisions.
One complete game is an **episode**. The trainer adjusts a policy using experience
from episodes. This is the same observe/act/step pattern used by
[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/), though this is
a Rust, two-player API rather than a Python Gymnasium adapter.

```rust
use tic_tac_toe::{Agent, Game, RandomAgent};

let mut game = Game::new();
let mut bot = RandomAgent::new(42);

while let Some(player) = game.current_player() {
    let observation = game.observe(player);
    let action = bot.select_action(&observation).unwrap();
    let step = game.step(action).unwrap();

    // A future trainer can record the observation, action, and rewards.
    let reward = step.reward_for(player);
    if step.is_terminal() {
        let opponent_reward = step.reward_for(player.opponent());
        // Both players need their terminal feedback.
    }
}

game.reset(); // A new empty board, with X to move.
```

The example bot can control both sides. A trainer can instead keep separate
agents, freeze one opponent, or play against older versions of the learning bot.

`RandomAgent` implements `Agent::select_action(&Observation) -> Option<usize>`.
It samples uniformly from legal moves and returns `None` on an opponent's turn
or when no action is available. It does not know any winning strategy. A learned
policy can implement this same interface. The engine validates every action even
if an agent ignores the legal mask.

## Observation and action contract

`game.observe(perspective)` returns an `Observation`:

| Field | Meaning |
| --- | --- |
| `perspective` | The player whose marks count as “mine” |
| `cells: [i8; 9]` | `+1` = mine, `-1` = opponent, `0` = empty, in row-major order |
| `to_move: i8` | `+1` = my turn, `-1` = their turn, `0` = game over |
| `legal_mask: [bool; 9]` | `true` where the current player may move |

For a position with X in the upper left and O in the center:

```text
X's cells: [ 1, 0, 0, 0, -1, 0, 0, 0, 0]
O's cells: [-1, 0, 0, 0,  1, 0, 0, 0, 0]
```

Encoding “mine” and “theirs” consistently lets one policy play either mark.
The `perspective` field is metadata; a model can consume the numeric features.
Supply observations to an agent on its turn, usually with
`game.observe(game.current_player().unwrap())`.

The mask describes legal actions for the **current player**, even if the
observation was requested from the other player's perspective. A nonterminal
observation with `to_move == -1` does not authorize that observer to act. Terminal
observations always have an all-false mask, including when empty squares remain.

For a neural policy with nine logits, set illegal logits to negative
infinity before softmax and sampling. For Q-learning, restrict both action
selection and the next-state maximum to legal actions. Never sample or bootstrap
from an all-false terminal mask.

`step(action)` accepts a `usize`. Out-of-range actions, occupied squares, and
moves after termination return a typed error and leave **all** state unchanged.
An illegal move does not consume a turn or generate a reward. If we later want
to teach a bot legality through penalties, that behavior belongs in an explicit
training wrapper.

## Rewards, returns, and two-player bookkeeping

Each successful `step` returns the acting player, action, and resulting outcome.
`step.reward_for(player)` uses this reward convention:

| Result after the move | X reward | O reward |
| --- | --- | --- |
| Still playing | 0 | 0 |
| X wins | +1 | -1 |
| O wins | -1 | +1 |
| Draw | 0 | 0 |

These are **sparse rewards**: no intermediate bonuses for taking the center or
blocking an opponent. Strategy must be learned from outcomes. The trainer's
credit-assignment method connects earlier decisions to those later outcomes.

A **reward** is feedback at one step. A **return** combines future rewards,
possibly discounting more distant ones by a factor gamma. With gamma = 1 and
only these terminal rewards, each turn's eventual return is its player's final
reward. `examples/self_play.rs` records that undiscounted return for every turn,
including the losing player's earlier actions. It produces experience but does
not update a policy.

One engine step is **one mark**, not a learner move plus an opponent response.
After X moves, the next actor is O. This is a turn-taking game, so do not blindly
apply a single-agent Q-learning update across alternating player perspectives.
Our Python `FixedOpponentEnv` applies the learner's move and then the opponent's
move, returning an observation and reward from the learner's fixed perspective.
It handles the opponent's opening move when the learner plays O, and stops
immediately if either move ends the game. Alternating self-play needs explicit value-perspective handling, or
whole-episode returns as illustrated here.

At termination, read rewards for **both players**. The losing player does not
get another move, but still needs its loss recorded. Terminal values have no
future reward to bootstrap from. Episodes naturally end in at most nine legal
moves, so the engine does not need a separate time-limit/truncation signal.

## Reproducibility and verification

The game is deterministic: a given action sequence always gives the same result.
Randomness lives only in the bot. It uses the small
[SplitMix64 reference algorithm](https://prng.di.unimi.it/splitmix64.c) and rejection
sampling to avoid modulo bias.
The CLI derives a different seed for O, keeps both RNGs running across episodes,
and reports the input seed. The same seed, controllers, and game count reproduce
the same run; seeds alone do not make different policies comparable.

The exhaustive test traverses every legal move sequence, stopping at wins and
draws, and compares the engine with a separate coordinate-based rules oracle.
It covers 5,478 distinct reachable positions and 255,168 completed sequences:
131,184 X wins, 77,904 O wins, and 46,080 draws. Those counts enumerate possible
sequences, **not** the probabilities under a uniformly random policy, because
different sequence lengths have different probabilities. Other tests cover
invalid moves, terminal behavior, observations, rewards, seeds, and the CLI.

## From transformer work to RL

A transformer is a model architecture. RL is a way of learning from the
consequences of decisions; it does not require a particular architecture. A
transformer could be the policy, but for this board a small table or neural
network will make the learning mechanism easier to inspect.

In next-token training, an example supplies the target next token. Here a game
usually does not label the best move at each turn. The agent tries moves, sees
eventual rewards, and the training algorithm uses them to improve future
decisions. Exploration matters because a move that currently looks bad may
actually be useful once the agent has more experience.

RL can exceed human play in a defined task: AlphaGo Zero learned through
self-play without human game examples, using a neural network with tree search.
That result is task-specific expertise, not a guarantee of general intelligence.
See the researchers' account,
[AlphaGo Zero: Starting from scratch](https://deepmind.google/blog/alphago-zero-starting-from-scratch/).

The first training experiment uses a small neural policy and PPO against the
random bot, then evaluates frozen weights as both X and O. A tabular learner
would also be a useful comparison: its table would be populated from experience
rather than hand-written strategy. The [training walkthrough](docs/training.md)
explains the implemented experiment and how we can extend it toward self-play.
Track wins, draws, and losses separately; an average reward can hide differences
in how often a bot loses.

## Blokus Duo

The second Rust engine is implemented in `src/duo.rs`, with the same
PyO3 → PyTorch → TorchRL training stack. It supports legal placements, rotations
and flips, both starting points, forced passes, scoring bonuses, human terminal
play, uniform random bots, a largest-piece baseline, and learned neural bots.

```sh
uv run blokus-rl duo-train --out runs/my-duo-bot
uv run blokus-rl duo-evaluate runs/my-duo-bot/best.pt --opponent largest
uv run blokus-rl duo-play runs/my-duo-bot/best.pt --human 0
```

During human play, `?` shows the remaining piece shapes. Enter a piece number
to list its legal placements, then `m MOVE_ID` to play one. `q` quits. You can
also use `duo-play random` or `duo-play largest` without a trained checkpoint.

Read the [Blokus walkthrough](docs/blokus-duo.md),
[declared pilot protocol](docs/blokus-duo-protocol.md), and
[experiment results](docs/blokus-duo-results.md). The
[overnight summary](docs/overnight-summary.md) explains the final findings and models.
The
[overnight work log](docs/overnight-status.md) records active runs and next steps.
Training artifacts live under
ignored `runs/`; preserve that directory if you want to keep the trained weights.

Four-player Blokus will need a new reward and opponent model; the two-player
zero-sum assumption should not carry over automatically.
