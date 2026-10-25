# Overnight result

The Rust Blokus Duo engine and PyTorch/TorchRL training pipeline work. Learning
from random initialization succeeded, and a staged curriculum improved policies
across three independent training pipelines. The final two follow-up hypotheses
did not improve on their controls, so automatic continuation was deleted under
the predeclared stopping rule. No training jobs remain intentionally active.

## Models to try

| Policy | Evaluation | Observed strict wins |
|---|---|---|
| `runs/duo-frontier-training-342/best.pt` | Random / largest-piece / corner-count heuristic, 400 fresh games each | 98.25% / 82.75% / 73.25% |
| `runs/duo-longer-margin-243/best.pt` | Random / largest-piece, 2,000 fresh games each | 99.15% / 91.35% |

These are named-opponent benchmarks, not optimality or human-level claims. The
first policy is stronger against the corner heuristic; the second has the highest
observed largest-piece result. Neither establishes a universal ranking.

```sh
uv run blokus-rl duo-play runs/duo-frontier-training-342/best.pt
```

Use `?` to show remaining shapes, a piece number to list its placements, then
`m MOVE_ID` to play. Rust checks every move. `q` quits.

## What we learned

1. Small PPO policies can cold-start on this game. Three initial 1,600-game runs
   reached 92.5–95.5% strict wins against random play, improving on their own
   untrained policies. Heuristic strength was initially much lower.
2. Terminal score-margin rewards and harder opponents helped. Three longer runs
   reached 82.35–91.35% wins against largest-piece play. A further corner-heuristic
   curriculum reached 63.5–73.25% against that stronger opponent; all three runs
   passed their declared improvement criteria.
3. A benchmark is a ruler, not a definition of general skill. Peer neural policies
   ranked differently from the heuristics. Paired randomized openings exposed
   weaknesses hidden by repeated deterministic empty-board games.
4. More training did not always help. A matched mixture of frozen neural and
   heuristic opponents gained only 0.5 score points over its parent against an
   excluded neural policy and lost 11.5 points against the corner heuristic.
   Its paired confidence interval included losses and gains.
5. More expressive representation did not win the small-budget screen. A spatial
   convolutional actor scored 18% against frontier versus the additive control's
   44%, using the same 1,600-game budget and more computation. That is not a
   general verdict on convolution; it failed this specific screen and does not
   justify replacing the earlier policies.

## Code and verification

Rust owns legal moves, forced passes, piece inventories, both starting points
and scoring bonuses. Python supplies PettingZoo observations, human play and
training. TorchRL computes PPO losses; PyTorch supplies the networks and gradients.
Both additive and convolutional models use the same interfaces. Checkpoints
record their architecture and load earlier formats through tested migration.

The latest full suite passed 52 Python tests; 20 Rust tests and core/binding
clippy passed before the Python-only follow-ups. The certified tic-tac-toe models
remain untouched. The Rust tests compare optimized move generation to an
independent cell checker. Python tests cover legal masking, perspectives,
on-policy log probabilities, frozen references, leakage guards, paired evaluation,
orientation/anchor mapping, PPO updates and checkpoint round trips.

All methods, negative results and weights are retained. Models live in ignored
`runs/`; preserve it when moving the project. No paid training resources, publishing,
messages to people, subagents, commits or pull requests were used.

Read the [learning walkthrough](blokus-duo.md),
[curriculum results](blokus-duo-results.md),
[neural cross-play probe](blokus-duo-crossplay-results.md),
[failed frozen-mixture experiment](blokus-duo-frozen-league-results.md), and
[convolution screen](blokus-duo-convolution-results.md). Each links its declared
protocol. Further work can be chosen after reviewing these results; automatic
overnight runs have stopped.
