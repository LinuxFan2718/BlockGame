# From tic-tac-toe to Blokus Duo

The same tools still fit: Rust enforces rules; Python exposes the environment;
PyTorch represents the neural network and computes gradients; TorchRL computes
the PPO loss. The game and the model's representation change, while the basic
learning loop stays recognizable.

## Why legal moves are now candidates

Tic-tac-toe always has nine possible board squares, so nine output logits are
natural. Blokus adds a piece choice, rotation/reflection and board position.
We number 91 distinct oriented shapes, then 196 top-left squares per orientation,
plus a forced-pass action. Some encodings are always invalid because a piece
would extend beyond the board. Rust filters all of them.

PettingZoo exposes a fixed action space of 17,837 integers for compatibility.
The neural policy works on a variable-length list of legal placements. This
avoids training thousands of impossible outputs at every turn. For each legal
candidate, its compact encoding holds a piece id and up to five occupied cells.

The network sees two relative board planes (mine/theirs), both inventories,
the active player's required starting squares and a first-player flag. Player
1's coordinates rotate 180 degrees; the engine's action ids remain absolute.
The model outputs scores for the 196 squares and 22 piece choices (including
pass). The score for a placement adds its square scores to its piece score:

```
logit(move | state) = piece_score[piece] + sum(square_score[cell] for cell in move)
policy(move | state) = softmax(logits of legal candidates)
```

Those scores depend on the whole observed state. Nothing says “large pieces
are good” in the weights or targets. This compositional scoring rule is an
inductive bias: we assume useful move preferences can be expressed through
context-dependent square/piece values. It is less expressive than an arbitrary
score for every placement. A later model could score each candidate with a
larger network, add convolutions, or use attention. We should change architecture
when measured limitations justify it.

It remains **feedforward**: one observation goes through the network without
a recurrent memory of previous turns. The board plus inventories tells us the
available placements; a history model is not needed to remember which pieces
were used. A transformer could still encode the board or candidates, but is
not required by reinforcement learning.

After the initial experiments, the observation also gained both current scores
and consecutive-pass count. Inventories imply negative unplayed-square scores,
but they cannot reveal whether a player who used every piece earned the
monomino-last bonus. That history-dependent bonus must be observable too.
The pass counter specifies exactly when a forced pass ends the episode.
Version-1 policies are loaded by adding zero weights for these three inputs,
preserving their learned behavior; new training can learn to use the inputs.
The original experiment processes retain their original observations in memory.

## What PPO actually learns

An episode fixes the learner's seat and opponent. The learner samples legal
moves from its current policy; the opponent supplies replies. The game ends
with a score, which becomes a win/draw/loss reward +1/0/-1. Earlier learner
decisions receive that final outcome as a Monte Carlo return (gamma 1).

The critic predicts that return. The advantage is observed return minus the
saved prediction. A positive advantage asks PPO to make the sampled action more
likely; a negative one asks it to make the action less likely. Many actions in
a won game may have been poor, so an individual update has noisy credit
assignment. Learning comes from repeated experience, not one game's labels.

During collection the weights stay fixed. We save each decision's candidates,
chosen candidate index, old log probability and value estimate. Minibatches
pad their candidate lists and mask the padding. TorchRL compares new and old
action probabilities and clips the policy objective; PyTorch differentiates
the loss and Adam updates the shared parameters. The shared network is only
added once to the optimizer.

The default pilot trains against random legal moves. The follow-up league
samples frozen historical neural opponents for complete games. The mixed
curriculum also includes a largest-piece baseline. A handcrafted **opponent**
is not a set of supervised best-move labels, but it still introduces human
choices into the training task. We record that distinction.

## How we judge progress

Blokus has no exhaustive certificate here. We freeze checkpoints and evaluate
fresh games in both seats against named, fixed baselines. The learner chooses
its highest-probability move. Random opponents and largest-piece tie breaks
supply randomness. Frozen neural opponents sample their saved distributions;
two greedy deterministic neural bots would otherwise replay only two games.

We report wins, draws and losses, average score margin, and strict-win confidence
intervals. A draw is half a point in the score rate but is not a win in that
interval. Validation selects a checkpoint; held-out games assess that selected
checkpoint. Once a test result informs another experiment, the next experiment
gets new final test seeds. Random-opponent success does not establish strong
human play, and a confidence interval over games is not a substitute for
replication across training seeds.

Useful commands:

```sh
uv run blokus-rl duo-train --out runs/duo-example --seed 42
uv run blokus-rl duo-evaluate runs/duo-example/best.pt --opponent largest --games-per-seat 500
uv run blokus-rl duo-evaluate runs/duo-example/best.pt --opponent runs/duo-example/initial.pt
uv run blokus-rl duo-play runs/duo-example/best.pt --human 1
```

Each run writes `protocol.json`, append-only `progress.jsonl`, `metrics.csv`,
`summary.json`, learning curves, and initial/best/final/latest checkpoints.
`--init` starts a new experiment from existing weights; it resets the optimizer
and randomness and records the parent file hash. It does not claim exact
resumption of the parent's training trajectory.

The rules source is [Mattel's official instruction sheet](https://service.mattel.com/instruction_sheets/FWG43-Eng.pdf).
The optimized Rust implementation is checked against a separate cell-by-cell
legality oracle across complete games. Python tests check the PettingZoo
contract, coordinate transforms, legal candidate padding, rollout log
probabilities, parameter updates and checkpoint round trips.

The later `frontier` baseline breaks largest-piece ties using the difference in
free corner anchors after a move. It provides another ruler for strength; its
scores do not label the learner's actions. You can evaluate against it with
`duo-evaluate CHECKPOINT --opponent frontier`. The additional
`--selection frontier` training option validates against all three baselines
and selects on mean largest/frontier score rate, then frontier. This selection
criterion was introduced for the separately declared frontier curriculum.

The later `--architecture conv` experiment uses convolutions to share local
board features across positions. Its policy tensor has one channel per oriented
shape: `logit[orientation, row, column]`, plus a pass output. It can assign a
shape-specific score to a whole placement. Inventories and scores modulate the
board features through a global context branch. Rust's legal candidate list
still decides which outputs are selectable. This changes representation, not
the PPO learning rule. Convolution is a kind of feedforward computation too.
The [bounded comparison protocol](blokus-duo-convolution-protocol.md) explains
the planned comparison; its existence alone is not evidence that it is better.
