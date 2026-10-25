# Self-play experiment protocol

This protocol is recorded before running the experiment. The question is whether
changing the source of experience improves the learned policy's tactical soundness.
Success and failure are determined by evaluation, rather than by training reward.

## Arms and budgets

- **Random-opponent PPO control:** the existing trainer's algorithm and random opponent.
- **Self-play PPO:** same network, initialization seeds, PPO loss, learning rate,
  batch game count, epochs, minibatch size, entropy coefficient, and reward.
  In 80% of episodes the opponent samples from the current policy, which is frozen
  during collection. In 20% it samples from a uniformly chosen saved earlier policy.
  Store an immutable snapshot every 10 updates, including the initial policy.
- **Small AlphaZero-style arm:** the same MLP and Rust rules, with 32 PUCT MCTS
  simulations per move, search-guided self-play, policy cross-entropy targets from
  visit counts, and value targets from final outcomes. This is a teaching-scale
  experiment, not a reproduction of DeepMind's full system. Report both the raw
  network and the network with search. Evaluate search before training as well.

Each arm starts from random weights for seeds **42, 43, 44**, uses 300 updates
and 128 complete games per update: **38,400 training games per run**. PPO uses
four epochs, minibatches of 256, Adam at 0.0005, gamma 1, clipping epsilon 0.2,
entropy coefficient 0.01, and value coefficient 0.5. The AlphaZero-style arm uses
four epochs, minibatches of 256, Adam at 0.0005 with weight decay 0.0001, value MSE
coefficient 1, and root noise (25% Dirichlet with alpha 0.3) during training only.

Equal games are **not equal compute**: search consumes extra work, and collecting
both players' search targets gives more positions per game. Record positions,
optimizer steps, MCTS simulations, and wall time. Conclusions about self-play's
effect come from the two matched PPO arms. Search-assisted deployment and raw
network performance are separate comparisons.

## Evaluation and checkpoint selection

Build the complete graph of reachable positions using the Rust engine; no second
Python rules implementation is used. Minimax is evaluation-only: it never supplies
training moves, rewards, value targets, or policy labels.

The primary target is a **deterministic policy that never loses from the empty
board, as either X or O, against ANY legal opponent sequence**. Enumerating every
opponent response can certify this for tic-tac-toe. This certificate covers that
policy and deployment mode; it does not cover stochastic move sampling or every
arbitrary midgame position.

Secondary measurements, reported separately for X and O:

1. Exact win/draw/loss probabilities against a uniformly random opponent.
2. Exact results against a perfect opponent that selects uniformly among all
   minimax-optimal actions. A solved agent draws every game in this test.
3. Avoidable move errors across all nonterminal positions: selecting an action
   with worse minimax value than the best legal action. This includes positions
   outside the evaluated policy's usual trajectories.
4. A concrete losing move sequence if the primary certificate fails.

These probabilities sum all branches, rather than estimate them from Monte Carlo
games. Sequence COUNTS are also recorded to certify nonzero losses, but they are
not probabilities and should not be compared as if differently sized trees have
equal weight.

Audit the initial model and every 25 updates, and retain the best checkpoint using
this predeclared lexicographic criterion: first a successful never-lose certificate;
then lower summed loss probability against perfect play; then fewer avoidable
action errors; then lower loss against random; then higher win probability against
random. Record the final checkpoint separately. The AlphaZero-style arm selects
its checkpoint using the raw network; search evaluation is an additional result.

The evaluator therefore participates in checkpoint selection, but never gradient
training. Report all three seeds, not just the strongest model. No change to these
criteria or the training budget is made to turn an unfavorable result into success.

## Interpretation

- **Solved in a run:** the exact never-lose certificate passes for both seats.
- **Partial improvement:** the certificate fails, but tactical errors and/or losses
  against perfect play decrease. Check whether this repeats across seeds.
- **No demonstrated benefit:** the matched self-play PPO arm does not improve
  those measures relative to the random-opponent control.
- **Regression:** improved random-opponent wins accompany worse tactical soundness
  or perfect-opponent performance. Preserve that result and inspect it.

Three seeds provide a small reproducibility check, not a broad statistical claim.
