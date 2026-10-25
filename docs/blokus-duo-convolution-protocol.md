# Bounded spatial-representation pilot

The frozen-neural curriculum failed its held-out transfer/retention test.
An additive contextual square/piece actor cannot express an arbitrary score
for each oriented placement. Test a more expressive spatial actor rather than
merely adding more games to the failed mixture.

The convolutional actor takes the same observation and legal candidates. Four
3×3 convolution layers (32 channels) share board features across positions.
Global inventory/score context modulates the spatial features. A policy head
has 91 orientation channels at every possible anchor, plus forced pass.
Rust still supplies legal moves; no handcrafted heuristic scores are inputs or
targets. This is still a feedforward network, not a transformer or recurrent model.

Matched cold-start pilot: additive MLP versus convolutional actor, random weights,
seed 542, 100 updates × 16 games = 1,600 games each against frontier. Reward is
terminal score margin; other PPO defaults unchanged. Validation every 10
updates, 30 games per seat on random/largest/frontier, seed 2890000. Select mean
largest/frontier score rate, tie-break with frontier. Final test 100 games per
seat on all three baselines, seed 3290000. Compare initial and selected policies.

This pilot screens whether the implementation learns at all at a small budget.
An encouraging result is a frontier score gain of at least five points over the
MLP control, with at least 90% random score rate. A pass warrants independent
seeds and a larger prespecified budget; it is not sufficient evidence of a general
architecture advantage. Report parameter count, transitions and elapsed time:
equal game budgets are not equal compute. Do not warm-start one arm with the
large pretrained model while starting the other from scratch.

Tests must check that every candidate maps to the correct native orientation
and anchor in both observer perspectives, legal padding stays masked, PPO
updates finite parameters, and checkpoint loading preserves predictions.

If this screen fails its criterion, do not scale it up automatically overnight.
Record the two unsuccessful follow-up ideas (frozen mixture and this screen)
and stop the continuation for review. If it passes, prescribe a larger matched
comparison across independent seeds before examining its final test results.
