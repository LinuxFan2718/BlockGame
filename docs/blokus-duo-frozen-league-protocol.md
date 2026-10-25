# Matched frozen-opponent curriculum experiment

Declared after the exploratory cross-play probe. Frontier-342 scored 61.75%
against frontier-343 on paired random openings, but only 45% against frontier-344
(95% paired score interval 37.75–52.75%). Frontier-344 had the lowest heuristic
benchmark among the three. This is evidence that heuristic rank alone does not
determine neural cross-play rank; it is not proof of a statistically significant
loss to 344. Empty-board greedy matches are separately just two games per pair.

Freeze frontier-342 as the common learner parent. Seed 442, 300 updates × 32
games = 9,600 games per arm, complete-game PPO with terminal score-margin reward,
other existing defaults (three epochs, hidden 128, Adam .0003, entropy .02).

- control: continue frontier-opponent training.
- mixed: per game choose 10% random, 20% largest, 20% frontier, 50% uniformly
  selected frozen neural opponents frontier-343 and longer-margin-243.

Neural opponents act greedily in training. The learner samples its on-policy
distribution, providing exploration even against a deterministic frozen rival.
Every opponent is fixed for the whole game. No training moves or data come from
frontier-344, the held-out neural opponent. That model informed the hypothesis
through the exploratory probe, but its policy is never consulted during training
or checkpoint selection; use fresh test openings for this experiment.

Validate every 25 updates: random/largest/frontier 30 games per seat, plus 20
paired four-ply randomized openings against frontier-343. Validation seed 1890000.
Select maximum mean of largest, frontier and known-neural score rates, breaking
ties with known-neural performance. The two arms share this selection criterion.

Final tests: 200 games per seat on baselines, and 100 paired opening positions
against frontier-343 (seen), longer-243 (seen) and frontier-344 (held-out). New
test seed 2290000. Report initial and selected checkpoints. Bootstrap whole
opening pairs for neural-match intervals. Do not assign binomial intervals to
the correlated two-game pairs.

Useful evidence for the mixed curriculum requires a held-out neural score gain
of at least five points over its parent and over the matched control, retaining
at least 90% random score rate and no more than a five-point loss against either
heuristic versus the parent on the same test sets. Treat a single-seed gain as
exploratory. A failure can motivate a representation change; it does not justify
indefinitely increasing game budgets against the same fixed opponents.
