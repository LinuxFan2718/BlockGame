# Probe generalization against other learned strategies

Declared after all three frontier curricula passed their specified criteria.
Frontier strict-win rates were 73.25%, 69.75%, and 63.5%, with improvements
over their respective parents of 26, 13.5, and 11 percentage points on fresh
paired-by-seed test sets. This establishes progress against named heuristics,
not a general solution to the game.

Freeze the three selected frontier models 342/343/344 and their selected
longer-margin parents 242/243/244. First compute exact deterministic empty-board
cross-play: one game per seat for each pair. Report these as exactly two games,
without confidence intervals or fictitious repetitions.

Then compare the frontier models against one another on 100 common randomized
openings. Each opening plays four uniformly sampled legal moves using Rust,
then runs the two greedy frozen neural policies to termination. Replay each
opening with the policies in both seats. These are games from a declared
opening distribution, not claims about perfect play or ordinary empty-board
win rates. Starting-board randomness tests whether a policy can recover from
unfamiliar geometry and inventories.

The two seat-swapped games share an opening, so confidence intervals must resample
whole opening pairs. Use a fixed-seed nonparametric cluster bootstrap with 10,000
resamples for mean score rate and strict-win rate. Do not treat the two games as
independent binomial observations. Save opening moves and a digest, hashes of
all checkpoints, results by seat, and one replayable loss example per matchup.

Opening seed 1790000. This is an exploratory probe; its findings may motivate a
subsequent frozen-opponent league. Such a training experiment must declare its
opponent pool, sampling, budget, validation and fresh final test seeds separately.
