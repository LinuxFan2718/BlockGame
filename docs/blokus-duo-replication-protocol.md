# Replicate the full curriculum with corrected observations

The seed-342 frontier stage passed: its selected policy won 73.25% against
frontier, versus its parent's 47.25% on the same 400 test games, retained 98.25%
wins against random and 82.75% against largest. This warrants replication.

Use the independent short-margin parents 143 and 144. For each, first run the
same 32,000-game longer-margin stage with seeds 243 and 244, then the same
6,400-game frontier stage with seeds 343 and 344. Both stages now use observation
version 2, which includes scores and pass count. These are replications of the
curriculum with the corrected representation, not bitwise reproduction of the
version-1 original longer stage.

Longer stages: largest opponents, terminal margin reward, default PPO settings,
validation every 50 updates × 80 games per seat at seed 570000, select balanced
random/largest score rate. Final test seed 1290000, 1,000 games per seat.

Frontier stages: 200 updates × 32 games, validate every 20 updates × 40 games
per seat at seed 870000, select mean largest/frontier score rate. Final test
seed 1590000, 200 games per seat. Same declared success conditions as the first
frontier stage (gain >=5 points, frontier strict-win 95% lower bound >50%,
random score rate >=90%). Record both seats and retained largest performance.

The shared validation seeds make selection comparable; new final seeds prevent
reuse of test outcomes that motivated these replications. Keep all runs,
including failures. Further exploration should probe generalization to neural
opponents or richer move representations, rather than indefinitely adding games
against the same heuristics.
