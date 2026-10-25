# Longer-budget margin experiment

Declared before its run. The short largest-opponent margin experiment's seed
142 checkpoint achieved 69% strict wins against largest-piece play. Its
independent replications are still running. This follow-up tests whether more
experience continues to improve that fixed parent; it does not choose a parent
by comparing the replication results.

Warm start `runs/duo-curriculum-largest-margin-142/best.pt`, seed 242. Keep
terminal score-margin reward, largest-piece opponents, hidden size 128, Adam
0.0003, entropy 0.02 and three PPO epochs. Run 1,000 updates × 32 games = 32,000
additional games. This is a new optimizer/RNG trajectory, not exact resumption.

Validate every 50 updates, 80 games per seat, seed 570000. Select the highest
mean random/largest validation score rate, tie-break with largest. Final test
uses 1,000 games per seat with fresh seed 690000. Test data do not choose the
checkpoint. Compare the warm-start parent and selected against both baselines.

A worthwhile gain is at least five percentage points in largest-opponent strict
win rate over the parent on the same test seeds, retaining at least 90% score
rate against random. This is still an exploratory fixed-opponent benchmark;
future work should probe another baseline and worst-case weaknesses, rather
than call a larger win rate a general solution to Blokus.
