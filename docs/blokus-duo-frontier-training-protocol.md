# Training against the held-out weakness

The short margin-142 policy won 40.5% against the frontier heuristic on 200
games (versus its 69% wins against largest-piece play on a different test set).
Frontier itself won 77.5% against largest. This motivates a new curriculum;
the original frontier probe becomes development evidence for the new experiment.

After the longer run finishes, start from its selected checkpoint. Train with
the corrected observation schema (scores and pass count), seed 342, terminal
score-margin reward, 200 updates × 32 episodes = 6,400 games against frontier.
Other PPO defaults stay unchanged. Parent weights migrate with zero new-input
columns; this does not claim exact optimizer resumption.

Validate every 20 updates on 40 games per seat starting at seed 870000. Evaluate
random, largest and frontier; select highest mean score rate against largest and
frontier, breaking ties with frontier. Random is monitored for regressions.
Final test: 200 games per seat with fresh seed 990000, all three opponents, plus
the sampled frozen starting policy. Save initial, selected and final.

Success: at least a five-point frontier strict-win gain over the parent on these
same test seeds, a strict-win lower 95% bound over 50% against frontier, and at
least 90% score rate against random. This is a new, exploratory experiment; a
pass would warrant replication or an additional held-out opponent. A failure
should be recorded without escalating compute indefinitely.
