# Results of the self-play experiment

We completed all nine declared runs: three algorithms, each with initialization
seeds 42, 43, and 44 and a budget of 38,400 games. The architecture stayed at
5,450 parameters. The [protocol](self-play-protocol.md) was written before the
experiment; the [full local report](../runs/self-play-study/report.md) and
[JSON results](../runs/self-play-study/comparison.json) preserve every run.

## The primary result

| Training method | Selected raw policies certified never to lose in either seat |
| --- | ---: |
| PPO against random | 0 / 3 |
| PPO against current and saved policies | 0 / 3 |
| Small AlphaZero-style search-guided learning | 2 / 3 |

The primary criterion was zero losing continuations from the empty board, as
both X and O, for a deterministic greedy policy. It was checked against every
legal opponent response, not a sample of games. The two passing checkpoints are:

- `runs/self-play-study/alphazero-seed-42/best.pt`
- `runs/self-play-study/alphazero-seed-43/best.pt`

Both were found at update 275, after 35,200 training games. Both final update-300
checkpoints regressed and lost the certificate. The declared checkpoint-selection
rule preserved the earlier passing policies.

We independently verified both saved checkpoints by replaying every opponent
continuation directly through a fresh Rust `Game` and live single-position
PyTorch inference, without the graph evaluator's cached decisions. That verifier
found zero losses in both seats and exactly matched the graph evaluator's leaf
counts. Its results are in `runs/self-play-study/independent-certificates.json`.
We also checked that the seed-42 random-opponent control reproduced the earlier
`ppo-demo` weights exactly.

## A model to play

The seed-43 raw network has fewer off-trajectory action errors than the seed-42
raw network and passes the same starting-position certificate. It needs no search
or lookup of optimal moves during play:

```sh
uv run blokus-rl play runs/self-play-study/alphazero-seed-43/best.pt
uv run blokus-rl play runs/self-play-study/alphazero-seed-43/best.pt --human O
uv run blokus-rl audit runs/self-play-study/alphazero-seed-43/best.pt
```

Its computed results against uniformly random opponents are:

| Seat | Win | Draw | Loss |
| --- | ---: | ---: | ---: |
| X | 96.6146% | 3.3854% | 0% |
| O | 87.3016% | 12.6984% | 0% |

Against an opponent selecting uniformly among perfect moves, it draws 100% of
games as either mark. More importantly, enumeration also rules out losses against
opponents that choose any other legal moves.

This certificate is specific: greedy inference from the normal empty-board
start under our Rust rules. The network still chooses suboptimal moves in 458
of the 4,520 nonterminal positions if arbitrary boards are presented. Those
positions are outside its own reachable decision paths or involve missed winning
opportunities compatible with never losing. We have not certified the stochastic
softmax-sampling policy or optimal exploitation of every opponent mistake.

## What failed or surprised us?

**The PPO self-play curriculum regressed.** Using the declared selected checkpoints,
the mean loss probability against perfect play (averaging seats and seeds) was
84.19% for the random-opponent control and 92.56% for self-play PPO. This specific
80% current / 20% past curriculum did not demonstrate better tactical soundness
within this budget. In seed 42, the checkpoint criterion even preferred the initial
weights to the trained policies. This is a result about this setup, not a general
claim that PPO or self-play cannot solve tic-tac-toe.

**Search-guided learning improved the raw network substantially.** The corresponding
mean perfect-opponent loss was 1.10%; two seeds reached zero losses and passed the
stronger all-opponent certificate. Search supplied move-distribution targets that
our model could learn using ordinary PyTorch cross-entropy and value losses.

**Adding bounded search was not uniformly beneficial.** With 32 simulations per
move, only one of the three selected search-assisted policies passed the certificate.
In seed 43, the raw network was safe from the starting position, but adding search
introduced a losing continuation. The counterexample was independently replayed
through Rust. Search reduced the total number of off-trajectory action errors
while still introducing critical errors on its own game paths. Approximate search
and learned leaf values do not guarantee monotonic improvement in safety.

**The last checkpoint was not the strongest.** The two raw policies that passed
at update 275 failed after another 25 updates. This is why evaluation and preserving
good checkpoints are part of the method rather than optional presentation details.

## What the comparison can establish

The matched PPO arms use the same game budget, architecture, and 2,400 optimizer
steps. Their observed difference isolates this opponent-curriculum change under
those settings. The search-guided arm spends substantially more compute: about
9.2 million training search simulations per run, roughly twice the collected
training positions, and about 4,900 optimizer steps. We did not establish that it
is superior at equal compute or that search alone explains every improvement.

Three seeds are a small reproducibility check. The failed seed and the regressions
are retained in the report. The experiment met its never-lose goal in two
search-guided runs, and did not meet it in the PPO self-play runs.

The [comparison plot](../runs/self-play-study/comparison.png) shows every seed,
including the later regressions. Use the protocol and per-seat JSON data to judge
success, rather than training reward or random-opponent win rate alone.
