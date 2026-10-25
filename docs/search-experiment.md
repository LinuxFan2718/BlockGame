# Blokus Duo search and self-play — 25 October 2026

Dennis authorized this new phase after the overnight PPO experiments. The earlier
stop decision concerned those experiments; it does not settle whether search or
self-play works. This is one rolling experiment note, with machine-readable run
configurations and results under `runs/`.

The main question is whether a network can improve by playing itself with search,
rather than learning only to beat our fixed hand-written opponents. This follows
the broad [AlphaZero learning loop](https://arxiv.org/abs/1712.01815); it is not a
reproduction of that system's architecture, scale or performance.

## What search changes

Without search, the network looks at the board once and chooses its highest-scored
legal move. With search, it tries moves in copies of the Rust game, examines likely
replies, and sends new positions to the network in batches. The search balances
the network's suggested moves, estimated outcomes, and exploration. Terminal
positions use the actual winner. Values change sign on each turn because the two
players have opposing objectives.

During self-play, search records how often it investigated each legal move. The
network learns two things: predict that visit distribution, and predict the final
win/draw/loss from the current player's perspective. These targets come from
self-play. Pentobi and the heuristic bots provide evaluation only. Training adds
noise to encourage exploration; evaluation search is deterministic and noise-free.

The search tree is created from live positions in Rust. It does not enumerate all
possible Blokus positions or use the tic-tac-toe position table. Several games and
leaves share a neural batch; independent CPU processes generate self-play games.

## Preservation and verification

- Starting source is Git commit `896a7e0`. No remote is configured.
- Local recovery archive `.backups/runs-before-search-20261025.tar.gz` contains all
  824 existing run files, about 51 MB compressed. SHA-256:
  `3dc614280eec192d22844e17f04706d7886661f952ec1839c61ba8abe7132d30`.
  This is on the same Mac, so it is not protection against loss of the computer.
- Original certified tic-tac-toe and Blokus checkpoints are preserved.
- Search tests check exact simulation counts, deterministic evaluation, unchanged
  root games, training noise, finite updates to both heads, and independent exact
  minimax answers in a reachable endgame for both players, including forced passes.
- Python suite: 57 passing tests, including a resumed self-play run that produces
  bit-identical final weights after an interruption. Native Rust tests (21) and
  clippy pass.

## Planning without retraining

Frozen policy: `runs/duo-frontier-training-342/best.pt`. Compare search with exactly
the same weights playing greedily. Every independently generated four-ply opening
is played in both seats; intervals resample whole opening pairs. Win scores 1,
draw 0.5, loss 0. These are randomized-opening results, not a proof from every board.

| Set | Simulations/move | Games | W/D/L | Score rate | Paired 95% interval |
| --- | ---: | ---: | --- | ---: | --- |
| Initial screen | 32 | 48 | 23/5/20 | 53.13% | 50.0–58.33% |
| Initial screen | 128 | 48 | 27/5/16 | 61.46% | 54.17–69.79% |
| Initial screen | 512 | 48 | 27/5/16 | 61.46% | 54.17–69.79% |
| Fresh confirmation | 128 | 200 | 116/9/75 | 60.25% | 56.0–64.5% |
| Fresh confirmation | 512 | 200 | 136/14/50 | 71.5% | 66.75–76.5% |

Screen: `runs/duo-search-budget-2541000/`; fresh confirmation:
`runs/duo-search-confirm-2542000/`. The larger screen did distinguish the budgets:
512 gained 11.25 score percentage points over 128 on the same openings, paired
95% interval +6.75 to +16.0 points. The first small screen had missed that gain.

## Independent benchmark: Pentobi

[Pentobi](https://github.com/enz/pentobi) 31.0 is built locally without its GUI,
pinned at `d5374ddc20eced902788e733d67ee16bd1836fd3`. The adapter uses its
[documented GTP interface](https://github.com/enz/pentobi/blob/v31.0/pentobi_gtp/Pentobi-GTP.md).
Matches use one thread, level 1, no opening book, no resignation, and its default
avoid-symmetric-draw setting. A level is an external opponent setting, not a human
rating. No Pentobi moves become training labels.

Two protocol details were verified empirically: Pentobi fixes the first player's
start to E10, so the adapter rotates the board when our first player chooses J5;
Pentobi omits forced passes, so the adapter sends moves with explicit player
colors. A full match compared every legal cell set and the final score margin.
Both engines agreed. The tag's documentation mentions `fixed_simulations`, but
the binary rejects that parameter, so benchmarks use the supported level setting.

On 100 fresh paired openings (200 games, seed 2552000), our raw frontier policy
scored **7.25%**: 12 wins, 5 draws, 183 losses, mean margin −18.395 squares;
paired 95% score interval **4–11%**. This is a much stronger yardstick than our
in-repo heuristics.

A separate, common set of 100 paired openings (seed2551000) compared raw and
128-simulation policies against Pentobi level1. Raw: 10W/6D/184L, **6.5%** score,
mean margin −18.855. Search: 15W/6D/179L, **9.0%** score, mean margin −18.02.
The paired difference is +2.5 score points, 95% interval +0.25 to +5.0 points.
Search helps modestly on this set and remains much weaker than Pentobi. Results:
`runs/duo-pentobi-{raw100,search128}-2551000/`. No human-level claim follows.

The same 200-game external comparison at **512 simulations** finished with
24W/4D/172L, **13%** score (paired95%CI8.5–17.75%), mean margin −15.685.
Compared with the raw policy on identical opening pairs, the gain is +6.5 score
points (paired95%CI+3.25 to +10.0). Increasing search helps against this outside
opponent too, while remaining much weaker than level1. Runtime was 391 seconds;
this is not a matched-compute comparison against Pentobi. Data:
`runs/duo-pentobi-search512-2551000/`.

## Hardware and first training budgets

The Mac has 8 CPU cores and 16 GiB RAM. Metal is available outside the sandbox.
Measured inference includes returning results to the CPU, as actual search needs:

| Batch size | CPU, one thread | Apple GPU (MPS) |
| ---: | ---: | ---: |
| 1 | 0.140 ms | 1.434 ms |
| 8 | 0.263 ms | 1.599 ms |
| 32 | 0.634 ms | 2.123 ms |
| 128 | 2.187 ms | 3.314 ms |

Saved measurements: `runs/search-hardware.json`. CPU inference wins for this
small additive network. This does not settle GPU training or larger CNNs.

Two runs are underway, each **4,096 games**, **128 simulations/move**, 32 rounds,
128 games/round, 3 one-thread self-play workers, 8 live games per worker, 20,000
replay positions, 100 gradient steps/round, minibatch 128, AdamW learning rate
0.0003. Network architecture stays constant to isolate the learning-loop change.

- `runs/duo-az-cold-642/`: random initial network, actual cold-start test.
- `runs/duo-az-warm-643/`: initialized from frontier-342, tests improvement from a
  useful existing policy. This must never be described as a cold start.

Both players always use the current frozen-per-round network plus noisy search.
Sample actions from visits for the first 10 plies, then choose most-visited moves.
Targets use terminal W/D/L, not score-margin shaping. Checkpoints include round
weights, replay, optimizer, RNG state and configuration for resumption. Validation
selects raw policies versus the frozen initial network on 40 paired openings,
every four rounds. The initial policy is retained if validation never exceeds 50%.
This selection criterion alone is insufficient to establish general strength.

After training, freeze the selected checkpoints and test against the original
frontier policy, largest specialist, heuristics and Pentobi on held-out seeds.
Report raw and searched performance separately. Look for an interval above 50%
against the frozen reference and improvement outside that one opponent, without
calling a single small run a general verdict on self-play or architecture.

## Commands

```sh
.venv/bin/blokus-rl duo-play runs/duo-frontier-training-342/best.pt --simulations 128
.venv/bin/python -m blokus_rl.duo_search_study --help
.venv/bin/python -m blokus_rl.duo_pentobi_study --help
.venv/bin/python -m blokus_rl.duo_az --help
.venv/bin/python -m blokus_rl.duo_az_evaluate --help
```

The external engine can be rebuilt without installing a GUI or system package:

```sh
git clone --depth 1 --branch v31.0 https://github.com/enz/pentobi.git .local/pentobi
cmake -S .local/pentobi -B .local/pentobi-build -DCMAKE_BUILD_TYPE=Release -DPENTOBI_BUILD_GUI=OFF -DPENTOBI_BUILD_GTP=ON
cmake --build .local/pentobi-build --target pentobi-gtp --parallel 4
```

Full reproducible arguments are stored in each run's `protocol.json`. Current
jobs, failures and next steps belong in `docs/search-status.md`.
The regenerable compact [results report](search-results.md) and figure distinguish
completed planning benchmarks, validation learning curves, and pending held-out
tests. Update with `.venv/bin/python -m blokus_rl.duo_search_report`.

Statistical limit: percentile bootstrap intervals describe variability among
sampled openings. When every opening has the same outcome they collapse; that
does not prove the unseen population has exactly that score. Report counts and
the sample size, and do not treat a degenerate small-sample interval as certainty.
