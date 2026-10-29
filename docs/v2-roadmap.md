# Blokus Duo agent: v1.0 → v2.0 roadmap

Started 28 October 2026 from v1.0 (`7a8ed0c`). Goal: the strongest Blokus Duo
agent we can train with the remaining Modal credits, climbing the Pentobi
levels one by one.

## Where v1.0 stands

- **Strength.** At 2,048 search simulations per move, v1.0 scores about 58%
  against Pentobi level 3 and 38–44% against level 4. Pentobi has nine levels.
  ([level 3](pentobi-level3.md), [level 4](pentobi-level4-5.md),
  [replication](control-replication.md))
- **Network.** 52,978 parameters: four 3×3 convolutions of width 32. Each output
  sees only a 9×9 patch of the 14×14 board. The only learning signal per
  position is one bit (the final win/draw/loss).
- **Data.** Its whole lineage trained on well under a million searched positions.
  AlphaBlokus's published network is about 17× larger and appears to have
  trained on about 26 million samples.
- **Throughput.** On the L40S the GPU is busy about 17% of the time, with an
  average batch of 85 out of 256. A local run with a stand-in network measured
  a mean request of 61 positions; 89% of positions arrived in requests smaller
  than 128. The causes:
  1. Fast and full searches take turns: during a 1,024-simulation full search,
     only the ~20% of games doing full search take part.
  2. Each worker has a single request in flight.
  3. Python runs once per searched position.
- **Method.** Most of the $56 spent so far went to small 16k-game continuations,
  each run as a careful paired study. Their effects are at the noise floor: one
  run gained 13.75 points against level 4, and its replication gained −1.25.

## Budget rules

- Spend only Modal credits. Metered cost was $55.56 of $130 at the start, so
  about $74 remained. **Hard cap: total metered cost stays below $118**, which
  leaves a margin for delayed metering.
- Check `modal billing summary` before and after every paid job, and record
  costs in this document.
- Do local development, tests, Pentobi data generation and ladder evaluation
  on the Mac for free.
- Anything that needs hundreds or thousands of dollars of GPU time is described
  under "Beyond the budget" below, not implemented.

## Phases

### Phase 1: Rust-native self-play engine

Move the whole self-play loop into Rust:
- many concurrent games, each searching independently on its own schedule (no
  lock-step between fast and full searches)
- multi-threaded leaf selection and expansion
- a batch interface where Python only runs the network

Python overlaps GPU inference for one half of the games with CPU search for the
other half (double buffering).

Also in this phase:
- **Search fixes:** first-play urgency (FPU) reduction, so an unexplored move is
  valued at the parent's value minus a margin instead of a draw; reuse of the
  search tree between moves; configurable fast/full simulation counts.
- **Richer rules-derived inputs:** own/opponent occupancy, attachment corners,
  forbidden cells and the cells each side's legal moves can cover, plus start
  squares and inventories.
- **Symmetry tables** for the four symmetries that preserve both starting
  squares: identity, 180°, and both diagonal reflections.

**Success:** Rust tests prove the feature, policy-index and symmetry mappings.
A local benchmark shows much larger inference batches and more positions per
second than the v1 collector on the same Mac.

### Phase 2: Exact endgame solver

An alpha-beta solver over the exact final score margin, with a transposition
table, move ordering and a node budget. Used in three places:
- self-play: perfect endgame moves and exact labels
- evaluation and app play
- search leaves, where cheap

**Success:** matches brute-force minimax on random late positions; reports how
early in a typical game it solves within budget.

### Phase 3: v2 network and training loop

- **Network:** a residual CNN with global pooling and repeated inventory
  conditioning.
- **Heads:** policy; win/draw/loss value; final score margin; final ownership of
  every cell.
- **Training:** symmetry augmentation; exponential-moving-average (EMA) weights
  for play; a replay window that grows with the data; checkpoints that resume
  without re-saving the whole replay every round.

**Success:** a local Metal smoke run learns: losses fall and it beats the random
and largest-piece baselines.

### Phase 4: Parallel Pentobi ladder (local, free)

Many concurrent games against Pentobi on the Mac's cores, with batched network
inference and the endgame solver. A level counts as cleared at ≥55% score over
at least 200 games (100 opening pairs with colors swapped).

**Success:** reproduces v1.0's known level-3/level-4 results within noise, at a
fraction of v1's evaluation time.

### Phase 5: Imitation of Pentobi

Generate Pentobi-vs-Pentobi games: on the Mac for free, plus cheap Modal CPU
containers if useful. Train the v2 network to imitate them: Pentobi's move,
final result, score margin and final ownership.

**Success:** the imitation network beats v1.0 head-to-head and on the ladder.

### Phase 6: Throughput benchmark on Modal

A short paid benchmark (about $1–3) of the new engine with the v2 network.
Choose the GPU type (L4 vs L40S) and CPU core count that give the most
self-play positions per dollar.

### Phase 7: Self-play reinforcement learning on Modal

Start from the imitation network and run one long, resumable self-play run with
everything above. Copy checkpoints to the Mac and run the Pentobi ladder there
as the run progresses.

**Success:** clears Pentobi levels beyond v1.0's level 3.

### Phase 8: v2.0 release

The local app plays the strongest v2 agent with the endgame solver. Final ladder
results, honest limitations and next steps go here.

## Beyond the budget

This section is filled in as the phases reveal which large-compute strategies
look most promising.

## Phase results

Each phase adds its outcome here after it finishes, successful or not.

### Phase 1 result: Rust self-play engine — success

28 October 2026. No Modal spend.

**What was built.**
- `src/duo_engine.rs` runs hundreds of concurrent games, split into two groups.
  - `prepare(group)` advances every game in a group in parallel with rayon. Each
    game searches until it needs network evaluations, moves as soon as its own
    search finishes, and a finished game is replaced immediately.
  - `apply(group, …)` backs up the results.
  - Python only evaluates the network, overlapping one group's evaluation with
    Rust searching the other (`python/blokus_rl/v2/engine.py`).
- `src/duo_mcts.rs` is a new search tree:
  - stores only the root position and replays paths with trusted moves
  - first-play urgency (FPU) reduction: an unexplored move is valued at the
    parent's value minus 0.25 × √(prior mass already visited), instead of a draw
  - keeps the chosen subtree between moves
  - backs up exact values for finished (and later, solved) positions without
    calling the network
- `src/duo_v2.rs` builds the inputs: nine rules-derived planes, in the mover's
  frame:
  - own and opponent occupancy
  - attach corners
  - forbidden cells
  - cells reachable by each side's legal moves
  - an on-board plane

  It also builds 48 scalars (both inventories, scores, passes, seat, and the log
  of each side's legal-move count) and the tables for the four board symmetries
  that keep both start squares.
- `src/duo_replay.rs` stores games as move lists plus visit counts. Features,
  targets, final ownership and margins are rebuilt at sampling time under a
  random symmetry, so the whole replay checkpoints in a few bytes per move.
- `python/blokus_rl/v2/model.py` is the v2 network: residual blocks with
  KataGo-style global pooling and scalar conditioning in every block. Its heads
  are the policy (17,837 outputs), win/draw/loss, score margin and per-cell
  ownership. At 6 blocks × 64 channels it has 571k parameters.

**Correctness.** 13 Rust unit tests, including proofs of:
- the symmetry tables against cell geometry
- encodings under all four symmetries
- each feature plane against its rule definition
- the policy-index frame
- tree visit accounting and virtual-loss cleanup
- subtree reuse
- engine games replaying to the same final scores
- external seats
- replay serialization round-trips

There are 4 new Python tests, and all 118 earlier Python tests still pass.

**Move generation.** Generating legal moves from a compact, cache-friendly
per-cell placement table, and reading the legal set from a bitset (already
sorted), raised single-core speed from 22.9k to 53.6k positions/s. That includes
generating the opponent's moves for the new mobility planes. The existing
test comparing move generation against a brute-force scan still passes.

**Throughput (M2 MacBook, 8 cores).** A stand-in network with uniform priors
isolates the CPU side:

| Setup | Positions/s | Mean batch |
|---|---:|---:|
| v1 collector, one worker (1 core), v1 settings | 31k | 61 |
| v2 engine, 1 core, same v1 settings and game count | 53.6k | 105 |
| v2 engine, 8 cores, 512 games, v2 settings (128 fast / 1,024 full at 25%) | **233k** | **567** |
| v1 complete pipeline on the Modal L40S, for reference | 66–79k | 85 |

With the real 6×64 network on the M2's Metal GPU, end-to-end self-play ran at
5.3k positions/s. The network alone runs at about 4.7k/s, so the loop keeps the
GPU busy and the M2 GPU is simply the limit. A 4×32 network ran at 9.8k/s.

Two Metal-specific slowdowns were found and fixed:
- mean/max over two dimensions was about 20× slower than flattening first
- `index_add_` was about 30× slower than a padded softmax

**Conclusion.** The CPU side is no longer the bottleneck: the Mac alone searches
3× faster than v1's whole L40S pipeline. On a CUDA GPU the network becomes the
limit, which is the right problem to have. Phase 6 measures this on Modal.

### Phase 2 result: exact endgame solver — success

28 October 2026. No Modal spend.

**What was built.** `src/duo_solver.rs` is a negamax alpha-beta search over the
exact final score margin (mover minus opponent) to the end of the game:
- Moves are ordered by the transposition-table move first, then larger pieces.
- The transposition table is Zobrist-hashed and keeps proven bounds. Each game
  owns a 1 MiB table, so later solves in the same game reuse earlier proofs.
  Searches that run out of budget store nothing.
- Maximizing the margin also maximizes the result: if a win exists, the best
  margin is positive.

The engine uses it in two ways:
- **Root solve:** when the mover's and the opponent's legal placements together
  are within a gate, the solver picks the move outright and the position is
  recorded with the proven move as its target.
- **Leaf solve:** solved search leaves back up exact values instead of calling
  the network.

Python can call it directly through `duo_solve`.

**Correctness.**
- It matches plain minimax (no pruning, no table) on 60 random late positions,
  and the returned move achieves the value.
- A Python test checks 18 more positions against an independent Python minimax.
- An engine test confirms solved moves are recorded and every game replays to
  the same score.

**Reach** (game-like positions, 2M-node budget):

| Legal placements, both sides | Proved | Median nodes | Slowest |
|---|---:|---:|---:|
| < 10 | 58/58 | 2 | 2 ms |
| 10–19 | 40/40 | 118 | 7 ms |
| 20–39 | 42/42 | 19k | 1.3 s |
| 40–59 | 13/19 | 86k | 1.2 s |

In practice that covers roughly the last 10–15 plies of a game.

**Cost in self-play** (512 games, stand-in network, 8 cores):

| Setting | Games/s | Moves solved |
|---|---:|---:|
| Solver off | 28.5 | 0 |
| Root gate 20 placements, 100k nodes | 24.4 (−14%) | 17% |
| Root gate 30, 300k nodes | 16.8 (−41%) | 21% |
| Root 30 + leaves (20k nodes) | 1.1 (−96%) | — |

Leaf solving at every eligible search leaf is far too expensive. Long solves
also stall a whole batch group, because `prepare` waits for its slowest game.
**Self-play uses root solving only (gate 20, 100k nodes).** Evaluation and the
app can afford larger budgets.

**Strength check.** Same search (uniform priors, 256 simulations), with versus
without the root solver (gate 24, 300k nodes): over 400 games (200 openings,
colors swapped) the solver side scored **61.4%** (paired 95% interval
58.5–64.3%). Against a strong network the gain will be smaller, but the solver
never makes an endgame mistake.

### Phase 3 result: v2 network and training loop — success

28 October 2026. No Modal spend.

**What was built.** `python/blokus_rl/v2/train.py` runs self-play and learning
in one process:
- The engine plays continuously with the EMA network. Between engine batches
  the learner takes gradient steps, so each recorded position is sampled about
  `train_ratio` (4) times.
- Batches come from the newest half of the replay (at least 100k positions),
  each position under a random board symmetry.
- **Losses:** policy cross-entropy over legal moves, win/draw/loss
  cross-entropy, score margin (smooth L1 on margin/20, weight 0.5) and per-cell
  final ownership (weight 0.5).
- **Optimizer:** AdamW with gradient clipping. An exponential moving average of
  the weights (decay 0.995) plays self-play and is what gets saved.
- **Search value** = P(win) − P(loss) + 0.05·tanh(predicted margin / 20).
- **Checkpoints** hold weights, optimizer and the compact replay (5.5 MB for
  60k positions; 0.08 s to save). Runs resume with `--resume`, and a Modal
  Volume can be committed after each checkpoint.

`python/blokus_rl/v2/arena.py` plays a network against any outside mover
(baselines, Pentobi), with each random opening played from both seats.

**Tests.** Two new Python tests:
- a small network overfits a fixed batch (policy, value and ownership losses
  all fall)
- checkpoint and replay round-trip exactly

**Local smoke run.** M2 Metal, 15 minutes, a small 4×32 network from random
weights, reduced search (128 full / 32 fast):
- 4,919 games at 8–9 games/s and 12.5k positions/s, 852 gradient steps
- Policy KL fell from 1.40 to 1.07, value loss from 0.71 to 0.51, ownership
  loss from 0.74 to 0.35
- About 17% of moves were played by the exact solver

| Opponent (64 sims, solver off, 100 games) | Untrained | After 15 min |
|---|---:|---:|
| Random mover | 68.5% | **100%** (100–0) |
| Largest-piece baseline | 22.0% | **100%** (100–0) |

For context, v1's trained raw policy scored 80% against largest-piece.

A first Pentobi check of the same smoke network at 128 simulations (endgame
solver on):
- **62.5%** against level 1 [55.5, 69.5]
- **54.5%** against level 2 [46.0, 63.0]

v1 needed several GPU training studies to reach 62.5% against level 1 and
42.0% against level 2 at 128 simulations.

A bug found along the way: elapsed time was double-counted at each checkpoint,
which made runs stop early. It is fixed.

### Phase 4 result: parallel Pentobi ladder — success

28–29 October 2026. No Modal spend.

**What was built.** `python/blokus_rl/v2/ladder.py` plays a v2 checkpoint
against Pentobi 31.0 (`--nobook --noresign`, one thread) at chosen levels:
- Many games run at once in the Rust engine, each with its own Pentobi process.
- Pentobi moves are generated in a thread pool while the engine keeps searching
  the other games.
- Openings are four random plies, each played from both seats; draws count half.
- The intervals are paired bootstrap intervals over openings, reusing v1's
  `paired_interval`.
- Root endgame solving is on by default (gate 30, 2M nodes).

Command:
`.venv/bin/python -m blokus_rl.v2.ladder CHECKPOINT --levels 3 4 5 --pairs 100 --simulations 800`.

**Correctness checks, on every game:**
- every Pentobi move must be legal in our engine
- seats alternate per opening
- after the run, every game is replayed into a fresh Pentobi process, and
  Pentobi's `final_score` must equal ours

I changed the planned success test. The plan called for reproducing v1's
level-3/4 results in the new harness, but v1's network cannot run in the v2
engine: the inputs and search differ. The per-game score agreement with Pentobi
is a stronger check of the harness. Re-running v1 would only re-measure an
already published result.

**Speed.** On the Mac, 100 games take 15 s at levels 1–2 (128 sims), and about
80 s at levels 3–5 (400 sims, while sharing the CPU with Pentobi data
generation). v1's harness took 341–404 s for 200 games on Modal.

Measured Pentobi time per game (both sides, one M2 core):

| Level | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| Seconds per game | 0.05 | 0.07 | 0.16 | 1.2 | 22 | 67 |

So levels 1–7 can be evaluated cheaply on the Mac; level 8 needs about 2 s per
Pentobi move.

**First ladder numbers** (the 15-minute Phase 3 smoke network, 4×32, 400
simulations, 100 games per level):

| Level | Score | Interval |
|---|---:|---:|
| 3 | 45.0% | [36.0, 54.0] |
| 4 | 22.5% | [16.0, 29.0] |
| 5 | 18.0% | [11.0, 25.5] |

These are a calibration point for an undertrained laptop network, not a v2
result.

### Phase 6 result: Modal throughput benchmark — success

29 October 2026. Benchmarks and image builds cost about $1.70. Phase 5's
Pentobi data job was running in parallel. Run out of numeric order because the
Phase 5 data was still being generated.

**Infrastructure.**
- `cloud/modal_v2.py` builds one image (Rust 1.91.1, Pentobi 31.0 from
  source, PyTorch 2.14.1) with CPU data, GPU benchmark, imitation and training
  functions, all writing to the `blokus-v2` Volume.
- `cloud/v2_budget.py` refuses any launch whose worst case (hourly rate ×
  timeout) would take total metered spend past $118. That keeps spending inside
  the $130 of credits.

**Benchmark rounds** (6×64 network unless noted, bf16, random weights,
v2 search settings):

| Round | Hardware | Positions/s | Games/s | What limited it |
|---|---|---:|---:|---|
| 1 | L40S, 32 cores | 74.5k | 8.9 | Serial Python glue between batches |
| 2 | L40S, 32 cores | 72k | 8.7 | All phases serial in one thread; 48 rayon threads on a smaller CPU quota |
| 2 | L4, 16 cores | 42k | 4.8 | GPU |
| 3 | **L40S, 32 cores** | **130k** | **15.9** | Balanced |
| 3 | L4, 16 cores | 51k | 5.9 | GPU |
| 3 | L40S, 32 cores, 8×96 network | 75k | 8.8 | GPU |

**Fixes found by measuring.**
1. The first image compiled Rust with `target-cpu=native`, and workers on other
   CPUs died with SIGILL. It now uses the generic target.
2. Index arrays are built on the GPU, plain copies replace per-batch pinned
   allocations, and logits are gathered before widening to float32.
3. Each slot group now has its own lock. Two Python threads drive the two
   groups, so one group's Rust search and backup overlap the other's GPU work.
   Waiting on the GPU polls a CUDA event, so the GIL stays free.
4. Batches are padded to multiples of 256 and cuDNN autotuning is on.
5. Rayon's threads are capped at the container's cgroup CPU quota.

**Choice for Phase 7:** an L40S with 32 cores ($3.72/hour) and the 6×64 network.

| Hardware | Cost per 1,000 games |
|---|---:|
| L40S, 32 cores | $0.065 |
| L4, 16 cores | $0.086 |

The 8×96 network costs 1.8× as much per game. With about $50 left for
reinforcement learning, that would buy roughly 380k games instead of about
700k. Since v1's whole history was about 100k games, data volume matters more
than capacity at this budget.

For comparison, v1's production pipeline ran 66–79k positions/s at 17% GPU
busy on the same L40S, with a network 10× smaller.

### Phase 5 result: imitation of Pentobi — success

29 October 2026. Modal spend: level 7/8 data about $5 (job still running at the
time of writing) and imitation training about $0.40.

**Data** (`python/blokus_rl/v2/pentobi_data.py`). Pentobi plays both colors,
one single-threaded process per worker. The first 0–3 plies are random and every
game gets a fresh seed, for variety. Every game's final score is checked against
Pentobi's.
- **Level 6 on the Mac (free):** 21,769 games in 90 minutes with 7 workers.
- **Levels 7 and 8 on a Modal CPU container** (32 cores, 60 workers, $1.64/hour,
  3 hours): about 31 s and 118 s per game per worker. It had produced 4,408
  games when imitation v0 started and was still running.

Measured on one M2 core, level 7 takes about 22 s per game and level 8 about
67 s. Level 9 was not used: its cost per game is far higher.

**Training** (`python/blokus_rl/v2/imitate.py`, L40S, 7 minutes). Targets:
- Pentobi's chosen move (one-hot)
- the game's win/draw/loss and score margin
- the final owner of every cell

Positions were sampled under random symmetries: 15,000 steps of 1,024, about
30 passes over 508k positions from 16,324 games, with 300 games held out.
- **Held-out move match: 38.4%.** The network's top choice equals Pentobi's
  move on 38% of positions; Pentobi itself is stochastic.
- **Held-out value loss** was best at step 8,000 (0.522) and then drifted
  (0.55 at the end) while training loss kept falling. Positions from the same
  game are correlated, so the value head overfits first. `best.pt` keeps the
  best-value checkpoint.

**Ladder** (final imitation weights, 800 simulations, endgame solver on,
100 games per level):

| Pentobi level | Score | Interval | v1.0 for comparison |
|---|---:|---:|---|
| 4 | **89.0%** | [83.0, 94.5] | 33.75–44% at 2,048 sims |
| 5 | **80.5%** | [73.5, 87.0] | not reached |
| 6 | **69.5%** | [63.0, 76.0] | — |
| 7 | 53.5% | [46.0, 61.0] | — |

Imitation alone, mostly of level-6 games, plus search and the endgame solver
clears levels 4–6 and is about even with level 7. v1.0's ceiling was between
levels 3 and 4. This network (`/vol/imitation/v0/init.pt`) seeds Phase 7.

### Phase 7 result: self-play reinforcement learning on Modal — success

29 October 2026. One run, `rl-a` (Modal Volume `blokus-v2`, `/runs/rl-a`), on an
L40S with 32 cores, started from imitation v0 and resumed in four chunks.
Fixed throughout:
- the 6×64 network
- root endgame solving (gate 20 placements, 100k nodes)
- 800 full / 128 fast simulations, with 25% of moves getting a full search
- first 10 plies sampled from the visit counts
- each recorded position trained on about 4 times, in batches of 512
- EMA decay 0.995

| Chunk | Hours | Games at end | Changes |
|---|---:|---:|---|
| 1 | 2.0 | 154,780 | Self-play from the empty board; Pentobi data 15% of each batch; learning rate 5e-4 |
| 2 | 3.4 | 440,779 | Games start after 0–4 random plies; Pentobi data 20% (all 32,358 games); learning rate 3e-4 |
| 3 | 3.0 | 697,588 | Pentobi data 10%; learning rate 2e-4 |
| 4 | 1.5 | **823,673** | Learning rate 1e-4 |

Totals: 8.5 million recorded positions and 74k gradient steps. Throughput was
21.5–24.6 games/s and 153–161k positions/s. RL compute cost about $37 of
credits; Modal metering for the whole v2 effort reached $103.83, against
$55.56 at the start.

**What went wrong in chunk 1, and the fix.** Losses fell (policy KL
1.06 → 0.84), but the first player's self-play win rate climbed from 77% to 92%.
Duo has a large first-move advantage: in Pentobi-vs-Pentobi games without
random openings, the first player scores 83.5%, 90.5% and 95% at levels 6, 7
and 8. As self-play got stronger, nearly every second-player position looked
lost, so the search had little to distinguish and that side's training targets
degraded.

Against level 7 (same 50 opening pairs), RL improved first-seat play from 82%
to 86–87% but cut second-seat play from 25% to 17%, for no net gain. A 10×
stronger score-margin term at play time made it worse (47%).

From chunk 2, self-play games start after 0–4 random plies, matching the
ladder's openings and Pentobi's data. The first-player rate settled near 65%,
and second-seat play recovered.

**Progress on fixed openings** (seed 7,000,000; 800 simulations; 100 games per
level test; head-to-head at 200 simulations over 200 games):

| Network | vs Pentobi 8 | Change vs imitation v0 (paired) | Head-to-head vs v0 |
|---|---:|---:|---:|
| imitation v0 | 37.5% | — | — |
| RL, 234k games | 45.0% | +7.5 [+0.5, +15] | 58.0% [53.75, 62.5] |
| RL, 315k | 51.0% | +13.5 [+5.5, +22] | — |
| RL, 441k | 50.0% | +12.5 [+6.5, +19] | 61.0% [56.75, 65.5] |
| **RL, 824k (release)** | see Phase 8 | — | **65.25% [61.0, 69.75]** |

That is about +110 Elo from reinforcement learning over the imitation network.
Successive checkpoints scored close to 50% against each other:

| Pair | Score |
|---|---:|
| 441k vs 315k | 53% |
| 569k vs 441k | 50.5% |
| 655k vs 441k | 50.25% |
| 824k vs 698k | 51.75% |

So the network largely plateaued after about 450k games at this size.

**Level 7 is noisy, level 8 is not.** Against level 7, every network measured
50–55% on these openings, from imitation through RL at 234k games. A 100-game
test resolves only about ±6 points; improvement showed up against level 8 and
head-to-head.

**Play-time variants that did not help** (imitation v0 against level 7):
- 3,200 instead of 800 simulations: +2 [−7, +11.5]
- averaging the network over the 4 board symmetries: −4 [−13, +4.5]

At this network size, the quality of its evaluations, not the amount of search,
sets the ceiling.

**First look at level 9** (441k network, 20 games): 45% [35, 50], winning 85%
from the first seat and 5% from the second. Pentobi took about 28 s per move.

**New tools added in this phase:**
- `blokus_rl.v2.arena` for network head-to-head
- `blokus_rl.v2.compare` for paired comparisons on the same openings
- `--opening-random-plies` in training
- `--symmetries` for ensemble evaluation
- schedule changes allowed on resume
