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
