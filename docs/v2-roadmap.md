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
