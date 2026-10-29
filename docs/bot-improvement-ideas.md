# Three ideas for improving the bot after v1.0

28 October 2026 · Milestone commit: `7a8ed0c`

These are my ranked recommendations, not established improvements. Keep the
52,978-parameter CNN fixed for the first experiments. Our earlier 986,205-parameter
residual model lost to the small CNN at its supplied training budget, while the
recent training replicas showed inconsistent transfer to Pentobi. Better data
and more useful work per dollar are my priorities before another architecture
expansion. [Architecture and results comparison](architecture-comparison.md)

## 1. Learn more geometry from each game with symmetry augmentation

**Hypothesis:** the bot can learn more transferable spatial patterns from its
existing replay by seeing equivalent positions in different orientations.

Our training path currently uses a fixed observer normalization, without random
symmetry augmentation. Seed 1071 evaluated over 102 million neural positions but
created only 114,011 new recorded training roots. Reusing those labels in valid
orientations could improve sample efficiency without additional self-play.
[Evidence](architecture-comparison.md#training-differs-as-much-as-the-network)

**First experiment:** randomly transform sampled replay records during fitting,
keeping the architecture, replay capacity, search budgets, learning rate and
training ratio unchanged. Compare augmentation on versus off from the same
parent and replay source.

The transformation must cover board planes, active starting squares, legal
actions, oriented-shape channels, anchor coordinates and visit targets. Inventory
identities and pass must remain consistent, and the outcome must retain the
correct player perspective. Start with candidate transformations that preserve
the two starting squares: identity, 180-degree rotation and the two diagonal
reflections. Do not assume all eight square symmetries preserve Duo's rules.
Use Rust-generated legal moves and replayed transitions to prove each mapping
and its inverse before using it for training.

**Success:** better fresh-book Pentobi-4 performance at the same playing search
budget and comparable training work, confirmed across training seeds. Lower
training loss alone does not establish success. Augmented copies remain
correlated examples, not independent games. Transformation bugs are the main
implementation risk.

## 2. Test the quality of the 80% of moves we currently search cheaply

**Hypothesis:** 32-simulation fast moves may produce weak continuations and noisy
terminal labels, even when recorded roots receive strong search.

Our current recipe records the 20% of decisions searched at 1,024 simulations;
the remaining 80% use 32. Doubling only the full-search teacher to 2,048 did not
establish an additional Pentobi-4 gain: teacher minus control was −0.25 percentage
points, with a nominal interval of [−7.25, +6.75]. This leaves fast-move quality
as an untested candidate, rather than evidence that more search never helps.
[Controlled comparison](integrated-overnight.md)

**First experiment:** change only `fast_simulations` from **32 to 128**. Retain
the 80/20 mixture, full-search budget of 1,024, recorded-root rule, architecture
and learning settings. Match completed training games in the initial ablation
and report actual simulations, gradients, runtime and provider cost separately.

Requested mean search work per decision rises from
`0.8 × 32 + 0.2 × 1024 = 230.4` to
`0.8 × 128 + 0.2 × 1024 = 307.2`: **33.3% more simulations**, not a guaranteed
33.3% increase in time. At equal compute, this change would instead buy fewer
games, so any improvement must eventually justify that tradeoff.

KataGo's [playout-cap randomization study](https://arxiv.org/html/1902.10565v5#S3.SS1)
explains the tension between policy-target quality and generating enough game
outcomes for value learning. Its Go results motivate testing this balance;
they do not establish the right settings for Blokus.

**Success:** reproducible strength improvement that remains attractive after
accounting for the extra cost. Possible failure modes are reduced exploration
and fewer useful labels per dollar.

## 3. Profile and reduce search–inference coordination overhead

**Hypothesis:** the existing L40S can produce substantially more useful self-play
per dollar by spending less time on host coordination and result handling.

The completed training window averaged **17.34% GPU busy**, **3.63 of 8 CPU
cores**, and actual inference batches of **85.52 out of 256**. Collection took
1,301.78 seconds; fitting took only 10.20. GPU busy is not peak-FLOP utilization,
and spare CPU averages do not identify an exact bottleneck.
[Measured profile](control-replication-performance.md#completed-training-phase-profile)

**First experiment:** benchmark the already implemented result-copy optimization
in a complete fixed-work collection. Its measured 5–32% component-time reduction
has not yet established an end-to-end gain. Then obtain a timeline separating
queue wait, feature packing, staging, transfers, GPU execution, result delivery
and persistence. The existing timing residual is not a direct Volume-commit
measurement.

Use the trace to choose one follow-up: bulk native packing into reusable buffers,
combining ready requests into larger inference batches, or overlapping transfers
and CPU search with double buffering. Test batching delay explicitly rather than
assuming larger batches always win. Preserve simulation budgets, output ownership,
legal-move behavior and checkpoint recovery; asynchronous scheduling need not
produce bitwise-identical games.

**Success:** a repeatable reduction in complete collection time, with actual
games, simulations and neural positions reported alongside cost. A 15% reduction
would be a useful engineering target, not a predicted gain. Strength must still
be tested separately. Our [M2/L40S benchmark](metal-training.md) also points to
self-play inference as the main cost, so optimizing backward alone is unlikely
to move overall training time much.

## Evaluation rules for all three

Before spending on a training ablation, settle a checkpoint-selection protocol
using validation opponents and openings separate from the final test. Existing
selection against one parent on 32 opening pairs was noisy and used a different
playing budget from the Pentobi test. Inspecting old test games is exploratory;
it cannot make those games fresh again. [Replication diagnosis](control-replication.md)

Change one factor at a time, retain a simultaneous control, freeze selections
before generating a fresh final book, and retest the preserved parent on that
same book. Use paired Pentobi-4 draw-half scores at the same 2,048-simulation
playing budget as the primary strength comparison; report raw and searched
head-to-head outcomes separately. An interval crossing zero is inconclusive.
Repeat promising training changes with additional seeds and report them
separately rather than treating all games as independent training replications.
Preserve the current checkpoints and record actual work and costs for every arm.
