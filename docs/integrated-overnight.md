# Integrated speedups and fresh controlled run — 28 October 2026

The requested overnight run did **not** start overnight. At 08:08 EDT on
28 October, integration and local verification were complete, but GPU validation
was still pending. This report records the actual launch times and outcomes;
it must not imply that unstarted training ran while Dennis slept.

## Implemented and locally verified

Rust legal move generation now uses corner-indexed placements and retains the
original scan as a correctness oracle. PUCT selection caches its best priority.
Native search can return compact policy indices directly, removing repeated
Python geometry conversion at leaf requests. Shared inference supports block
staging, direct shared-memory replies, and immediate draining of ready requests.
CUDA Graph inference uses persistent buffers and caller-owned outputs, with fresh
captures after optimizer updates. Both training and frozen evaluation can opt in.

Local verification: Rust tests passed; original and integrated native extensions
matched on 32 complete random games / 964 positions and 12 arbitrary-prior search
roots / 3,072 simulations per implementation. All 96 Python tests passed (seven
existing warnings). The initial sandboxed test run lacked POSIX shared-memory
permission; the full suite passed with that access. Logs and original extension
are preserved in `runs/integrated-speedups-baseline-20261028/`.

Caching remains disabled: the earlier measured neural cache slowed collection.
Channels-last remains unselected because its previous collection improvement was
small and variable. Prior measurements favored the L40S over H100 for this small
network and host-driven search. Saturating an H100 in a GPU-resident microbenchmark
does not imply that the complete game/search pipeline saturates it.

## GPU validation before training

One source-only payload (19 allowlisted files, no model/replay upload) is frozen
under `runs/integrated-payload-1070/`. Linux builds use the immutable base image's
locked Rust manifests. The validation compares the original and integrated native
extensions, checks eager versus captured outputs before and after two discarded
optimizer steps, and tests exact single-worker checkpoint recovery on two
unpromoted eight-game smokes with a 256-position replay. This is a narrow recovery
check, not a claim of bitwise recovery across asynchronous worker scheduling.

The frozen-model throughput experiment tests legacy rows, compact rows with ready
batching, and compact bulk staging, each at full search 1,024 and 2,048 with two
reversed-order repeats. Every case uses 12 warm-up and 256 measured games, six
workers, 64 live games per worker, shared inference batch256, leaf8, zero wait,
CUDA bfloat16 and TF32. All variants use the validated Rust core and CUDA Graphs.
Select the lowest combined mean collection-plus-capture time among variants no
slower than legacy at either search budget; otherwise retain legacy transport.
No test score or learned checkpoint enters that choice.

## Fresh matched study after validation passes

Run `duo-integrated-teacher-1070` starts both branches from the preserved selected
round112 parent CNN (52,978 parameters), SHA256
`95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`.
Both inherit the same full 100,000-position replay from round128 state SHA256
`df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911`.
Both reset AdamW because the later optimizer moments do not match selected weights.
The intentionally stopped older study and all its checkpoints stay preserved.

Both branches plan64 rounds ×256 =16,384 additional games, fresh seed1070. Control
uses1,024 and teacher2,048 simulations on the recorded20% full-search roots; the
other80% use fast32. Same runtime and all other scientific settings, six workers
×64 live games, inference256, leaf8, zero wait, replay100000, AdamW0.001 with weight
decay0.0001, minibatch256, training ratio3, policy CE weight0.5 plus terminal WDL
MSE, gradient clip1. Soft training caps are6,000 and10,000seconds. Report unequal
actual completed games/experience and compute honestly if either cap binds.

Validation every16 rounds uses32 paired openings, seed2607300, versus the frozen
parent at512 simulations each. The unchanged parent is selectable at50%; it is
additional round0 even though its embedded update is112. Freeze selected files
before generating the held-out book. Test seed2607900 has100 fresh four-ply opening
pairs and both seats. Parent/control/teacher each plays pinned Pentobi31.0 Duo
levels3 and4 at2,048 playing simulations (six cases /1,200 external games with
full action traces). Teacher-control search2048 and raw, teacher-parent raw and
control-parent raw add800 games with paired outcomes/counters, without traces.
Finish all prespecified cases regardless of intermediate scores. No next level,
extra search, new architecture or additional training is automatic.

Primary success is teacher-minus-parent paired Pentobi4 score difference with a
nominal whole-opening95% interval entirely above0. Teacher-minus-control and
head-to-head address whether the additional training search helped beyond more
training. Absolute teacher Pentobi4 interval entirely above50% is a stretch goal.
Crossing0 is inconclusive, below0 negative evidence. Report raw/search separately,
validation/test separately, WDL, win rate, draw-half score, margin and actual work.
One parent/training seed, inherited later replay, selection, unequal compute and
multiple nominal intervals limit generalization; no human Elo, Classic, frontier
or AlphaBlokus claim follows.

## Costs, safeguards and completion

Validation has one1,200-second L40S function; study one28,800-second (8h) L40S
function, each8 capped cores /24GiB, retries0, max1container. Rate$2.5204/hour.
Validation reservation includes90s startup and$0.15 build allowance. Combined
reservation ceiling$23; keep$50 of the original$130 existing credits. CLI billing
before launch was$46.03137374 metered; the launcher conservatively floors prior
commitment at$46.25. No top-up or plan change. Ledger
`runs/modal-integrated-cost-ledger.jsonl` is authoritative; reservation is not spend.

Exclusive remote directories refuse duplicate or infrastructure replay. Atomic
checkpoints and pairs commit incrementally. Failure preserves state and costs;
no automatic repeated launch. Final audit must verify source/frozen/model hashes,
all training counters, optimizer/replay lineage and selection; replay all1,200
external traces, recompute paired summaries and validate head-to-head counters
without claiming head-to-head action replay. Download only after finished markers
and stopped app0tasks into an unused pre-created local parent. Verify every backup
archive member by readback, inspect billing/stopped apps, commit validated local
code/docs without pushing, report results here and delete the follow-up.

First submission `ap-KwsyrmMhvZpLz8Xy9P9LSy` built the Linux extension but the
prelaunch app guard then counted its own zero-task submission app and refused to
spawn a GPU call. No validation/training directory or GPU function was created.
The guard now excludes its own app; the other-active-app check remains. Original
log is retained with `-prelaunch-guard` suffix and any image-build metering remains
part of this phase's costs. The source-only payload and scientific protocol did
not change.

The first actual GPU submission `ap-tUfMbBe5gfBIYvlmZ3wD25` failed during Modal's
module import (`ModuleNotFoundError: modal_integrated`), before the helper or run
directory existed. It was explicitly stopped. Its original reservation of
$1.05314333 remains counted in full, along with all logs and costs. The corrected
runner uses the established `/opt/blokus/cloud` path with explicit PYTHONPATH
and an image-build import check before GPU allocation. The allowlisted
`validate-import` stage permits this single corrected submission; its own duplicate
guard still refuses repeats. Including both validation reservations and the main
8h reservation gives$22.33249, below the unchanged$23 cap. No scientific payload,
model, replay or study setting changed.

The new image-build check initially tried to import Modal itself, but the SDK is
injected into function containers and is absent in the base image builder. This
was caught before GPU allocation/submission; the check now verifies Python module
discovery and compilation without depending on the injected SDK. Its build log is
retained with `-build-check` suffix. All build costs remain included.

App`ap-CioAB9zwQl4m3qfaDzvsrf` passed the complete native comparison, then its
subprocess lacked `/pkg`, the runtime-injected SDK path, because the wrapper
replaced PYTHONPATH. It failed before numerical checks, smoke games or training.
The wrapper now preserves its runtime Python paths behind the explicit overlay
and verifies child SDK import **and mounted Volume commit** before scientific
work. Failed artifacts remain under the original validation directory; the
single corrected `validate-sdk` stage uses unused directory
`duo-integrated-validation-1070-sdk`. The payload remains unchanged. A ledger
settlement keeps$1 reserved for all stopped startup attempts and releases only
their unused reservation. With the corrected validation and full8h run, net
reservations will be$22.27935 under the original$23 cap. Actual charges for every
attempt are reported separately and are never erased by settlement.
