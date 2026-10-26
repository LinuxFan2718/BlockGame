# Deep CNN and batched self-play — overnight study

Dennis explicitly authorized the new phase on25 October2026 after the architecture
study: save the explanation, try a deep CNN and batched self-play, use Modal if
helpful, and continue overnight. Existing certified tic-tac-toe and all earlier
Blokus checkpoints are preserved. This supersedes the completed-study stop for
new experiments, while leaving all completed earlier runs untouched.

**Completed:** see [results and plot](deep-results.md). Both models finished their
full 32,768-game budgets; the smaller CNN was stronger in the held-out comparisons.
This protocol is preserved as the record of the design, not an instruction to
launch another run. The rolling completion audit is in [deep-status.md](deep-status.md).

## Hypothesis and design

A10-block64-channel residual CNN, inventory conditioning and shared GPU inference
may support stronger cold-start self-play than our4-layer32-channel CNN. Start
both networks randomly; no Pentobi or heuristic demonstrations or rewards.
The Rust engine owns legality and terminal WDL. This is inspired by AlphaBlokus,
not a reproduction or a claim to have reached its strength.

Both architectures use the same game RNG seed764, search and optimization setup:
planned128 rounds×256 games (32,768 games/architecture),6 CPU search workers,
32 concurrent games per worker, up to256 positions/GPU batch,32 simulations on
80% of moves and256 on20%. Root noise only on full-search moves; sample moves
for the first12 plies. Record only full-search positions and final WDL targets.
Replay100,000 positions, training ratio3 examples/new recorded position,
minibatch256, AdamW lr0.001/weight decay0.0001, policy CE weight0.5 plus WDL MSE.
Clip gradient norm1. GPU bfloat16 autocast, float32 losses/softmax, permitted TF32.

The residual model uses GroupNorm to avoid running batch statistics and reduce
sensitivity to asynchronously assembled inference batches. This differs from
AlphaBlokus's BatchNorm. Both spatial models share the same fixed orientation
lookup and existing normalized634-element observation. Indexed legal moves are
computed on CPU; only observations and canonical move indices cross to the GPU.
Weights stay frozen for each collection round; the GPU trains between rounds.

Search processes share a queue feeding the parent GPU evaluator. No worker owns
a CUDA model. Replies are routed by worker/request IDs. Replay merges in fixed
worker order. Seed/config reproducibility is recorded; asynchronous GPU floating
point batches are not promised bit-identical across hosts/scheduling.

## Controls, selection and held-out evaluation

Every8 rounds, compare searched policy against the frozen original frontier342,
both128 simulations/move,24 paired validation openings, seed2563000. Select each
architecture's best validation checkpoint. This small selection set is validation,
not independent evidence. Freeze and hash both selected checkpoints before test.

Primary held-out comparison: deep search128 versus shallow search128 on100 fresh
four-ply opening pairs (200 games), seed2569000. Report W/D/L, score=win+halfdraw,
whole-pair bootstrap95% interval, actual game/simulation/gradient counts and cost.
A lower confidence bound above50% supports a deep-model gain for this seed only.
A single seed cannot establish a reproducible architecture advantage.

Secondary comparisons: raw versus raw, and each searched/raw agent versus the
original reference. The cloud evaluates Pentobi31.0 level1 on the same reserved openings, with
resumable pair boundaries and every final score cross-checked. It builds the
pinned public Pentobi31.0 source independently; Pentobi supplies no training data.
After downloading frozen checkpoints, compare against the preserved warm round8
agent locally without exporting that additional earlier checkpoint.
Pentobi remains evaluation-only. No held-out result changes checkpoint selection.
Normal-start paired games are reported separately as descriptive evidence.

The cloud run has a2.5hour training limit per architecture; a hard7hour function
limit covers both runs and evaluation. If deep training stops early, the shallow
control is limited to its actual completed round count. Report any failure to
match actual budgets rather than describing unequal runs as a matched comparison.
The study has one serial GPU job, with durable atomic optimizer/replay/RNG round
checkpoints in a private Modal Volume. No automatic retries or duplicate jobs.

## Hardware and credit limits

First run bounded600second timing/finite-loss screens, with frozen weights,
comparing one versus six shared search workers. Updated benchmark weights are
discarded. Measure real games/neural positions per second and price, including
CPU resources; sample GPU busy time, memory IO and allocation separately. L4 is
first; try L40S/H100 only to test a concrete throughput/cost hypothesis. Select
hardware using throughput and cost, never reserved strength tests.

Profile linuxfan2718; **this new overnight phase is capped at $35 in conservative
resource reservations**, within Dennis's overall $130 existing-credit budget.
No top-ups, subscriptions, pushes, publishing, external messages or subagents.
The $35 is a ceiling, not a spending target. Ledger: runs/modal-deep-cost-ledger.jsonl.
Only allowlisted project source, the new runner and the same explicitly approved
frontier342 reference checkpoint are uploaded. No AlphaBlokus code or weights are
uploaded or used as training initialization. The private Volume stores our new
checkpoints/data; retrieve them locally before any cleanup. Billing snapshots
must distinguish estimated resources, metered usage and applied credits.

## Reproducible entry points

Config: runs/deep-study-config.json; runner: cloud/modal_deep.py; trainer:
python/blokus_rl/duo_deep.py. First benchmark command:

```sh
.venv-modal/bin/modal run --profile linuxfan2718 cloud/modal_deep.py --stage benchmark --gpu l4 --run-name deep-benchmark-l4-764
```

The study command uses --stage study --gpu SELECTED --run-name duo-deep-study-764
and --detach at the Modal CLI level. Resume only an incomplete existing directory
with the exact original configuration after checking app state and round logs.
Rolling state, failures and next actions belong in docs/deep-status.md.

## Hardware screen results and final run size

The L4 screen completed in77.4seconds. Six workers generated2.30 games/s versus
0.99 with one worker. The L40S screen generated5.28 versus2.26 games/s. Six-worker
neural throughput was3831 positions/s on L4 versus8721 on L40S; measured mean
GPU batch112/128. Requested CPU/GPU/RAM rate is $1.3704/hour on L4 versus
$2.5204/hour on L40S. For this small frozen screen that is roughly $0.166 versus
$0.133 per1,000 games, excluding startup/build/storage. L40S is the best measured
complete system; host CPUs differ, so the gain is not isolated GPU acceleration.
No H100 rerun is needed without a new bottleneck hypothesis.

L40S also passed an actual CUDA/bfloat16 two-round interrupted/resumed smoke
training run: exactly24 games, finite losses and valid searched evaluation.
Those weights were discarded. Smoke validation used2561000, separate from the
study's validation and held-out seeds. The deep model has986,205 parameters.

Based solely on throughput and successful recovery, before main training starts,
the planned upper limit was increased from64 to128 rounds per architecture.
The2.5hour time limit is retained; shallow control uses the deep run's actual
completed round count. Freeze runs/deep-study-config.json before launch.

## Interpretation limits clarified after completion

The deep/shallow comparison changes an architecture bundle, including depth,
width, normalization, coordinate inputs, inventory modulation at different layers
and tanh versus linear value outputs. Both models receive remaining-piece
inventories. Equal game counts and fixed training rules need not produce exactly
equal numbers of recorded positions, simulations or gradient examples, because
their game trajectories differ. Report those actual counts and elapsed time.
Neither a gain nor a failure isolates depth as its cause.
