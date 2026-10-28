# Controlled training with 1,024 versus 2,048 search simulations

**Checkpoint recovery authorized; results pending.** Dennis confirmed the
cancellation was unintentional and explicitly requested resuming the saved
checkpoint within the original $32 cap. The recovery continues control round 6
without resetting its optimizer, RNG or replay, then runs the original teacher
branch and tests. No training settings or test/selection books are changed.

At 22:55 EDT on 26 October,
Modal delivered `InputCancellation: Input was cancelled by user` while the
control was collecting its next round. The app stopped at 22:57:09 with zero
tasks. The logs do not identify who requested cancellation or explain why;
do not relabel this as preemption, OOM or a learning failure without evidence.
The control has a consistent atomic checkpoint after five rounds / 1,280 games.
The teacher branch and held-out tests had not started. No new model was promoted.
Dennis subsequently confirmed this cancellation was not intentional.

26 October 2026. Dennis approved the proposed controlled self-play experiment
after our frozen agent lost to Pentobi level 4. This is a new bounded training
study, not permission for indefinite training or a further ladder benchmark.

## What success means

Primary: on a new held-out book against Pentobi 31.0 Duo level 4, the selected
teacher2048 model improves on our preserved current agent, both playing with
2,048 simulations per move. Resample the same opening pairs together; a nominal
95% interval entirely above zero for the score difference supports improvement.
The old 33.75% score used another book; the parent is retested on this new book
so our primary comparison has a simultaneous baseline.

Separate questions:

- Does 2,048-search training beat the 1,024 control against level 4 and head to
  head? If both improve similarly, further training helped but the larger
  training search budget did not establish an advantage.
- Stretch target: the teacher model's absolute level 4 score interval exceeds
  50%. Improving from the parent without beating level 4 is still progress.
- Do gains remain at raw-policy play and against level 3? Report these as
  secondary evidence, not alternate primary outcomes selected after results.

If the primary interval crosses zero, evidence is inconclusive. If it lies
below zero, this treatment failed for this lineage/seed. Retain every model and
the unchanged old best. Lower training loss, selected validation wins, or beating
the parent alone are not enough to claim improved outside strength.

After the test: report results, actual work and cost, preserve and back up all
artifacts, commit code/docs locally and stop the follow-up. A clear improvement
motivates a fresh-seed replication before scaling; an inconclusive or negative
result motivates reviewing learning curves and search/target quality before
another paid run. Beating level 4 would motivate a separately approved level 5
benchmark. None of those further jobs starts automatically.

## Shared starting state and intervention

Both branches start from the same best small CNN, selected round 112,
SHA256 `95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`.
It has 52,978 parameters and 61,440 ancestral games through selection. The
source atomic state is from round 128, SHA256
`df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911`.
Its latest weights and AdamW moments differ from the selected round 112 model.
Therefore both branches reset AdamW and inherit the identical 100,000 replay
positions from that round 128 state. The replay contains later self-play
experience than the selected model; it contains no held-out labels or Pentobi
moves. This is explicit off-policy warm initialization, not exact crash recovery.
No source checkpoint is modified. New training seed 1066 is shared by the two
branches; this is one controlled parent lineage, not independent seed replication.

| Setting | control1024 | teacher2048 |
| --- | ---: | ---: |
| Planned additional games | 16,384 | 16,384 |
| Rounds × games | 64 × 256 | 64 × 256 |
| Search on unrecorded 80% of moves | 32 | 32 |
| Search on recorded 20% of moves | 1,024 | 2,048 |
| Soft training limit | 9,000 seconds | 16,200 seconds |

All other settings match: six CPU workers × 32 live games, leaf batch 8,
shared inference batch 256, replay capacity 100,000, about three gradient
examples per new recorded root, minibatch 256, AdamW lr 0.001/weight decay
0.0001, policy CE weight 0.5 plus terminal WDL value MSE, gradient clipping 1.
CUDA bfloat16 and TF32 permitted. No architecture change, demonstrations or
heuristic rewards. Stronger search changes trajectories and outcomes as well
as targets; this tests the whole budget treatment. Equal games do not mean
equal compute. Report unequal counts if a soft cap truncates either arm.

Every 16 rounds, select against the frozen parent at 512 simulations each on
32 validation opening pairs, seed 2605300. The unchanged parent is selectable
at score 50%, so a branch can retain it. Freeze both selected files and hashes
before generating the test book. Test seed 2605900, 100 four-ply openings,
both seats: 200 games per comparison. Validation and held-out books are disjoint.

Held-out comparisons, all prespecified:

- Each of parent, control1024 and teacher2048 against Pentobi levels 3 and 4
  at 2,048 playing simulations: six cases, 1,200 external games, full traces.
- Teacher vs control at 2,048 simulations and raw; each new model vs parent raw:
  four cases, 800 head-to-head games, paired outcomes and counters. The existing
  batched head-to-head evaluator does not retain action traces; disclose this.

Report W/D/L, win rate, score (draws half), mean margin, 10,000-resample
whole-opening 95% intervals (seed 17), paired differences, actual simulations,
neural positions, gradients, generated and selected-checkpoint games, timing,
hardware and costs. Nominal secondary intervals are not multiplicity adjusted.
One parent/seed and random four-ply openings do not establish reproducibility,
human Elo, four-player Classic strength or frontier status. No AlphaBlokus games.

## Resource and preservation limits

One detached L40S function, eight capped CPU cores, 24 GiB RAM, 12-hour hard
timeout, retries 0, max 1 container. Rates: $1.95 GPU/hour + 8 × $0.0473 CPU/hour
+ 24 × $0.008 RAM/hour = $2.5204/hour. Conservative reservation including
90 seconds startup: **$30.30781**, below a **$32 phase ceiling**. Retain **$50**
from the original $130 existing credits. Provider metering was about $32.67
at preparation and all older apps were stopped; the guard uses at least $40
for prior work to cover lag. No paid cloud outside this single bounded job.

Profile linuxfan2718; private Volume blokus-deep-selfplay-20261026. Parents
already exist privately; no new model/replay upload. Immutable base image
im-BnyLYweX6vlwu6iCpDnqrC; only cloud/modal_teacher_search.py and the updated
continuation helper are uploaded. Installed training-module hashes must match
their local preserved sources; immutable evaluator and Pentobi binary hashes
are checked. All certified tic-tac-toe and earlier Blokus artifacts are preserved.

Run: duo-teacher-search-1066; ledger runs/modal-teacher-search-cost-ledger.jsonl;
log runs/duo-teacher-search-1066.log. A discarded CUDA fork/train/recovery smoke
precedes the two serial branches. Exact-config atomic round checkpoints remain
available; infrastructure may restart despite retries0, so an existing-directory
guard prevents duplicate training. On interruption diagnose, preserve state and
report; do not automatically retry or exceed the cap. No subagents, top-ups,
plan changes, unrelated uploads, publication, pushes or messages to people.

Launch once after tests and commit:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_teacher_search.py > runs/duo-teacher-search-1066.log 2>&1
```

After finished.json and a stopped app with zero tasks, download into a
pre-created unused local parent. Verify both frozen hashes, parent preservation,
optimizer resets, replay inheritance, selection, budgets and all 1,200 external
traces; recompute every paired summary/difference. Produce a compact report/plot
here, back up all new artifacts with every archive member hash verified, confirm
final metering/stopped apps and commit validated local code/docs. The quiet
follow-up must notify only meaningful completion/failure, not interim validation
scores, and delete itself when the bounded study is finished or blocked.

## Preparation and startup audit

Both continuation tests passed in 19.74 seconds, including preservation of the
original full-state fork and exact interrupted/recovered training from selected
weights with a fresh optimizer. A real local fork also verified the actual round
112 CNN, empty AdamW state and all 100,000 inherited replay positions. Existing
installed training sources match the archived original image payload. Formatting,
lint and diff checks passed; previous apps were stopped before launch.

The first launch, app ap-HNw7nCM4nv87Q9cqQYx0Hm, stopped after 32 seconds
(22:39:43–22:40:15 EDT) before creating any study directory or training games.
The copied helper was loaded without its Python package context, causing an
ImportError for its relative imports. The loader now names the module
blokus_rl.teacher_continuation, and the recovery test exercises that exact
dynamic import path. Scientific settings and source model/replay are unchanged.

The failed app is stopped with zero tasks and the private study directory does
not exist. Preserve the first executed runner/helper/log/billing snapshot with
the -startup-failed suffix. The ledger retains the original launch record and
an explicit settlement of its unused reservation, keeping a conservative $1
allowance for the stopped startup. One corrected launch reserves $30.30781:
combined allowance/reservation **$31.30781**, still below the original $32 cap
and leaving the $50 reserve intact. This is a diagnosed pre-training startup
correction, not an automatic retry of training or a second experiment. No further
automatic launch is authorized by this correction.

Corrected app ap-jflsIhwP1giVlNWVH3ArFn launched at 22:42 EDT on 26 October
(02:42 UTC on 27 October). Source commit
`7021e444faae89a30ba9b17546ae04f8d55cdfc8`; executed runner SHA256
`90e2982ec88de7d771b2560dcdff95712ca8e5fe542787d2e1cb8b917ce5f550`;
helper SHA256 `fa56da7f6668bf2e1e5f3e540a96c424aff0b62a473091b4f4462ad472a26465`.
All private source, installed training-module, evaluator and binary hash checks
passed. The discarded CUDA smoke completed two rounds / 16 games with an
interrupted/recovered training boundary, then control1024 started. Its weights
are not used by either main branch. The dynamic-import recovery tests passed
again in 19.59 seconds. The quiet 15-minute follow-up
finish-controlled-blokus-training-search-study monitors this app only, prepares
the final audit/report and deletes itself on completion or a reported blocker.

## Interruption preservation — 26 October, 22:55 EDT

The current app ran 14 minutes 39 seconds from creation to stopped state.
The failure record covers 777.06 seconds of the function. The saved control
state covers 597.97 seconds through the last committed round; subsequent
in-flight work is not counted as retained training. Smoke: 16 discarded games.

| Retained main training | control1024 | teacher2048 |
| --- | ---: | ---: |
| Completed rounds | 5 | Not started |
| Additional games | 1,280 | 0 |
| Full-search roots | 8,794 | 0 |
| Gradient updates / examples | 105 / 26,880 | 0 |
| Simulations in retained rounds | 10,150,752 | 0 |
| Neural positions in retained rounds | 7,965,254 | 0 |
| Held-out games | 0 | 0 |

The first scheduled validation was round 16, so the selected checkpoint is
still the unchanged parent (additional round 0, embedded ancestral update 112).
No success/failure inference about the 2,048 training hypothesis is possible.
The training weights in latest.pt match resume.pt exactly; the five progress
rows, 1,280 games, 100,000 replay entries and all 105 AdamW steps agree. The
best.pt file was reserialized during resume, so its byte hash differs from the
parent while all model tensors and embedded update match. This is serialization,
not evidence of a changed selected model. Source and certified hashes pass.

Downloaded complete stopped state:
`runs/duo-teacher-interrupted-20261027T0311/duo-teacher-search-1066`
(24 files / 809,929,770 bytes). The parent directory contains
interruption-verified.json and download-sha256.json. Atomic control resume SHA256:
`a3a1875a9379024b9d989e52b0304caa89cefe3cf241d733dbaf7044a9162466`.
Both original source files and all earlier certified/reference checkpoints
remain unchanged. The private copy remains available as well.

Provider metering snapshot: failed startup **$0.00561232** plus corrected app
**$0.55936204**, total **$0.56497436**. This is actual metering, not the net
$31.30781 reservation. The ledger and both attempts are preserved without
silently deleting costs. Billing may lag. All paid compute is stopped and the
$50 credit reserve is intact. No further launch or settlement has occurred.

Verified second local copy: .backups/teacher-search-interrupted-20261026.tar.gz;
the sidecar contains the archive SHA256 and every member hash. All members were
read back and matched their source hashes. Provider logs and snapshots are in
runs/teacher-search-provider-log-20261026.txt,
runs/teacher-search-apps-stopped-20261026.json and
runs/teacher-search-interrupted-billing-20261026.json. The monitoring follow-up
is deleted after reporting this blocker; further progress requires clarification
of the cancellation. A future explicitly authorized recovery should resume the
exact round-5 configuration and RNG/optimizer/replay state, preserve the failure
record, then initialize the teacher only once and complete the reserved tests.

## Explicitly authorized checkpoint recovery — 27 October

Human answer: “No—resume from the saved checkpoint within the original $32 cap.”
The recovery refuses any state other than the verified round-5 SHA256 above,
checks the saved configuration and original protocol exactly, and records an
exclusive recovery-execution.json marker before doing work. It preserves the
original execution/failure files and five progress rows, skips the discarded
smoke, resumes control at round 6 and initializes teacher2048 only once. A new
failure is written to recovery-failure.json. No further automatic attempt.

Hard recovery timeout 42,000 seconds (11 hours 40 minutes), shorter than the
original 12-hour call. Same L40S/8 capped cores/24 GiB, rate $2.5204/hour,
retries0/max1container. Reservation including 90 seconds startup **$29.46767667**.
The original failed/cancelled apps are stopped; their final snapshot total
$0.56497436 fits the existing conservative $1 allowance. An explicit journal
entry releases only the cancelled call's unused reservation; no old entry or
actual cost is removed. Net allowance plus new reservation **$30.46767667** is
below the original $32 ceiling. The $50 reserve is unchanged.

Submission now uses one asynchronous Function.spawn call with a saved call ID
and --detach, so the local client can exit normally without waiting for training.
This reduces dependence on the terminal/session lifecycle; it is not proof of
the cancellation's cause. See [Modal's detached-app documentation](https://modal.com/docs/guide/apps)
and [asynchronous job submission](https://modal.com/docs/guide/job-queue).
Only the updated runner and identical helper are uploaded; private model/replay
stays in place. Command, exactly once after validation and commit:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_teacher_search.py --recover > runs/duo-teacher-search-1066-recovery.log 2>&1
```

Recovery launched once at 19:41 EDT on 27 October (23:41 UTC), app
ap-PxLumtPesHAVP1q9Gz82TA, asynchronous call
fc-01M5ZVVMEGFF6XHPZNH77XSGHZ. Source commit
`57ec799575478f6da365419a523373658605d8ea`; exact executed recovery runner
SHA256 `120489b04cc6b4e0565dbb7214da05d008abde2a33ad4282d7d47388c44caaac`.
The helper is unchanged. The submission client exited normally; the cloud app
remains active with one task and printed that it resumed control from round 5.
The local CLI's “App completed” message describes submission/client completion,
not completion of the training job. Completion still requires finished.json
and stopped compute. Original private execution/protocol/failure files remain
unchanged and recovery-execution.json records this distinct call.

The continuation/recovery tests passed again in 18.93 seconds. Original
DeepConfig and head-to-head call expressions were compared, both scientific
branch configurations matched exactly, and the round-5 checksum passed.
Formatting/lint/compilation and diff checks passed. Quiet follow-up:
finish-resumed-blokus-training-study, every 15 minutes, through the original
held-out report only. Final provider billing must include 27 and 28 October UTC
because this call can cross midnight UTC. No additional paid work is scheduled.

## Local final-audit preparation — 27 October, 20:05 EDT

Prepared `python/blokus_rl/duo_teacher_report.py` while the recovery was healthy.
It launches no games or training. After finished.json and stopped compute, it
checks the original interruption manifest, unchanged historical files and five
progress rows, both executed source archives/commits, parent preservation,
branch initialization and cumulative optimizer/gradient counters. It handles an
unchanged selected parent at additional round 0 (embedded update 112), and a
soft-cap final validation that occurs after the last atomic round checkpoint.

The external auditor independently replays every saved action, checks book,
seeds, seats, scores, margins and actual simulation counts, then recomputes all
summaries and same-opening differences. Head-to-head checks cover paired
outcomes and counters; those games have no retained action traces. Missing app
billing is rejected instead of treated as zero cost. The final audit still needs
the complete artifacts and provider snapshots; these preparations are not
results from the running study.

Nine regression checks passed in 2.26 seconds, including deliberate corruption
of traces, seeds, scores and simulation counts, parent selection, final
validation and lost optimizer steps. The same external auditor also replayed
all 200 previously completed level-4 games and recovered their existing
6,346,752 simulations and 33.75% score. A read-only mirror of the interrupted
download passed the historical/source audit and recovered the $30.46767667 net
reservation. No new remote source was uploaded or scientific setting changed.

Once complete, combine the saved stopped-app snapshot for the two earlier
attempts (`runs/teacher-search-apps-stopped-20261026.json`) with the final recovery
snapshot: Modal's current app list can omit older stopped apps. Fetch billing
for both UTC dates, then run locally:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m blokus_rl.duo_teacher_report --run <complete-download>/duo-teacher-search-1066 --apps <combined-stopped-apps.json> --billing <provider-report-27-and-28-October.json>
```

The tool writes verification.json and a complete downloaded-file hash manifest
only after the checks pass. A final report, plot, verified backup, final costs
and stopping the follow-up remain required after the reserved tests finish.
