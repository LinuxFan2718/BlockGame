# Controlled training with 1,024 versus 2,048 search simulations

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
