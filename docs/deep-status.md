# Deep self-play overnight status

User-authorized overnight phase,25 October2026. Methods and budget:
docs/deep-self-play.md. Chat explanation: docs/alphablokus-explanation.md.
Original study: docs/alphablokus-study.md. Earlier models remain unchanged.

## Current state: study complete

The bounded cloud study finished successfully in14,721.58seconds (4.089hours)
at06:09UTC; its app is stopped with zero tasks. Full artifacts are downloaded and
verified at runs/duo-deep-study-764-download-20261026T0634/duo-deep-study-764/.
Both frozen selected hashes match the protocol and best.pt; earlier certified
tic-tac-toe/frontier checkpoints and preserved warm-round8 hash are unchanged.

The primary deep-model hypothesis failed: deep search128 vs shallow search128
scored25.5% (paired95%CI19.75–31.5). Shallow search128 scored78.25% against the
original searched reference and31.25% against Pentobi31.0 level1 (25.25–37.5).
These are frozen held-out results, unlike the earlier validation scores. The
shallow model is the stronger of these two cold-start agents for this seed;
Pentobi still wins most games. No new training follows this result.

All eight local comparisons are complete: runs/duo-deep-local-764/completed.json,
log runs/duo-deep-local-764.log. Shallow CNN raw scored63.0% against preserved warm
round8 raw (paired95%CI57.5–68.5); with128 simulations for both sides it scored
72.75% (66.5–78.75). Deep search128 scored53.25% (46.75–60.0), without a confirmed
gain over warm round8. All checkpoints remain unchanged, including earlier models.

Read [the final report and compact plot](deep-results.md). There are18 frozen
comparisons×200 games on the same100 reserved opening pairs, plus8 descriptive
normal-start games. All reported intervals are nominal paired95%, not uncertainty
across training seeds. Training used no outside teacher; Pentobi remains stronger.
No training, evaluation or cloud processes remain. Next hypothesis: independently
replicate the shallow setup across seeds before architecture/teacher changes.
It is documented as future work; no new experiment is launched by this status.

## Validated implementation

- All68 Python tests pass; lint and formatting pass. New tests cover real-process
  shared inference parity, full-game move indices, learning both heads, batched
  evaluation, atomic resume/selected checkpoint recovery, time-limit selection,
  and Pentobi pair resume with model/config guards. No Rust rules were changed.
- L4 benchmark finished:0.99 games/s with one worker,2.30 with six. L40S finished:
  2.26/5.28 games/s, about20% lower requested-resource cost/game than L4 in this
  screen. These include different host CPUs, not isolated GPU comparisons.
- L40S passed actual CUDA/bfloat16 resumed training:24 games in two rounds,
  finite policy/value losses. Smoke weights discarded; no held-out seeds used.
- Benchmark apps ap-RoaDV7u20Yt9BGIxn9Kqk8 and ap-9R1Y2Qelk553FZkSeB6RlK are
  stopped, zero tasks. No duplicate training process existed before launch.
- Selected hardware: one L40S,8 capped CPU cores,24GiB capped RAM,6 search workers
  and up to256 positions/inference batch. Model:10 residual blocks×64 channels,
  986,205 parameters. GroupNorm, inventory FiLM and normalized board coordinates.
- Frozen planned config runs/deep-study-config.json:128 rounds×256 games/architecture,
  max2.5hours training each. Deep then shallow serially, matched actual deep round
  count. Full256-search targets on20% of moves, cheap32-search on80%, no expert
  demonstrations. One seed764: report this limitation.

## Cloud run and limits

Runner cloud/modal_deep.py; app blokus-deep-selfplay-study; private Volume
blokus-deep-selfplay-20261026. Main artifact folder duo-deep-study-764. The hard
function timeout is7hours, retries0, one container. Both architectures and the
reserved held-out/Pentobi evaluations run serially in that one call.

New overnight phase cap: $35 conservative resource reservations, within Dennis's
$130 existing credits; no top-ups or plan changes. Profile linuxfan2718.
Ledger runs/modal-deep-cost-ledger.jsonl. Only allowlisted updated source and the
same explicitly approved original frontier342 checkpoint leave this Mac. Pentobi
is fetched from its public v31.0 source, pinned d5374ddc20eced902788e733d67ee16bd1836fd3.
No AlphaBlokus code/weights are used for initialization or uploaded.

Study app: ap-H3fgjNS4auEbHTqNTsQXon. Source commit6963505b5a96318dcd136368388d7cac1d886e72.
Started26 October02:04UTC, in detached mode. Linux Pentobi compiled successfully,
and the independent full-game legality/score cross-check passed before training.
At the first download check, round7/1,792 games had committed with finite losses.
Downloaded runs/deep-live-checkpoint-20261026T0209.pt and matching status snapshot;
the checkpoint loads on this Mac with986,205 parameters and produces a legal move.
This smoke check is not a strength evaluation or checkpoint selection.

During the study, follow-up automation finish-overnight-deep-blokus-self-play ran
every30minutes, quiet while healthy; it is removed at completion. A temporary
9hour AC-only system-sleep assertion supported local follow-ups and is released
after completion. No persistent power settings were changed.

Launched command:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_deep.py --stage study --gpu l40s --run-name duo-deep-study-764 > runs/duo-deep-study-764.log 2>&1
```

## Historical follow-up instructions (completed)

Check app/process state, ledger and this log before any action. Do not launch a
second study while one is active. Healthy progress needs no notification.

Private Volume paths under /duo-deep-study-764/:
- residual/status.json, progress.jsonl, resume.pt, best.pt, latest.pt, completed.json.
- conv/ equivalents (starts after deep completes, capped to its actual round count).
- frozen-residual.pt/frozen-conv.pt, heldout-protocol.json, heldout-*.json.
- telemetry.jsonl, rules-crosscheck.json, finished.json.

Fetch small status files with modal volume get ... REMOTE_PATH LOCAL_PATH,
--profile linuxfan2718; avoid overwriting an earlier snapshot silently. A failure
must be diagnosed; after confirming no active duplicate, resume with exactly the
same local config using --resume. Atomic checkpoint includes optimizer/replay/RNG
and selected-model state. Reservations count retries too; never bypass the $35 cap.

After finished.json appears, download the entire new artifact folder to an unused
local runs directory, verify frozen checkpoint hashes against heldout-protocol,
freeze the compared checkpoints, and evaluate against preserved warm round8 locally
using batched evaluation. Keep raw/search results and validation/test separate.
The cloud evaluates raw/search128 against original reference and Pentobi31.0
level1 on100 paired held-out openings, plus the primary deep-vs-shallow128 case.
Pentobi partial cases resume at whole-pair boundaries and validate model/config.
Normal-start games are saved separately without a population CI.

Build a compact results plot/report with actual budgets, methods, failures,
uncertainty and costs. Preserve all negative evidence. One training seed and many
secondary comparisons limit conclusions. Back up new artifacts locally, check
billing and stopped apps, commit validated code/docs locally, report results here,
and delete the follow-up schedule. No pushing, publishing, outside messages,
subagents or indefinite larger training series. Further hypotheses need a bounded,
predeclared design and remaining authorized time/credits; do not tune on these tests.

## Planned local evidence after cloud completion (now complete)

Also evaluate each frozen selected policy raw against its own untrained initial
network and the largest-piece heuristic on the same reserved opening pairs.
These are additional descriptive learning/baseline checks, not checkpoint selection
or replacements for the prespecified primary case. Label initial networks as
untrained networks, not uniform-random opponents. Keep comparison with preserved
warm round8 raw and searched128 separate; use existing evaluate_many in
blokus_rl.duo_deep_evaluate, CPU threads chosen by a small timing-only check if useful.
Do not export the warm checkpoint; do this local comparison after download.

The architecture comparison changes depth, width, normalization and inventory
conditioning together; it does not isolate residual depth as the cause of a gain.
This phase uses AlphaZero-style self-generated labels rather than the earlier PPO
objective. At launch, no new results had yet established a stronger agent.

## Follow-up check:26 October02:36UTC

The single detached study remains healthy, with one active container and one
matching local Modal client; both hardware-screen apps are stopped. Residual
training has completed32 rounds/8,192 games with finite policy/value losses.
Round32 validation score0.4896 is checkpoint-selection evidence on24 opening
pairs, not a held-out result. The shallow control and reserved tests remain ahead.
No job was restarted, configuration changed, or additional reservation made.

The ledger still reserves $18.45155 total for this phase, including the entire
main job's hard timeout. Provider billing currently reports $1.54136 metered
workspace cost and $0 billed after credits/allowances; accounting can lag and
this workspace figure includes earlier work. The $35 phase reservation ceiling
remains in force. Next action: let the existing call finish its frozen protocol,
then download and verify artifacts before the local warm-round8 comparison.

## Follow-up check:26 October03:04UTC

The same single app/container and local client are active, without duplicates or
errors. Residual training reached61 rounds/15,616 games; policy/value losses and
gradient norms remain finite. The shallow control and reserved evaluations have
not started. No configuration, checkpoint selection rule or launch was changed.
Provider workspace billing now reports $2.74136 metered, covered by existing
credits/allowances ($0 billed); this is a lagging aggregate, not final study cost.
The conservative phase reservation remains $18.45155 of the $35 ceiling. Continue
the existing bounded call; collect final artifacts and local comparisons afterward.

## Follow-up check:26 October03:34UTC

The same single detached study remains active and healthy:90 residual rounds,
23,040 self-play games, finite losses/gradients, no duplicate or retry. Round88
validation score0.5833 remains selection evidence on the small validation set;
there are no held-out results yet. Shallow training and tests remain pending.
Provider workspace metered total is $4.04136, with $0 billed after credits and
allowances; billing components update asynchronously. Reservations remain
$18.45155, with no additional launch or configuration change. Continue the
existing protocol and defer strength conclusions until frozen held-out evaluation.

## Follow-up check:26 October04:04UTC

One healthy detached app/container and matching local client; residual training
has reached121 rounds/30,976 games without errors or retries. Round120 validation
score0.625 remains checkpoint-selection evidence only. The control and held-out
tests are still pending. Provider workspace metered total is $5.36136, covered by
credits/allowances; final per-app accounting remains pending. Phase reservations
are unchanged at $18.45155. No new job, upload or configuration change was made.

## Follow-up check:26 October04:35UTC

Deep training finished its full128-round/32,768-game budget in7,555.25seconds
(round limit, not time limit). The private Volume completion record was downloaded
to runs/deep-residual-completed-20261026T0434.json. Best validation score0.625 is
still selection evidence, not a test result. The existing call proceeded directly
to the shallow control, now32 rounds/8,192 games, with finite losses. One app and
one container remain active; no retry, duplicate, new upload or config change.
Held-out evaluation remains pending. Workspace metered cost is $6.51136 and
covered by credits/allowances; conservative reservations remain $18.45155/$35.

## Follow-up check:26 October05:04UTC

The shallow control has reached70 rounds/17,920 games with finite losses and
gradients. Deep remains complete at32,768 games. One detached app/container and
matching local client are healthy, with no duplicate or retry. Held-out tests
remain pending; round64 shallow validation0.6354 is selection evidence only.
Provider workspace metered total is $7.84136, covered by credits/allowances.
Reservations remain $18.45155/$35; no configuration or launch was changed.

## Follow-up check:26 October05:34UTC

The same single app/container is healthy, with the shallow control at104 rounds/
26,624 games and the deep run complete. Losses/gradients remain finite; no retry,
duplicate, new upload or change to the frozen protocol. Held-out tests have not
started. Provider workspace metered total is $9.20136, covered by existing credits
and allowances; reservations remain $18.45155/$35. Let the existing call finish
training, checkpoint freezing and reserved evaluation before final download.

## Follow-up check:26 October06:07UTC

Both runs have finished128 rounds/32,768 games. Frozen-checkpoint cloud tests
are in progress: all six neural-vs-neural cases and deep raw/search Pentobi cases
have completed; shallow Pentobi testing is active. One app/container remains
healthy. No selection or training was changed after looking at test progress.
Do not report a final strength conclusion until all paired results are collected.
Provider workspace metered total at the initial check was $10.41136, covered by
credits/allowances; phase reservations remain $18.45155/$35.

Prepared a local-only finisher, blokus_rl.duo_deep_local, for the already planned
warm-round8/untrained/largest comparisons. It requires finished.json, verifies
frozen cloud hashes, copies local checkpoints, locks against concurrent duplicate
evaluation, saves whole opening pairs atomically and rejects changed settings on
resume. Its real-game interrupted/resumed test passes and matches uninterrupted
outcomes, seed offsets and intervals; lint/format pass. No extra model was exported.
A timing-only round7 snapshot benchmark on32 repeated initial positions measured
297/299/300 positions/s with1/2/4 CPU threads, too similar to justify extra cores;
use one thread. No validation or held-out games were used for that timing choice.

After downloading the completed cloud folder to an unused local directory, run:

```sh
.venv/bin/python -m blokus_rl.duo_deep_local --run runs/DOWNLOADED_STUDY --out runs/duo-deep-local-764 --threads 1
```

Keep it local, check no existing finisher before launch, capture its log and use
--resume only after diagnosing an interruption. It evaluates eight fixed cases,
100 opening pairs each, in10-pair chunks; CPU float32 differs from cloud bf16.
It does not train, select checkpoints or change the primary held-out comparison.

## Follow-up check:26 October06:42UTC

Cloud download contains78 files/869,381,726bytes, with local SHA256 manifest
download-sha256.json. Backup .backups/deep-study-cloud-20261026.tar.gz is
241,064,231bytes, SHA256dc01800cc1ebd52a5aa7ab18aa50c0e59cdd5ff9c9b082fe4551dbb9266c4fb7.
The first recursive download failed on a nonexistent local destination; retrying
into its empty created directory worked. No cloud compute was retried. A client
log-tail timeout occurred after the successful remote return; finished.json,
complete artifacts and stopped app confirm success.

Provider per-app snapshot runs/modal-deep-final-billing-full-days-20261026T0635.json
reports main compute/resources $10.32494, screens $0.03641/$0.03760 (L40S/L4),
about $10.39895 phase app total. Storage/download metering can arrive later.
Workspace snapshot $10.52136 is covered by credits/allowances, with $0 billed;
it includes earlier work. Reservation ledger remains $18.45155/$35, with no new
cloud job. Mean whole-call GPU busy12.2%, median7%, peak allocated1.59GiB.

Actual deep/shallow training simulations:85,781,504/86,621,152; recorded full-search
positions223,537/225,539; gradient examples688,128/691,712. Equal games and settings
did not produce equal wall time or exact example counts. Architecture changes also
include coordinates and bounded-vs-linear value heads; depth is not isolated.
All68 Python tests passed (runs/deep-local-regression-20261026.log), and lint/format
pass. Remaining work: let the local finisher complete, generate/inspect plot and
report, back up local results, refresh final billing/app state, commit validated
changes locally, report final results and delete this heartbeat. No further training.

## Completion audit:26 October2026

All18 comparisons finished (3,600 games on100 shared opening pairs), and all
nominal paired intervals were independently recomputed from the saved pair scores.
Full downloaded cloud manifest (78 files), frozen checkpoints, certified models
and preserved warm-round8 checksum verified again. Audit record:
runs/deep-study-final-audit.json. No cloud training/evaluation job or local finisher
remains active. The temporary study sleep assertion was explicitly released.

Final report/plot: docs/deep-results.md and docs/assets/deep-results.{png,svg}.
The plot was visually inspected; tables distinguish raw/search, validation/test,
actual training counts, nominal paired intervals and one-seed limitations. The
best candidate from this study is the frozen shallow checkpoint at
runs/duo-deep-local-764/conv.pt; README includes a human-play command.

Provider final snapshots are runs/modal-deep-final-billing-*-20261026T0704.json.
Phase per-app metering remains $10.39894, versus $18.45155 reserved and a $35
ceiling. Workspace aggregate $10.52136 is covered by existing credits/allowances.
The completed main call returned after14,721.58seconds; its finished marker was
written at14,718.37seconds, before the final Volume commit. No retry was needed.

Cloud archive: .backups/deep-study-cloud-20261026.tar.gz. Final local results,
source, reports, execution payloads, ledger, billing and audit are captured in
.backups/deep-study-final-20261026.tar.gz, with a neighboring hash manifest.
These local backups share this Mac; the private Modal Volume also retains the
original cloud artifacts. Existing earlier backups/checkpoints are preserved.
Validated changes are committed locally at completion, with no push/publication.
The completion heartbeat is removed after the study; no further work is scheduled.
