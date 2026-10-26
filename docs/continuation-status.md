# Controlled Blokus Duo continuation — 26 October 2026

Dennis renewed permission to work while away for about 12 hours and spend
existing Modal credits on stronger agents while keeping a reserve. This
supersedes the prior study's stop on new training. It does not alter the running
levels 1–9 benchmark, its frozen model or its test protocol.

## Decision and hypothesis

Continue the strongest small CNN. Its previous run's last checkpoint had the
best validation score, providing a reason to test more experience rather than
assume training has plateaued. This is not proof that continued improvement will
be linear. Test whether stronger self-play search gives more useful learning
than the existing search setup at equal additional game counts.

Two branches inherit identical parent network weights, AdamW moments and all
100,000 replay positions. Both use a fresh declared RNG seed865. This is a
controlled continuation of one parent lineage, not two independent replications
of cold-start learning.

| Setting | standard256 | strong1024 |
| --- | ---: | ---: |
| Additional games planned | 32,768 | 32,768 |
| Rounds × games | 128 × 256 | 128 × 256 |
| Search on 80% of moves | 32 | 32 |
| Search on recorded 20% of moves | 256 | 1,024 |
| Soft training limit | 2 hours | 5 hours |

Other settings retain DeepConfig defaults: shallow 52,978-parameter CNN, six
CPU workers, 32 live games per worker, batch256, replay100,000, three training
examples/new recorded root, AdamW0.001, policy CE0.5 plus terminal WDL MSE,
bfloat16 CUDA inference. No expert demonstrations or heuristic rewards.
Only full-search roots are recorded. Stronger search also changes trajectories
and terminal outcomes; this tests the search-budget treatment, not target
quality in isolation. It spends more compute per game.

Parent frozen SHA256:
`0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86`.
Parent round128 optimizer/replay SHA256:
`2df38e9b7585c23fe0b8159cc4511db1811c0b745534d8a9223e885dec1db52d`.
The state contains 32,768 ancestral games and matches the frozen model's tensors.
Those games are not counted again as new training.

## Selection, diagnostics and success criteria

Every16 rounds, evaluate against the frozen parent at128 simulations each on
32 validation opening pairs, seed2583000. Include the unchanged parent as a
selection candidate at50%, so a branch need not replace it with a weaker model.
Selection never uses reserved test results.

Before training, compare the unchanged parent at512 and2048 simulations against
itself at128 on24 diagnostic opening pairs, seed2581000. This measures playing
search benefit independently of learning. It does not choose the training arm,
test playing budget or selected checkpoint.

Freeze both selected checkpoints before generating100 reserved four-ply opening
pairs, seed2589000. Primary: strong1024 versus standard256, each128 simulations.
Report paired W/D/L, score and nominal whole-pair95% interval. A lower bound
above50% supports the stronger-search treatment for this parent/seed. If training
caps produce unequal actual game counts, report that limitation rather than
claiming a matched-game result.

Secondary: raw head-to-head; each branch versus the parent, raw/raw and128/128;
each branch and parent versus Pentobi31.0 level1 on the same fresh book, raw and
128. Report paired score differences against Pentobi as well as absolute scores.
Gaining against the old parent alone does not establish outside strength. One
lineage and multiple secondary tests limit generalization. Preserve all models,
including weaker branches, and report actual simulation/gradient/game budgets.

Immediate milestone: improve Duo strength against the previous best and
Pentobi. Matching Pentobi level9 in Duo would be substantial evidence of progress,
but cannot establish the four-player Classic frontier. Our agents have not
played AlphaBlokus; its inspected published agent uses a different variant.

## Resource limits and preservation

Profile linuxfan2718. **$25 phase reservation ceiling**, within the original
$130 existing credits; retain at least **$50 uncommitted** at launch. Current
workspace metering is about$11, covered by credits; provider accounting can lag.
The guard uses at least$15 for earlier work plus the existing benchmark's
$3.43666 outstanding reservation before adding this new reservation.

One detached L40S function,8 capped CPU cores,24GiB RAM,9-hour hard timeout,
retries0,max1 container. Verified provider rates are$1.95/GPU-hour,
$0.0473/core-hour and$0.008/GiB-hour: total$2.5204/hour.
Conservative reservation including90 seconds startup is **$22.74661**.
The separately authorized CPU benchmark continues; these are two intentional
different jobs, not duplicates. No H100 or hardware sweep is needed.

All parent models/replay already exist in the private Volume
blokus-deep-selfplay-20261026. Only allowlisted project source/new runner is
exported; no additional model/replay, third-party weights or unrelated files are
uploaded. Earlier certified tic-tac-toe/Blokus artifacts are untouched. The new
run writes an unused directory duo-continuation-865. Local code first verifies
forking and exact interrupted/recovered training. A tiny actual CUDA recovery
screen runs before main training; its weights are never promoted.

Source: python/blokus_rl/duo_continuation.py and cloud/modal_continuation.py.
Read app/process state, logs and ledgers before launch. No automatic retries,
top-ups, plan changes, publishing, pushes, messages to people or subagents.
On failure preserve checkpoints and diagnose; do not launch a repeated9-hour
reservation. The phase cap does not permit one.

## Execution and follow-up

Executed once after validation; do not repeat:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_continuation.py > runs/duo-continuation-865.log 2>&1
```

Ledger: runs/modal-continuation-cost-ledger.jsonl. The exact executed source
archive, runner, billing-before snapshot and source commit are retained locally.
Monitor atomically committed round states, finish/failure markers, telemetry,
provider costs and app state. Healthy progress needs no notification.

On completion retrieve artifacts into a pre-created unused directory, verify
frozen hashes and budgets, produce docs/continuation-results.md with a compact
plot and paired intervals, back up new artifacts and commit validated code/docs
locally. Preserve the existing benchmark's separate result/report obligations.
Stop the shared follow-up after both bounded jobs and reports finish. Do not
automatically begin a larger training series or release the$50 reserve.

### Launch audit

Running detached: **ap-DQEFQZzvb7k0kP8eAjXRWz**, app
blokus-controlled-continuation. Ledger entry26 October09:32EDT reserves
$22.74661. The only other active app is the intentional CPU benchmark. No new
model or replay was uploaded. The private parent model/state hashes passed
before artifact creation. Source commit:
`1ebb18e97014dafee86d11ed6bc5ebbc2b9e5018`.
Exact uploaded payload SHA256:
`84182ab9d14994a80bbcd90b6765bd9f55a40aad7faecc61d5797821a919061d`.
Runner SHA256:
`9636c0be374901f3587015d7e0278db931e1b61a84230bb6a0c79616a5c9cc74`.

All71 Python tests passed (7 existing compatibility/recommendation warnings),
including inherited optimizer/replay checks, unchanged parent files and exact
interrupted/recovered fork training. Lint/format and diff checks passed.
Log: runs/continuation-regression-20261026.log. Prelaunch source/docs backup:
.backups/continuation-start-20261026T133030Z.tar.gz, SHA256
`66313b13a4e2529ffd5f05b30ba9a765d222a4597e01fb17086e865f52686d0c`.

The existing15-minute heartbeat finish-pentobi-levels-1-9-benchmark was updated
to monitor both bounded studies and their separate reporting obligations. It
must not delete itself merely because the CPU benchmark finishes first.

Actual CUDA/bfloat16 smoke passed:16 games,2 rounds,finite losses and a successful
atomic-checkpoint recovery between rounds. These are functionality checks;
its tiny validation result is not strength evidence and its weights are not
promoted. Downloaded completion and remote protocol were checked locally;
the main arms differ only in full-search budget and soft runtime cap.
Launch payload/ledger/docs backup:
.backups/continuation-launch-20261026T133251Z.tar.gz, SHA256
`989c3083815bc135baf880460f25b92224d9fccfb009f82628cc9e616126cfb0`.

### 09:50EDT health check and completed search diagnostics

Both intentional apps remain healthy with one container/client each. Standard256
has completed at least15 rounds (3,840 additional games), with finite losses;
strong1024 has not started because the arms run serially. The Pentobi benchmark
has passed50 pairs at levels1–8 and40 at9. No reserved continuation test has
been run. Workspace metering is$11.97137, covered by credits; accounting can lag.

The two prespecified frozen-parent diagnostics completed on24 opening pairs
(48 games each), seed2581000, GPU bfloat16. Each side uses the same network;
only playing search differs:

| Parent search vs parent search128 | W/D/L | Score | Nominal paired95% interval | Evaluation seconds |
| --- | ---: | ---: | ---: | ---: |
| 512 simulations | 35/0/13 | 72.92% | 62.50%–83.33% | 73.74 |
| 2,048 simulations | 44/2/2 | 93.75% | 87.50%–98.96% | 230.98 |

This supports a playing-search benefit against the same policy at128 simulations.
It does not measure improvement against Pentobi or establish that training with
more search will improve the network. No protocol, selection rule or test budget
was changed in response. Local records:
runs/continuation-diagnostic-search{512,2048}-20261026.json.

### 11:35EDT stage transition

Standard256 finished all128 rounds/32,768 additional games in6,326.93 seconds
(1.76 hours), ending at its round limit. Its best validation score against the
parent is77.34%; this selected-on-validation result is not held-out evidence.
Strong1024 has started from the same preserved parent state and has passed five
rounds with finite losses. No continuation held-out comparison has started yet.
Local first-arm completion record: runs/continuation-standard-completed-20261026.json.

The explicitly approved CPU recovery is progressing:100/100 pairs at levels1–8,
95/100 at9 in its latest committed status. It emits an unsupported-NNPACK
hardware warning while PyTorch continues on a fallback path; no traceback or
legality/score failure is present. Preserve the original float32 protocol and
report the cross-host/backend limitation in the eventual benchmark report.
Both active jobs remain within their reserved limits; workspace metering is
$17.67137 so far and$0 billed after credits/allowances, subject to accounting lag.

### 11:50EDT benchmark completion

The approved CPU recovery finished all 900 pairs / 1,800 games and stopped.
Both benchmark apps have zero tasks. Results and verified plot are in
[pentobi-level-results.md](pentobi-level-results.md): the frozen parent scored
27.50% against level 1 and remains below it under the paired opening protocol.
All original 820 pair files are unchanged and all 1,800 games replayed correctly.
Benchmark metering totals $1.82439453, distinct from $4.76284 reservations.

Only the GPU continuation remains active. Strong1024 is progressing with finite
losses; its held-out comparisons have not started. The completed level sweep
does not change this study's configuration, checkpoint selection or test budget.
The shared follow-up now monitors only the remaining GPU study and must not
repeat the completed benchmark's notification or launch more level games.

### 15:50EDT training complete; selected checkpoints frozen

Both branches finished all 128 rounds / 32,768 additional games at their round
limits. Standard256 took 6,326.93 seconds; strong1024 took 15,953.60 seconds
(4.43 hours), within its 5-hour soft limit. Neither arm was truncated, although
their actual simulation and gradient budgets still require the final audit.

The job froze the selected policies and wrote heldout-protocol.json before
starting the reserved evaluations. Frozen SHA256 values are:

- standard256: `4cbeb255cb5688a6b931ac4ac6964f21e3c6430be2e6dd558e6d7c9a14d43b19`
- strong1024: `95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`

The unchanged parent hash, seed 2589000, 100 opening pairs, 128-simulation playing
budget and bfloat16 precision match the reserved protocol. Local completion and
freeze records are in runs/continuation-freeze-audit-20261026T1950/. These are
stage records, not final test evidence. The existing app remains active with
one task; do not start a separate evaluation or training process.

### 16:07EDT completed and downloaded; final audit

The function committed finished.json at 16:06EDT, and app
ap-DQEFQZzvb7k0kP8eAjXRWz stopped at 16:06:32 with zero tasks. All 12 reserved
held-out cases completed (2,400 games); no preemption, retry, training cap
mismatch or scientific-protocol change occurred. The final CLI log-reader
timeout followed successful remote completion and is not a study failure.
The discarded CUDA smoke contributed 16 separate functionality-check games.

The complete download is
runs/duo-continuation-final-20261026T2007/duo-continuation-865:
74 files, 1,194,949,217 bytes. The report generator independently verifies all
paired summaries and all 1,200 saved external action traces, frozen/initial model
hashes, final atomic states, inherited+new AdamW steps, original parent hashes,
selection against separate validation and preserved certified checkpoints.
Derived verification and every downloaded file hash are saved alongside the run.

The primary strong1024 versus standard256 comparison, both playing search128,
scored 60.25% (116/9/75 W/D/L; nominal paired 95% interval 54.50%–66.00%), meeting
the declared lower-bound-above 50% criterion. Raw strong versus raw standard
scored 68.25% (62.00%–74.25%). These are held-out results, not validation scores.
Against Pentobi 31.0 level 1 on the same fresh book, parent/standard/strong
search128 scored 30.25%/50.75%/62.50%; raw scored 10.50%/20.25%/36.25%.
Strong search128 had 121/8/71 W/D/L, interval 55.75%–69.00%, and a 32.25 percentage
point improvement over the parent (paired interval +23.00 to +41.25 points).
The complete tables, paired differences, methods and plot are in
[continuation-results.md](continuation-results.md).

Each arm generated 32,768 additional games. Actual standard/strong training used
87,904,544/263,393,152 simulations, 228,636/228,639 recorded full-search roots,
2,739/2,748 gradient updates and 701,184/703,488 gradient examples. The selected
strong model is round 112 (28,672 additional games), while standard selected
round 128; strong's last 16 rounds were generated but not selected. Equal generated
games do not imply equal compute or selected-checkpoint experience. Neither the
single lineage/seed nor beating level 1 establishes reproducibility or a Blokus
frontier result. The new agent has not played higher levels or AlphaBlokus.

Saved provider metering is $16.57150713 for the continuation app, versus its
$22.74661 reservation and $25 ceiling. Workspace metering at that snapshot is
$29.14137374, $0 billed after credits/allowances; accounting may lag. Approximately
$100.86 of the original $130 allowance remains, and the $50 reserve stays
uncommitted. All paid compute is stopped. No further training or benchmark is
authorized by this completed protocol; proposed next work remains a proposal.

Private Modal parent weights and optimizer/replay were also downloaded directly
and hashed after completion; both match their original hashes. An earlier
read-only check piped Modal CLI stdout to shasum, which included the CLI
completion message and therefore produced invalid stream hashes. Direct file
hashes resolved this audit-tool issue; no parent changed. The note and correct
hashes are retained in runs/continuation-remote-preservation-20261026.json and
runs/continuation-stdout-hash-check-note-20261026.md. Exact executed payload and
runner hashes were checked against the launch ledger as well.

Final local backup: .backups/continuation-final-20261026T2015.tar.gz,
248,673,712 bytes, SHA256
`eb763d26b0a9292dd5ec158e0da1457f5e5bca3af8f1f4666798b7ec7c05f8c6`.
It contains102 files /1,195,482,612 uncompressed bytes: the complete new run,
all checkpoints and atomic states, derived audits, exact execution payload,
cost records, report/plot and relevant code/docs. Every archived member was
read back and its SHA256 compared with the original; the sidecar JSON records
the full manifest. This is a second local copy, not an off-site backup.

The final report/audit and Ruff formatting/lint plus git diff checks pass.
No engine, training or checkpoint change followed the freeze. All original
certified tic-tac-toe, earlier Blokus and new selected policies remain preserved.
The final app snapshot confirms stopped/zero tasks. The completed study now
requires only local commit and removal of the shared follow-up schedule; no
new paid work is queued.
