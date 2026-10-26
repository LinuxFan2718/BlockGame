# Pentobi levels 1–9 benchmark — 26 October 2026

Dennis explicitly requested testing our best model against every Pentobi level.
This is evaluation only: no training or checkpoint selection. Earlier studies,
models and certified tic-tac-toe checkpoints are preserved.

Supporting explanation: [why-pentobi-is-stronger.md](why-pentobi-is-stronger.md)
records what we tested against AlphaBlokus (no matches), verified Pentobi
implementation differences and hypotheses about our current strength gap.
It does not use partial results from this running benchmark.

## Frozen design

- Model: the selected shallow CNN, 52,978 parameters, round128 of the completed
  cold-start study. SHA256:
  `0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86`.
- Our agent: fixed128 MCTS simulations/move, CPU float32, one PyTorch thread.
- Opponent: pinned Pentobi31.0, commit
  `d5374ddc20eced902788e733d67ee16bd1836fd3`, Duo, levels1–9, one thread,
  no opening book or resignation; default symmetric-draw setting retained.
- Fresh held-out seed2579000:100 random four-ply opening pairs, identical across
  levels, both seats per opening:200 games/level,1,800 games total.
- Independent engine per pair, deterministic seed per game. Every final margin
  is checked against Rust. No early stopping based on observed scores.
- Report W/D/L, score (draws count half), mean margin, actual actor search counters
  and wall time. This is fixed-agent strength testing, not equal CPU budgets.
- Use nominal95% whole-pair bootstrap intervals. For all-zero outcomes, also
  report the conservative one-sided95% upper bound `1 - 0.05**(1/n_pairs)`.
  A degenerate bootstrap interval does not prove a true score of exactly zero.
  This bound follows because a pair's score cannot exceed the indicator that it
  earned any points;100 independent opening pairs give an upper bound of2.95%.
- The result is a performance curve against these opponents, not a calibrated
  human Elo rating. If all scores remain below50%, report below level1 for this
  fixed agent/search setup rather than inventing an interpolated level. Distinguish
  point estimates from confidence intervals crossing50%.

## Runtime screen and budget

Separate pilot seed2571000 is excluded from the benchmark. One game each on the
Mac took0.50s at level1,9.96s at7,41.54s at8 and197.41s at9. This is timing-only
evidence, not a strength estimate. Raw pilot records are in
runs/pentobi-ladder-timing-20261026.{log,json}.

Use one bounded CPU-only Modal function:8 capped CPU cores,24GiB capped RAM,
8 independent workers;6hour hard timeout, retries0, max1 container. No paid GPU.
Profile linuxfan2718; phase ceiling **$5 conservative resource reservations**
within Dennis's already-authorized $130 credits. Estimated CPU/RAM cost is
$0.5704/hour, at most about$3.44 including startup at the hard timeout.
No top-ups, plan changes, publishing, pushes, messages or subagents.

The best model already exists in the authorized private Volume
blokus-deep-selfplay-20261026 at duo-deep-study-764/frozen-conv.pt. The runner reads
it there and checks its hash. **No new model upload** is needed. Only the existing
allowlisted project source and benchmark runner are exported.

## Execution and follow-up

Run name: duo-pentobi-ladder-2579000. Source: cloud/modal_pentobi_ladder.py and
python/blokus_rl/duo_pentobi_ladder.py. Local logs and cost ledger live in runs/.
Before launch/resume inspect apps, processes, provider billing and
runs/modal-pentobi-ladder-cost-ledger.jsonl; never start a duplicate.

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_pentobi_ladder.py > runs/duo-pentobi-ladder-2579000.log 2>&1
```

Each completed opening pair is written atomically. The private Volume commits
about every45seconds and on completion. Levels are scheduled round-robin; worker
completion order never determines which openings count. Resume explicitly only
after diagnosing a failure, ensuring no active duplicate and retaining the exact
original model/config/binary. No automatic repeated retries or larger jobs.

After completed.json and hardware.json exist, confirm the app stopped, download
artifacts into a pre-created unused local directory, verify hashes and all900 pair
records, render the level/score/margin plot and compact report, inspect costs,
back up results and commit validated code/docs locally. Report results here and
remove the bounded follow-up schedule. Healthy progress needs no notification.

## Current state

**Completed and verified:** all 900 opening pairs / 1,800 games finished.
The original and approved recovery apps are stopped with zero tasks. All 820
earlier pair files are unchanged; all 1,800 games and final margins were replayed
locally and all search budgets checked. See [results, table and plot](pentobi-level-results.md).
The separate GPU continuation remains active; retain the shared follow-up for
that study. No further games in this level benchmark are authorized or needed.

Source commit da7e33985269a846cad82e8a08295876ae21432a. All70 Python tests passed,
including real parallel Pentobi games, missing-pair recovery, unchanged replayed
actions/scores and rejection of changed model/configuration. Lint/format pass.
Current provider rates were verified and saved in
runs/pentobi-ladder-current-rates-20261026.json. Ledger reserves **$3.43666**;
actual metering is separate. Source payload SHA256:
`8925ebfa95687c61d5c9c21f86f23ccdabb2834c9e5d0b167596c2ee6c1a155f`.
Runner SHA256:
`83fdedc1aee18201f9b7f8ef5adce0a3a0dd42d4b19edb43125f890c8a796b97`.
Exact executed source and runner are saved with the run-name/time prefix in runs/.

Follow-up schedule **finish-pentobi-levels-1-9-benchmark** checks every15minutes,
quiet during healthy progress, reports completion or an actionable failure, and
deletes itself after the reports for both authorized jobs. Dennis subsequently
authorized a separate controlled continuation study; see
[continuation-status.md](continuation-status.md). It does not change this
benchmark's frozen agent, openings or budget. The shared follow-up must avoid
duplicating either job and finish each report independently.
The cloud call continues independently if this Mac sleeps or disconnects.

### First scheduled follow-up

The 12:58UTC heartbeat verified healthy progress: one remote container and one
local detached CLI client, with no duplicate training/evaluation process.
The committed status downloaded during this check contained 21 completed pairs
at levels1–6,20 at7,19 at8 and15 at9 (180/900 pairs total). No completion or
failure marker is present yet. No result-dependent decisions were made.

Provider resource snapshot `runs/pentobi-ladder-billing-report-20261026T1306.json`
attributes **$0.22369045** so far to this benchmark, with CPU, RAM and network
egress only; metering can lag. The workspace summary remains $0 billed after
credits/allowances. The conservative $3.43666 reservation is unchanged.
Use the installed `modal billing summary` and `modal billing report` CLI
commands for later billing checks. A browser reader timed out during this
follow-up; the CLI checks succeeded and the remote benchmark was unaffected.

### 10:50EDT interruption audit and concrete recovery proposal

Modal preempted the CPU container shortly after90 pairs completed at most levels.
Its infrastructure restarted the function with its original resume=false input,
despite retries=0; our existing-directory guard then stopped the restart. This
was not a game-legality or score mismatch. Provider documentation confirms
preemption restarts are automatic and CPU functions can opt out at3x CPU/RAM
price: https://modal.com/docs/guide/preemption. No agent-initiated retry has run.

The stopped app has zero tasks; only the separate GPU study remains active.
Committed pair counts:92 at each level1–7,91 at8 and85 at9, totaling820 pairs
(1,640 games);80 pairs remain. All saved fingerprints, seats and seeds passed
local verification. Frozen model hash is unchanged. Do not report these partial
scores as the completed benchmark or discard any losses.
All1,640 saved games were also replayed through the local Rust engine: every
move was legal, opening prefixes matched and terminal scores/margins agreed.

Downloaded tree: runs/duo-pentobi-ladder-interrupted-20261026T1450/.
Manifest: interruption-verified.json in that directory. Backup:
.backups/pentobi-ladder-interrupted-20261026T1450.tar.gz, SHA256
`488fdfe6b7a03309d4a3c3acebaa7adcb9a701ec4994c9ef2fbf53d46c6adae6`.
Original protocol fingerprint:
`6f9db97f61307eb281b99ae7505364701316ee76abf4ea90650654063479fd3c`.
Original provider metering snapshot attributes$1.13421073 to the interrupted
app, subject to lag. Resource reservations remain$3.43666 until approval.

Prepared cloud/modal_ladder_recovery.py reuses the exact original built image
im-uHIEZ1NXBeX8O5FJeGIZqA, preserving the installed binary and Python/Rust code.
Only a new orchestration wrapper would be uploaded; no model/replay is uploaded.
It verifies the frozen protocol and resumes only missing whole pairs, with the
same8 workers/model/seeds/settings and exact original binary hash enforced by
the existing run() validator. It refuses active duplicates and completed runs.

Proposal: one45-minute non-preemptible CPU function, no GPU,retries0,8 cores24GiB.
Conservative additional reservation including90 seconds startup is$1.32618;
combined reservations$4.76284 stay below the original$5 phase ceiling. The3x
CPU/RAM pricing avoids repeating this particular infrastructure interruption.
Lint, formatting and Python compilation passed; existing interrupted-pair resume
tests already cover validation and missing-pair recovery. This new provider
wrapper has not been executed. An explicit approval flag is required.

Pending human question asks permission for this one recovery because the saved
heartbeat explicitly prohibits automatic retries. Continue monitoring the GPU
study while waiting; do not repeatedly notify about this unchanged interruption.

### Approved recovery execution

Dennis explicitly approved the prepared recovery in this chat. Executed once:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_ladder_recovery.py --approved > runs/duo-pentobi-ladder-2579000-recovery.log 2>&1
```

App ap-aceNu72e6upXDaPHCMAOsZ is CPU-only,non-preemptible,45-minute hard timeout,
retries0. The ledger now contains the original$3.43666 and recovery$1.32618
reservations: **$4.76284 total**, within the original$5 ceiling. No model/replay
or new engine source is uploaded; the only upload is the recovery wrapper.
Executed wrapper SHA256:
`226359451bb9b8e17f28fe036d68248965fa5800e1c0b945c8d818394d1a6270`.
Original experimental protocol fingerprint is unchanged. The new image
im-quaFj4Ja4cFgUL5Z867XL0 adds only the wrapper to the preserved original image.
Prelaunch app/process checks found only the intentional GPU continuation active.
No second recovery is authorized or affordable under the remaining reservation
ceiling. Keep the shared follow-up until both studies' reports are finished.

### Completion audit — 26 October, 11:50EDT

Both completed.json and hardware.json were retrieved; recovery app stopped at
11:42:12EDT, with zero tasks. The final log-tail timeout followed a successful
return and is not a missing-game failure. Fresh final download directory:
runs/duo-pentobi-ladder-final-20261026T1550/, containing the complete run,
download-sha256.json and verification.json. All 900 pair fingerprints, indices,
seats, seeds and opening prefixes verified. Every original 820 pair file matched
its interruption manifest exactly. All 1,800 games were independently replayed
and scores/margins matched. Actor counters total 3,829,248 simulations over
29,916 searched decisions, exactly 128 each, with 2,939,535 neural evaluations.
The existing certified/checkpoint manifest also matched all three entries.

Scores by Pentobi level: 27.50%,15.75%,7.75%,1.25%,0.75%,0%,0%,0%,0%.
Level 1 paired nominal95% interval:22.25%–33.00%. Levels6–9 report a conservative
one-sided paired upper bound of2.95%, rather than a misleading[0,0] interval.
This frozen model remains below level1 under the declared opening protocol.
The continuation agents are not evaluated by these results.

Provider per-app metering snapshot: original$1.13421073 plus recovery$0.69018380,
total **$1.82439453**, covered by existing credits/allowances; accounting can lag.
Conservative reservations remain$4.76284 under$5. App lifetimes were2.015h
and0.406h (2.421h combined,excluding approval idle time). Recovery function
duration1413.80s covers only the second call. The mixed-host CPUfloat32 recovery
used a PyTorch fallback for unsupported NNPACK; the final report records this
limitation and distinguishes the fresh book from earlier GPU-bfloat16 tests.

The reproducible report entrypoint python/blokus_rl/duo_pentobi_report.py
independently verifies every saved game, budget and summary before rendering.
No further training or benchmark was launched to produce this report.

Final artifacts/report/code/provider snapshots backed up locally:
.backups/pentobi-ladder-final-20261026T1550.tar.gz (1,017,169 bytes), SHA256
`22681d51c45a0d338fa1be793aed195538bdc1974b58298e95a2a42146463be6`.
The reproducible report validator, lint/format and Markdown link/diff checks
passed. The plot was visually inspected; PNG and SVG are preserved. The shared
heartbeat was narrowed to the still-running GPU continuation and will delete
itself after that study's final report.
