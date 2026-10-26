# Pentobi levels 1–9 benchmark — 26 October 2026

Dennis explicitly requested testing our best model against every Pentobi level.
This is evaluation only: no training or checkpoint selection. Earlier studies,
models and certified tic-tac-toe checkpoints are preserved.

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

**Running:** app ap-8J7F0oIUjPAAYL4nbs591O, started26 October08:42EDT
(12:42UTC), detached. Exactly one app/container is active. First completed pairs
at multiple levels and committed Volume progress were verified; the model's
original private-Volume hash passed. No new model was uploaded and no training
was started. Do not launch another copy.

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
deletes itself after the report. No new experiment follows this benchmark.
The cloud call continues independently if this Mac sleeps or disconnects.
