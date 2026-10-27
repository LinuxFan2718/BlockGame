# Frozen strong agent at 2,048 simulations vs Pentobi level 3

The agent scored **58.25%** against Pentobi 31.0 Duo level 3, with a nominal
95% whole-opening interval of **52.25%–64.25%**. The lower bound exceeds 50%,
supporting stronger play under this protocol. All 200 games finished and were
independently replayed. This evaluates one frozen model with additional playing
search; no learning or checkpoint selection occurred.

| Simulations/move | W/D/L | Win rate | Score (draws half) | Nominal paired 95% interval | Mean margin | Mean game seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 113/7/80 | 56.50% | 58.25% | 52.25%–64.25% | +0.695 | 9.135 |

This is a fresh book of 100 four-ply openings, with both seats played per
opening. It differs from the level 2 search sweep, so the change from that
study's 75% score is not a matched-book estimate of the effect of opponent level.
The nominal interval concerns this opening distribution, not independent
training seeds, human Elo or frontier strength. No level 4 test is queued.

## Predeclared design

26 October 2026. Dennis requested the next level after the search sweep.
Test only level 3; do not increase search further, retrain or advance to level 4.

Fixed model: strong1024, selected additional round 112, SHA256
`95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`.
Use exactly 2,048 simulations per move throughout this test.
Use 100 fresh four-ply opening pairs, seed 2603000, both seats: 200 games total.
CUDA bfloat16, TF32 permitted, one PyTorch thread; pinned Pentobi 31.0 Duo
level 3, one thread, no opening book or resignation, default symmetric draw.
Every final margin is checked against Rust and every pair saved atomically.

Primary: absolute score (wins + half draws)/games, with nominal 95% whole-opening
bootstrap interval (10,000 resamples, seed 17). Report WDL, win rate, mean margin,
actual search counters and runtime too. A lower interval bound above 50% supports
beating level 3 under this protocol; an upper bound below 50% supports the reverse;
an interval crossing 50% is inconclusive. Finish all 200 games regardless of interim
scores, with no adaptive extension. This fresh book differs from the level 2
sweep; do not interpret the score difference as a matched-book effect.

Reuse the validated immutable evaluator in image im-BnyLYweX6vlwu6iCpDnqrC,
SHA256 `80727fc7fee1fcdab75eebc989f41f617e6721d4f546e266367ca53f294a0493`.
The existing runner now selects from two explicit fixed studies; the historical
sweep remains protected by its original ledger and saved executed source.
New mode: cloud/modal_search_sweep.py --study level3. Only this updated runner
is uploaded; the model/replay stays in its private Volume. Its source model
hash is checked before and after evaluation.

Run: duo-strong-search2048-level3-2603000, Volume blokus-deep-selfplay-20261026,
profile linuxfan2718. Ledger: runs/modal-level3-cost-ledger.jsonl.
One L40S, 2 capped cores, 8 GiB RAM, 2-hour hard timeout, retries 0, max 1 container.
Provider rates checked before launch: GPU $1.95/hour, CPU $0.0473/core-hour,
RAM $0.008/GiB-hour. Reserve $4.269915 under a $5 phase ceiling; retain $50 of the
original $130 existing credits. No duplicate, automatic repeated retry, model
upload, top-up, plan change, publication, push, messages to people or subagents.
Inspect apps/processes/billing first; all earlier paid studies are stopped.

The evaluator's targeted tests include a real Pentobi match, equality advancing,
the first losing level stopping, exact missing-pair recovery and rejection of
configuration changes. Re-run them and lint/compile/diff checks before launch.

Launch once:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_search_sweep.py --study level3 > runs/duo-strong-search2048-level3-2603000.log 2>&1
```

After finished.json/hardware.json and a stopped app with zero tasks, download
into an unused existing parent directory, independently replay all 200 games,
verify unchanged source/model/evaluator/binary hashes and actual budgets,
recompute summaries and intervals, append results here, save final provider
costs, back up new artifacts with verified member hashes and commit validated
code/docs locally. Preserve all prior certified checkpoints. Remove any quiet
follow-up when complete. No further paid study is queued by this test.

## Launch audit

Launched once at 20:14 EDT on 26 October (00:14 UTC on 27 October): app
ap-7FACuY7ou2YoW5eIBuZTSO. Source commit
`1ef7bc6832224b769de0535e2ce4b427c02289d0`; exact executed runner SHA256
`032f5fca06f0e67f66e1dd2e402b3bff1a3bdcedbc116c11a419edd9b2b1be6a`.
The single ledger entry reserves $4.269915. Hash and study-configuration checks
passed and the intended search 2048 arm started. No model was uploaded. All
prior apps were stopped before this launch. The two targeted tests passed in
2.10 seconds; lint/format/compile and diff checks passed.

Prepared local audit: runs/duo-strong-search2048-level3-2603000-verify.py; pass
the downloaded run directory as its argument. It launches no games. Final
provider billing must include 27 October UTC for this job, while the study's
client date remains 26 October EDT. The quiet five-minute follow-up finishes
only this level 3 test and deletes itself after final reporting or a reported
blocking failure; it must not begin level 4 or another study.

## Completed execution and local verification

Finished at 20:46 EDT on 26 October; app stopped at 20:46:32 with zero tasks.
One NVIDIA L40S, two capped CPU cores and 8 GiB RAM; PyTorch 2.14.1+cu130,
CUDA 13.0, bfloat16 inference. Function runtime was 1,844.26 seconds (30.74
minutes); arm runtime 1,843.12 seconds; app lifetime 1,905 seconds. Total game
runtime was 1,827.07 seconds, including both players and protocol overhead.
The mean game time above is not GPU-only or our-agent-only latency.

There were **3,089 searched decisions**, **6,326,272 actual simulations**,
4,489,734 neural positions and 615,615 inference batches. Every searched
decision used exactly 2,048 simulations. Summed policy search time was
1,817.66 seconds. No preemption, retry, scientific-protocol change, adaptive
extension or time-limit truncation occurred. The CLI timed out waiting for
final streamed logs after the result was written; this was a log-stream warning,
not a failed evaluation. Both completion records exist and the app is stopped.

Provider metering in the saved final snapshot: **$1.09062610**, versus the
$4.269915 conservative reservation and $5 phase ceiling. Resource metering was
$1.00858315 L40S, $0.04893907 CPU and $0.03310388 RAM. Billing used 27 October
UTC, corresponding to the evening of 26 October EDT. Metering may lag; the
separate account summary reported $31.64137374 metered and $0 billed for the
month at its snapshot. The $50 reserve remains uncommitted and all compute is
stopped.

All 100 pair records / 200 action traces were replayed through the Rust engine.
Legal moves, terminal scores/margins, fresh opening book, seeds, seats, complete
pair indices and exact simulation counts matched. Every stored summary and
nominal paired interval was recomputed. Frozen checkpoint, executed runner,
immutable evaluator and pinned Pentobi binary hashes passed, as did the
source-commit/configuration checks. The runner checked the private source
model before and after play. Certified tic-tac-toe and earlier reference
checkpoints, inherited parent weights/replay checkpoint and both selected
continuation models remain unchanged.

Complete artifacts: `runs/duo-level3-final-20261027T0049/duo-strong-search2048-level3-2603000`
(111 files / 434,523 bytes). Derived verification and every downloaded SHA256
are in that directory's parent. The verifier launches no games:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python runs/duo-strong-search2048-level3-2603000-verify.py runs/duo-level3-final-20261027T0049/duo-strong-search2048-level3-2603000
```

Provider snapshots: `runs/level3-billing-report-final-20261026.json`,
`runs/level3-billing-summary-final-20261026.json` and
`runs/level3-apps-final-20261026.json`. Replay log:
`runs/level3-final-audit-20261026.log`. Verified second local copy:
`.backups/pentobi-level3-final-20261026.tar.gz`; its sidecar records the archive
SHA256 and every archived member hash. The backup includes all new model/game
artifacts, exact executed source, verifier, provider snapshots and this report.
Each archived member was read back and checked. Local formatting/lint and diff
checks passed. The completed study's follow-up is deleted upon final reporting;
there is no further automatic evaluation or training.
