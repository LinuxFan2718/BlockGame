# Frozen strong agent at 2,048 simulations vs Pentobi level 3

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
