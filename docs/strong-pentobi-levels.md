# Strong continuation agent vs higher Pentobi levels — 26 October 2026

## Completed result

**Stopped at level 2, as requested.** The frozen agent scored 31.5% against
Pentobi 31.0 level 2, below the 50% stopping threshold. Its actual win rate was 28%.
No level 3–9 games were started.

| Pentobi level | Our W/D/L | Win rate | Score (draws count half) | Nominal paired 95% score interval | Mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 56/14/130 | 28.00% | 31.50% | 26.00%–37.00% | −3.505 |

This used 100 fresh opening pairs / 200 games and the same 128 playing simulations
and CUDA bfloat16 precision as the earlier 62.5% level 1 result. The books differ;
the earlier level 1 result belongs to the prior controlled study, not this new
book. Our current evidence places this fixed agent above level 1 and below level 2
under these protocols; it does not define a human Elo or an interpolated rating.
The interval is a nominal whole-opening bootstrap (10,000 resamples, seed 17),
not training-seed uncertainty or a multiplicity-adjusted level guarantee.

The function finished in 169.65 seconds; app lifetime including startup/shutdown
was 205 seconds. Provider metering at the saved snapshot is $0.11013427 (about ten
cents; accounting can lag), distinct from the $4.269915 maximum reservation.
The app ap-L0aEXeCzRNZCbf6AAbeEWo is stopped with zero tasks. The $50 reserve
remains untouched. There was no retry, preemption or protocol change; the final
CLI log-reader timeout followed a successful completed return.

All 200 action traces replayed legally, with matching scores/margins, correct
seats/seeds/openings, unchanged model hash and exactly 128 simulations for each
of 3,082 actor decisions: 394,496 actual simulations, 322,391 neural positions.
All original certified checkpoint hashes also pass. Atomic pair records,
protocol, results, hardware and frozen model were downloaded to
runs/duo-strong-pentobi-steps-final-20261026T2110/duo-strong-pentobi-steps-2599000.
The parent directory holds verification.json and download-sha256.json. The local
verifier is saved as runs/duo-strong-pentobi-steps-2599000-verify.py; rerun it with
the downloaded run directory as its argument to reproduce the audit without
playing games. Provider snapshots are runs/pentobi-steps-{apps-final,billing-report,
billing-summary}-20261026.json. Exact executed sources and cost ledger remain
under runs/. All earlier agents and results are preserved.

## Fixed design and execution

Dennis requested sequential testing until the new model falls below half.
The previous 62.5% was a **score** (wins + half draws), not a win rate; use that
same score threshold and also report actual wins/draws/losses and win rate.

Frozen model: strong1024, selected additional round 112 of the controlled
continuation, SHA256
`95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`.
No training, model selection or larger playing-search budget is involved.

Start at Pentobi 31.0 Duo level 2, then 3–9 in order. Finish 100 opening pairs
(200 games) at each level; stop at the first **complete level scoring <50%**.
Exactly 50% advances. Do not stop partway through a level based on scores.
Use 100 identical fresh four-ply opening pairs at each tested level, seed 2599000,
both seats per opening. Our agent keeps 128 search simulations, CUDA bfloat16,
TF32 permitted; Pentobi uses one thread, no book or resignation. Every final
margin is checked against Rust. Save whole pairs atomically, including moves,
seeds and actor counters. Report nominal whole-pair 95% bootstrap intervals;
the stopping rule makes this a descriptive bracket, not a simultaneous claim
that every passed level has a true score above 50%.

One bounded detached L40S call, two capped CPU cores, 8 GiB RAM, 2-hour hard limit,
retries 0, max 1 container. Conservative reservation **$4.269915**, phase ceiling
**$5**, within existing credits and retaining the $50 reserve. Expect only a
small fraction of that reservation if the agent falls below half at an early
level. Reuse immutable image im-uHIEZ1NXBeX8O5FJeGIZqA with pinned Pentobi 31.0
commit d5374ddc20eced902788e733d67ee16bd1836fd3. Upload only the new evaluation
module and runner. The frozen model stays in its existing private Volume.

Source: python/blokus_rl/duo_pentobi_steps.py and cloud/modal_pentobi_steps.py.
Run: duo-strong-pentobi-steps-2599000; private Volume
blokus-deep-selfplay-20261026; profile linuxfan2718.
Ledger: runs/modal-pentobi-steps-cost-ledger.jsonl.

Two targeted tests passed, including equality advancing, the first losing level
stopping, exact missing-pair recovery without changing saved pairs, changed
configuration rejection, and a real two-game Pentobi level 2 functionality check
with a separate seed and reduced search (excluded from benchmark evidence).
Read apps/processes/billing before launch. No duplicate, automatic repeated
retry, model upload, push, publication, top-up or subagent. Preserve all earlier
models and results. After completion and stopped app, download into an unused
directory, replay all new games, verify hashes/budgets, back up artifacts,
append results here and commit validated code/docs locally.

Launch once:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_pentobi_steps.py > runs/duo-strong-pentobi-steps-2599000.log 2>&1
```

Launched once at 17:06 EDT: app ap-L0aEXeCzRNZCbf6AAbeEWo. Source commit
`dd2e8b4d01ca36b67e91d9706453813d771faf26`; exact executed module/runner and
source hashes are retained under runs/duo-strong-pentobi-steps-2599000-*.py and
the ledger. No model was uploaded. The run completed level 2 and stopped below half. Its completed result and audit are above.

Final backup: .backups/strong-pentobi-steps-20261026.tar.gz,282,347 bytes,
SHA256 `60e8c1923cd579a2011f4aed6b0ff6c8a25374426c8ffba14c8caedb1111cdcc`.
All130 archived members were read back and matched their source hashes. The
sidecar JSON holds the full manifest. Lint/format and diff checks pass; the two
targeted tests and the complete independent200-game replay audit passed.
The completed run needs no follow-up schedule or further levels. Do not rerun
the launch command; all source models remain unchanged and all paid compute is
stopped.
