# Frozen-agent playing-search sweep — 26 October 2026

Dennis approved the proposed search-budget experiment. This changes thinking
time only: the selected strong1024 small CNN is frozen; no training occurs.
The previous level 2 test scored 31.5% at 128 simulations, on a different book.

Complete 100 fresh four-ply opening pairs at each of 128, 512, 2,048 simulations:
200 games/budget, 600 total, seed 2601000, identical book/seats/Pentobi seeds across
budgets. Pentobi 31.0 Duo level 2, one thread, no book/resignation; our inference
is CUDA bfloat16, TF32 permitted. Keep all results and finish all three budgets
regardless of interim scores. Compare runtime as well as strength.

Primary: paired score difference 2048 minus 128; lower nominal 95% paired interval
above zero supports an improvement for this frozen model/book distribution.
Secondary: 512 minus 128, 2048 minus 512, absolute scores/WDL/margins and runtime.
Resample whole opening pairs together across budgets (10,000 draws, seed 17).
Report absolute-score intervals too; winning above 50% against Pentobi is a
separate question. These are nominal intervals for one model, not training-seed
replication, a guarantee of monotonic improvement or a human Elo rating.

Model SHA256:
`95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`.
It remains in the private Volume blokus-deep-selfplay-20261026, at
duo-continuation-865/frozen-strong1024.pt. Reuse immutable image
im-BnyLYweX6vlwu6iCpDnqrC and its validated evaluator SHA256
`80727fc7fee1fcdab75eebc989f41f617e6721d4f546e266367ca53f294a0493`.
Only cloud/modal_search_sweep.py is uploaded; no model/replay or unrelated files.
Each arm retains atomic whole-pair traces and native/actor budget counters.

One detached L40S, 2 capped cores, 8 GiB RAM, 2-hour hard timeout, retries 0,
max 1 container. Rates verified from provider billing: GPU $1.95/hour,
CPU $0.0473/core-hour, RAM $0.008/GiB-hour. Reserve $4.269915 under a $5 phase ceiling;
keep $50 of the original $130 credits uncommitted. No duplicate, automatic repeated
retry, larger study, top-up, publication, push, messages to people or subagents.
If interrupted, preserve committed pairs and diagnose before any recovery.

Run: duo-playing-search-sweep-2601000. Ledger:
runs/modal-search-sweep-cost-ledger.jsonl. Before launch check apps, processes
and provider billing. Existing playing/training studies are complete and stopped.
The underlying evaluator's targeted tests include a real Pentobi match, exact
missing-pair recovery and configuration rejection. Re-run those tests plus
lint/compile checks before launching the new orchestration wrapper.

Launch once:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_search_sweep.py > runs/duo-playing-search-sweep-2601000.log 2>&1
```

On finished.json/hardware.json and stopped app, download into an unused existing
parent directory, verify model/evaluator/binary hashes and exact shared book,
replay all 600 games, check every simulation count, recompute paired differences,
produce a compact score/runtime plot and results below, inspect actual costs,
back up all new artifacts and commit validated code/docs locally. Preserve every
earlier model. Remove any follow-up after final reporting; no further automatic
training or benchmark is part of this authorization.
