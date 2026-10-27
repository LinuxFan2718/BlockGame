# Frozen agent vs Pentobi levels 4 and, conditionally, 5

26 October 2026. Dennis requested level 4, followed by level 5 only if we win;
otherwise recommend the next step. This supersedes the completed level 3
study's restriction on further levels. No new training is part of this test.

Use the same frozen strong1024 CNN (selected additional round 112), SHA256
`95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`,
at exactly 2,048 simulations per move. Each tested level completes 100 fresh
four-ply opening pairs / 200 games with both seats, seed 2604000. If level 5
is reached, it uses this same book/seeds. Pinned Pentobi 31.0 Duo, one thread,
no opening book or resignation, default symmetric draw; CUDA bfloat16,
TF32 permitted and one PyTorch thread. No model selection or adaptive extension.

The primary outcome is score (wins + half draws)/200, with a nominal 95%
whole-opening bootstrap interval (10,000 resamples, seed 17). Report W/D/L,
win rate, score, margin, counters, runtime and costs. A level 4 score above 50%
with the interval's lower bound also above 50% triggers level 5. An upper bound
below 50% supports losing; an interval crossing 50% is inconclusive. Either
non-passing case stops the study and gets an explicit next-step recommendation,
without claiming an inconclusive result proves a loss. Complete all 200 games
of a started level regardless of interim scores. Never run level 6 or increase
search automatically. The 65,536-simulation idea remains dropped.

Only after local replay verifies all level 4 traces, settings, hashes and stopped
compute may the conditional level 5 job launch. A launcher guard reads the
verified record and refuses non-passing or incomplete evidence. This is a
sequential benchmark with nominal per-level intervals; conditional advancement
does not provide multiplicity-adjusted ladder-wide confidence or training-seed
replication. Do not infer human Elo, Classic strength or frontier status.
Level 3 used a different book; any level 3/4 comparison is not matched.

Reuse private image im-BnyLYweX6vlwu6iCpDnqrC, immutable evaluator SHA256
`80727fc7fee1fcdab75eebc989f41f617e6721d4f546e266367ca53f294a0493`
and pinned Pentobi binary SHA256
`ccb2362fc2eb066be9337b76488c527f22aae4ad4b72296ae105e54b2c015882`.
Only the updated cloud/modal_search_sweep.py runner is uploaded. Model/replay
stays in the existing private Volume blokus-deep-selfplay-20261026,
duo-continuation-865/frozen-strong1024.pt. Preserve all prior checkpoints.

Profile linuxfan2718; each level is a separate serial bounded call: one L40S,
two capped CPU cores, 8 GiB RAM, two-hour hard timeout, retries 0, max 1 container.
Rates checked at launch preparation: $1.95/GPU-hour, $0.0473/core-hour and
$0.008/GiB-hour. Reserve $4.269915 per level under a $5 per-level ceiling;
combined ceilings $10 and combined maximum reservation $8.53983. Preserve
the $50 reserve from the original $130 existing credits. Level 5 is authorized
conditionally, not a commitment to spend. No duplicate, automatic retry,
unrelated upload, top-up, plan change, subagents, messages, push or publication.
If infrastructure interrupts a job, preserve atomic pairs and diagnose/report
instead of silently restarting or changing the protocol.

Runs / ledgers:

- duo-strong-search2048-level4-2604000 / runs/modal-level4-cost-ledger.jsonl
- duo-strong-search2048-level5-2604000 / runs/modal-level5-cost-ledger.jsonl

Commands (each at most once; the second requires the verified passing record):

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_search_sweep.py --study level4 > runs/duo-strong-search2048-level4-2604000.log 2>&1
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_search_sweep.py --study level5 > runs/duo-strong-search2048-level5-2604000.log 2>&1
```

Before launch check apps, processes, billing, ledger and frozen model hash.
Each completion requires finished.json, hardware.json and a stopped app with
zero tasks. Download into a pre-created unused local parent and run the local
auditor (it launches no games):

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m blokus_rl.duo_pentobi_level_report --run <complete-run-directory> --level 4 --apps <stopped-app-snapshot.json> --app-id <app-id>
```

The auditor checks the full book, seeds, seats, legal action traces, terminal
scores/margins, exact simulation counts, all summaries, model/evaluator/binary
hashes and source commit vs exact executed runner. It verifies certified and
parent checkpoints unchanged. Level 4 writes runs/pentobi-level4-verified.json
only after these checks pass; that record controls level 5 eligibility.
After the stopping level, append results, actual budgets, failures and final
provider costs here; back up all new artifacts with every archive member hash
verified, commit validated code/docs locally and delete the quiet follow-up.
No additional training or paid study is implied by the next-step recommendation.

## Preparation checks

The generalized auditor independently replayed the completed level 3 study's
200 games and reproduced 58.25%, checking the committed runner hash as well.
The conditional launch guard accepted a complete passing audit and rejected
eight losing, tied, inconclusive, incomplete, active-app, changed-model or
changed-config cases. These checks launched no paid jobs. The unchanged
evaluator's targeted tests include a real Pentobi match and missing-pair/config
checks. Current apps are stopped and provider metering is about $31.50
($0 billed at the snapshot); accounting can lag or fluctuate.

## Level 4 launch audit

Launched once at 21:32 EDT on 26 October (01:32 UTC on 27 October): app
ap-JR0Ul9EyWFQKBom7k8Pl5D, one active task. Previous apps and local clients
were stopped before launch. The ledger reserves $4.269915. Source commit
`464ce06e4d61d1c7d75891e7a5fd0927f0abe865`; exact executed runner SHA256
`b9700b6d19a0f4660ed13fa2c12ce3dd926b73809dd6d82ba85b9172a5ad3844`.
The runner/model/evaluator checks passed and the intended 2,048-simulation arm
started. No model/replay was uploaded. Two targeted evaluator tests passed in
1.44 seconds; formatting/lint and diff checks passed. The existing level 3
study and all certified checkpoints remain preserved.

Quiet follow-up: finish-pentobi-levels-4-and-conditional-5. It monitors only this
level 4 job and the explicitly authorized conditional level 5 test, verifies
results before advancing, and deletes itself after the stopping result and
recommendation. Healthy progress and partial scores need no notification.
