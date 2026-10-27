# Frozen agent vs Pentobi levels 4 and, conditionally, 5

**We lost to level 4 and stopped; level 5 was not launched.** At 2,048
simulations per move the frozen agent scored **33.75%**, with a nominal paired
95% interval of **27.75%–39.50%**. The upper bound is below 50%, meeting the
declared criterion for a loss under this protocol. All 200 games completed and
passed independent replay; this is not a truncation or inconclusive result.

| Pentobi level | Simulations/move | W/D/L | Win rate | Score (draws half) | Nominal paired 95% interval | Mean margin | Mean game seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2,048 | 63/9/128 | 31.50% | 33.75% | 27.75%–39.50% | -3.655 | 9.496 |
| 5 | — | — | — | Not run: condition failed | — | — | — |

Our prior level 3 test supported a win; this test establishes a practical
benchmark to improve on. The opening books differ, so the two scores are not
a matched estimate of the opponent-level effect. These nominal intervals do
not establish strength over all openings, independent training seeds, human
Elo or the Blokus frontier.

## Recommended next experiment

Return to self-play learning with the existing small CNN. Compare 2,048 versus
1,024 simulations on the recorded training roots, starting both branches from
the same best checkpoint with otherwise matching settings and additional-game
budgets. Keep the 32-simulation fast moves unchanged. Use separate validation
openings for checkpoint selection, then freeze both candidates before a new
held-out book against the preserved parent and Pentobi levels 3 and 4 at the
same 2,048 playing simulations. Report compute and cost differences too: equal
games are not equal compute. A later replication would be needed to establish
that gains survive another training seed.

Why this hypothesis: the [controlled continuation](continuation-results.md)
already found a benefit from 1,024 versus 256 training search, and the
[playing-search sweep](playing-search-sweep.md) found useful improvement up to
2,048 for the resulting frozen agent. Together they motivate testing stronger
self-play guidance, without proving that training with 2,048 will work.
The learning aim is to make the network reproduce better search move
distributions and predict outcomes from those games, which is the
[AlphaZero training loop](https://arxiv.org/pdf/1712.01815). That improves the
policy and value estimates used by future search. It is a hypothesis about
further learning, not a gain measured by today's frozen-model tests.

Primary success should be a positive paired score difference against level 4
versus the preserved model at the same playing budget, with a nominal interval
above zero. Beating level 4 outright (score interval above 50%) is a separate,
more ambitious target. This recommendation is not an automatic training
launch; the current bounded evaluation is complete and the reserve stays intact.

## Predeclared design

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

## Completed execution and verification

Finished 26 October at 22:04 EDT; app stopped at 22:04:33, zero tasks. One L40S,
two capped CPU cores, 8 GiB RAM; PyTorch 2.14.1+cu130, CUDA 13.0, bfloat16.
Function runtime 1,916.18 seconds (31.94 minutes), arm wall time 1,915.25 seconds,
app lifetime 1,954 seconds. Total game runtime was 1,899.21 seconds, including
both players and protocol overhead; mean game time is not pure GPU or our-agent
latency. Policy search time summed to 1,886.17 seconds.

Actual execution: 3,099 searched decisions, 6,346,752 simulations, 4,510,302
neural positions and 620,249 inference batches. Every searched decision used
exactly 2,048 simulations. No preemption, retry, scientific-protocol change or
time-limit truncation occurred. The CLI timed out waiting for final streamed
logs after the records were committed; completion, hardware and stopped-app
evidence confirm the evaluation succeeded.

Provider metering in the saved snapshot was **$1.13514112**: $1.04976059 L40S,
$0.05092685 CPU and $0.03445368 RAM, versus $4.269915 reserved under the $5 cap.
The level 5 reservation was never incurred. The separate account summary
reported $32.72137374 metered and $0 billed at its snapshot; accounting can lag
or fluctuate. Billing includes 27 October UTC, corresponding to 26 October
EDT. All compute is stopped and the $50 reserve remains uncommitted.

The auditor independently replayed all 100 whole pairs / 200 action traces and
checked legal moves, scores/margins, seeds, seats, the full fresh opening book,
exact actor simulation counts and every saved summary/interval. Frozen model,
immutable evaluator, Pentobi binary and executed runner/source-commit hashes
passed. Certified tic-tac-toe, reference342, parent weights/replay checkpoint
and both continuation models remain unchanged. The runner checked the private
source model before and after play. The actual verified record was submitted
to the level 5 launch guard, which rejected it as intended without launching
compute. No level 5 ledger exists.

Complete download: `runs/duo-level4-final-20261027T0210/duo-strong-search2048-level4-2604000`
(111 files / 435,516 bytes). The parent contains verification.json and a full
download SHA256 manifest. Reproduce the local audit without playing new games:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m blokus_rl.duo_pentobi_level_report --run runs/duo-level4-final-20261027T0210/duo-strong-search2048-level4-2604000 --level 4 --apps runs/level4-apps-final-20261026.json --app-id ap-JR0Ul9EyWFQKBom7k8Pl5D
```

Replay log: runs/level4-final-audit-20261026.log; conditional decision:
runs/level4-advancement-decision-20261026.json. Provider snapshots:
runs/level4-apps-final-20261026.json, runs/level4-billing-report-final-20261026.json
and runs/level4-billing-summary-final-20261026.json. Verified second local copy:
.backups/pentobi-level4-final-20261026.tar.gz; the sidecar stores the archive
SHA256 and every member hash. Every archived member was read back and checked.
The backup includes all new game/model artifacts, the conditional audit record,
exact executed source, auditor, reports and provider snapshots. Formatting/lint
and diff checks passed; results are committed locally without pushing. The
completed study's quiet follow-up is deleted upon final reporting.
