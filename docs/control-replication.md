# Independent control replication — seed 1071

Dennis authorized the [completed study’s recommendation](integrated-overnight.md)
on 28 October 2026: replicate with a fresh training seed and test book before
scaling. This study tests the cheaper 1,024-search training recipe. It starts
from the **original preserved parent**, rather than continuing the successful
seed-1070 control. No new result is available yet.

The original control improved its Pentobi-4 score by 13.75 percentage points
(nominal whole-opening 95% interval +5.5 to +22.25). That result motivated this
replication; its test games are not reused or pooled as new observations.

## Prespecified method and success

One new seed, **1071**, plans 64 rounds × 256 = **16,384 additional self-play
games**. Scientific settings match the earlier control except training,
validation and test seeds. The initial model is the selected round-112 CNN
(SHA256 `95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`),
with fresh AdamW and the complete 100,000-position replay from the later round-128
state (`df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911`).
The later replay and shared parent limit independence; this is training-seed
replication, not an independent model lineage or training from scratch.

Recorded 20% of moves use full search 1,024; unrecorded 80% use fast search 32.
Six workers × 64 live games use shared-memory bulk inference, compact Rust
indices, ready batching, CUDA Graphs and cache disabled. Inference batch 256,
leaf batch 8, zero wait, CUDA bfloat16/TF32, replay 100,000, training ratio 3,
minibatch 256, AdamW learning rate .001 / weight decay .0001, policy CE .5 plus
terminal WDL MSE, gradient clip 1. The soft training cap is 6,000 seconds; report
any cap-bound work honestly. No architecture or runtime change is being tested.

Validation every 16 rounds uses 32 pairs, seed **2608300**, against the frozen
parent at 512 simulations each. The unchanged parent remains selectable at 50%
(additional round 0, despite embedded update 112). Selected weights freeze before
any held-out book is generated. Tests use **100 fresh four-ply opening pairs,
seed 2608900**, both seats:

- Parent and selected control each against pinned Pentobi 31.0 Duo levels 3 and 4,
  at exactly 2,048 playing simulations: **800 external games with full traces**.
- Selected control against parent at 2,048 each and raw: **400 head-to-head games**
  with paired outcomes/counters, without saved action traces.

Finish every prespecified case regardless of intermediate scores. The primary
criterion is control-minus-parent Pentobi-4 draw-half score difference whose
**nominal whole-opening 95% interval lies entirely above zero**. An interval
crossing zero is inconclusive; entirely below zero is negative evidence. A
control absolute Pentobi-4 interval entirely above 50% is a stretch goal, not a
condition for stopping. Raw/search and validation/test results remain separate.

Report this fresh result alongside seed 1070 without treating different books
as a matched comparison or pooling games as independent training replicates.
There are only two training seeds total, sharing one parent and warm replay;
checkpoint selection and multiple nominal intervals further limit claims. No
human Elo, Classic/frontier or AlphaBlokus claims. No next experiment is automatic.

The complete frozen configuration is in `control-replication-protocol.json`.
The validated immutable runtime image is `im-yX3zmnEKl2EgUtndThw9li`; native SHA256
`257b5a55c2c9fefcf3af5f15e240c7b18245abe49c44b781956b2b5351dac30e`.
Only the new launcher and study helper source are uploaded; model/replay stay in
the existing private Volume. The 19 inherited payload hashes remain pinned to the
completed GPU validation. Initial replay content/order, parent tensors, fresh
optimizer and fresh RNG receive a retained audit record before training starts.

## Budget and completion

One L40S, eight capped CPU cores, 24 GiB RAM, **four-hour hard limit**, retries 0,
max one container. Phase ceiling **$11**; reservation **$10.29461**, comprising
four hours plus 90 seconds at $2.5204/hour and $0.15 build allowance. Reservation
is not actual spend. The account last metered $53.03137374; the launcher uses a
conservative $54 floor and checks total commitment below $80, retaining **$50 of
the original $130 credits**. No top-up or plan change.

Ledger: `runs/modal-control-replication-cost-ledger.jsonl`. The local ledger and
exclusive private run directory block duplicate submission/infrastructure replay.
No automatic retry or recovery. On failure preserve atomic states and logs,
report the blocker and stop. Existing studies and checkpoints remain unchanged;
`runs/control-replication-preserved-before.json` pins their protected hashes.

After both finished markers and stopped app / zero tasks, download to a new,
pre-created unused local parent. Run:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m blokus_rl.duo_control_replication_report \
  --run <complete-run-root> --apps <stopped-apps.json> \
  --app-id <app-id> --billing <billing-report.json>
```

The auditor launches no games. It checks source/runtime/native/model lineage,
fresh initial optimizer/replay/RNG, selection, all gradient/game/search counters,
and independently replays all 800 external traces. It recomputes every paired
summary and audits the 400 head-to-head outcomes/counters without claiming action
replay. All-zero scores use a conservative one-sided paired upper bound.
`cloud/write_control_replication_results.py --verification <verification.json>`
writes the compact report and figure; inspect the plot and add interpretation,
actual costs, limitations and a recommendation. Check CLI billing including all
attempts, stopped apps and preserved models. Back up with
`cloud/backup_control_replication.py --name <unused> --run <download-parent>`;
every archive member must pass SHA256 readback. Commit locally without pushing,
report the final result here, and delete the completion follow-up.

## Launch and follow-up

Submitted **28 October 2026 at 11:53 EDT** (15:53 UTC):

- App `ap-oDPcA20WshoYQlqxcurDXF`; call `fc-01M61KFFT1D23GBPCZQJJP7PV6`.
- Run `duo-control-replication-1071`, profile `linuxfan2718`, existing private
  Volume `blokus-deep-selfplay-20261026`.
- Executed source commit `f4a3dd5eae44537731c4e37dd01ee93ee353d1e7`.
- Provider confirmed the exact child SDK/Volume commit probe and started the
  independent control branch. One active container; previous study stopped.
- Offline paid-launch guard tests: four passed. Only the three declared seed
  fields differ from the earlier control configuration; new validation/test books
  do not overlap that run's books. All 35 protected checkpoint hashes passed.
  The initial replay-content digest was checked locally, and report/plot generation
  passed a temporary historical-fixture smoke test; that fixture is not new evidence.
- Reservation is $10.29461 under the $11 ceiling. Actual metering is reported at
  completion. Expected work is roughly 75–100 minutes based on the earlier run;
  this is an estimate, with a four-hour maximum function lifetime.
- A quiet completion follow-up audits, reports, backs up and commits the result,
  then deletes itself. Partial training/validation scores are not final evidence.

Startup verification at **11:56 EDT** read committed initialization/status records:
parent tensors, all inherited replay content/order, empty AdamW state and fresh
CPU/NumPy RNG matched locally; one CUDA RNG state was retained. Three rounds /
768 new games were committed. These are health/counter checks, not strength
results. Snapshot: `runs/control-replication-startup-snapshot-20261028T155635Z/`.
Initial source/receipt backup `.backups/control-replication-startup-20261028.tar.gz`
read back all 41 members successfully (SHA256
`e24466e8b1620d8848eb2919860dcc0656f2b851e8bff22c0214e45372193874`).
