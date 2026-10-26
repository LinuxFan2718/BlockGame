# Deep self-play overnight status

User-authorized overnight phase,25 October2026. Methods and budget:
docs/deep-self-play.md. Chat explanation: docs/alphablokus-explanation.md.
Original study: docs/alphablokus-study.md. Earlier models remain unchanged.

## Validated and launching

- All67 Python tests pass; lint and formatting pass. New tests cover real-process
  shared inference parity, full-game move indices, learning both heads, batched
  evaluation, atomic resume/selected checkpoint recovery, time-limit selection,
  and Pentobi pair resume with model/config guards. No Rust rules were changed.
- L4 benchmark finished:0.99 games/s with one worker,2.30 with six. L40S finished:
  2.26/5.28 games/s, about20% lower requested-resource cost/game than L4 in this
  screen. These include different host CPUs, not isolated GPU comparisons.
- L40S passed actual CUDA/bfloat16 resumed training:24 games in two rounds,
  finite policy/value losses. Smoke weights discarded; no held-out seeds used.
- Benchmark apps ap-RoaDV7u20Yt9BGIxn9Kqk8 and ap-9R1Y2Qelk553FZkSeB6RlK are
  stopped, zero tasks. No duplicate training process existed before launch.
- Selected hardware: one L40S,8 capped CPU cores,24GiB capped RAM,6 search workers
  and up to256 positions/inference batch. Model:10 residual blocks×64 channels,
  986,205 parameters. GroupNorm, inventory FiLM and normalized board coordinates.
- Frozen planned config runs/deep-study-config.json:128 rounds×256 games/architecture,
  max2.5hours training each. Deep then shallow serially, matched actual deep round
  count. Full256-search targets on20% of moves, cheap32-search on80%, no expert
  demonstrations. One seed764: report this limitation.

## Cloud run and limits

Runner cloud/modal_deep.py; app blokus-deep-selfplay-study; private Volume
blokus-deep-selfplay-20261026. Main artifact folder duo-deep-study-764. The hard
function timeout is7hours, retries0, one container. Both architectures and the
reserved held-out/Pentobi evaluations run serially in that one call.

New overnight phase cap: $35 conservative resource reservations, within Dennis's
$130 existing credits; no top-ups or plan changes. Profile linuxfan2718.
Ledger runs/modal-deep-cost-ledger.jsonl. Only allowlisted updated source and the
same explicitly approved original frontier342 checkpoint leave this Mac. Pentobi
is fetched from its public v31.0 source, pinned d5374ddc20eced902788e733d67ee16bd1836fd3.
No AlphaBlokus code/weights are used for initialization or uploaded.

Command being launched after this validated source commit:

```sh
.venv-modal/bin/modal run --detach --profile linuxfan2718 cloud/modal_deep.py --stage study --gpu l40s --run-name duo-deep-study-764 > runs/duo-deep-study-764.log 2>&1
```

## Follow-up instructions

Check app/process state, ledger and this log before any action. Do not launch a
second study while one is active. Healthy progress needs no notification.

Private Volume paths under /duo-deep-study-764/:
- residual/status.json, progress.jsonl, resume.pt, best.pt, latest.pt, completed.json.
- conv/ equivalents (starts after deep completes, capped to its actual round count).
- frozen-residual.pt/frozen-conv.pt, heldout-protocol.json, heldout-*.json.
- telemetry.jsonl, rules-crosscheck.json, finished.json.

Fetch small status files with modal volume get ... REMOTE_PATH LOCAL_PATH,
--profile linuxfan2718; avoid overwriting an earlier snapshot silently. A failure
must be diagnosed; after confirming no active duplicate, resume with exactly the
same local config using --resume. Atomic checkpoint includes optimizer/replay/RNG
and selected-model state. Reservations count retries too; never bypass the $35 cap.

After finished.json appears, download the entire new artifact folder to an unused
local runs directory, verify frozen checkpoint hashes against heldout-protocol,
freeze the compared checkpoints, and evaluate against preserved warm round8 locally
using batched evaluation. Keep raw/search results and validation/test separate.
The cloud evaluates raw/search128 against original reference and Pentobi31.0
level1 on100 paired held-out openings, plus the primary deep-vs-shallow128 case.
Pentobi partial cases resume at whole-pair boundaries and validate model/config.
Normal-start games are saved separately without a population CI.

Build a compact results plot/report with actual budgets, methods, failures,
uncertainty and costs. Preserve all negative evidence. One training seed and many
secondary comparisons limit conclusions. Back up new artifacts locally, check
billing and stopped apps, commit validated code/docs locally, report results here,
and delete the follow-up schedule. No pushing, publishing, outside messages,
subagents or indefinite larger training series. Further hypotheses need a bounded,
predeclared design and remaining authorized time/credits; do not tune on these tests.
