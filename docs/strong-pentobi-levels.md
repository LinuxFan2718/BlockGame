# Strong continuation agent vs higher Pentobi levels — 26 October 2026

Dennis requested sequential testing until the new model falls below half.
The previous62.5% was a **score** (wins + half draws), not a win rate; use that
same score threshold and also report actual wins/draws/losses and win rate.

Frozen model: strong1024, selected additional round112 of the controlled
continuation, SHA256
`95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`.
No training, model selection or larger playing-search budget is involved.

Start at Pentobi31.0 Duo level2, then3–9 in order. Finish100 opening pairs
(200 games) at each level; stop at the first **complete level scoring<50%**.
Exactly50% advances. Do not stop partway through a level based on scores.
Use100 identical fresh four-ply opening pairs at each tested level, seed2599000,
both seats per opening. Our agent keeps128 search simulations, CUDA bfloat16,
TF32 permitted; Pentobi uses one thread, no book or resignation. Every final
margin is checked against Rust. Save whole pairs atomically, including moves,
seeds and actor counters. Report nominal whole-pair95% bootstrap intervals;
the stopping rule makes this a descriptive bracket, not a simultaneous claim
that every passed level has a true score above50%.

One bounded detached L40S call, two capped CPU cores,8GiB RAM,2-hour hard limit,
retries0,max1container. Conservative reservation **$4.269915**, phase ceiling
**$5**, within existing credits and retaining the$50 reserve. Expect only a
small fraction of that reservation if the agent falls below half at an early
level. Reuse immutable image im-uHIEZ1NXBeX8O5FJeGIZqA with pinned Pentobi31.0
commit d5374ddc20eced902788e733d67ee16bd1836fd3. Upload only the new evaluation
module and runner. The frozen model stays in its existing private Volume.

Source: python/blokus_rl/duo_pentobi_steps.py and cloud/modal_pentobi_steps.py.
Run: duo-strong-pentobi-steps-2599000; private Volume
blokus-deep-selfplay-20261026; profile linuxfan2718.
Ledger: runs/modal-pentobi-steps-cost-ledger.jsonl.

Two targeted tests passed, including equality advancing, the first losing level
stopping, exact missing-pair recovery without changing saved pairs, changed
configuration rejection, and a real two-game Pentobi level2 functionality check
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
