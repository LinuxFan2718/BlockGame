# Search and bounded Modal study complete — 25 October 2026

Subsequent overnight phase also complete: [deep self-play results](deep-results.md)
and [completion audit](deep-status.md).
The completed experiments below remain frozen; do not restart them.

Read `docs/search-results.md` for the compact results and plot,
`docs/search-experiment.md` for methods, and `docs/modal-experiment.md` for cloud
readiness and upload authorization. Both local training runs, their held-out
comparisons, and the bounded checkpoint-selection follow-up are complete.
A process check found no training/evaluation/Modal jobs still active. Do not
restart or duplicate completed runs.

## Authorization and current cloud step

Dennis authorized continued RL work and up to **$130 of existing Modal credits
if worthwhile**, explicitly choosing profile **linuxfan2718**. First benchmarking
stage is capped at **$5 estimated**, with bounded single-GPU functions. No top-ups,
plan changes, spending beyond the authorized credits, publishing, pushes,
messages to people or subagents are authorized.

**Automatic approval review rejected the first Modal launch** because credit
permission did not explicitly authorize uploading private source/model data.
The human was asked to approve the allowlisted source archive and one checkpoint.
Dennis subsequently answered **“Yes, upload the code and model”** to that exact
upload question. This resolves the export block for the prepared allowlisted
source and frontier342 checkpoint. No upload or paid compute occurred before
that approval. Read-only app listing returned no active apps.

The previous follow-up schedule was deleted after the local report. The approved
L4 and H100 benchmarks and matched Mac CPU measurements have now completed.
Both Modal apps are stopped; no cloud training job or follow-up schedule is active.
Provider-reported app usage totals $0.05911950 (about six cents), subject to billing
updates. See [the cloud assessment](modal-experiment.md) and
[hardware tables/plot](modal-results.md). Do not duplicate these completed studies.

## Completed evidence

- Source baseline:896a7e0; initial search implementation:24a915c;
  benchmark/report preparation:c8ba2dc. Further validated changes are saved in
  the final local commit. No remote is configured and nothing is pushed.
- `duo-az-cold-642` and `duo-az-warm-643`: **4096 training games each**, 32 rounds,
  128 simulations/move,3 CPU workers per run. About35.59million simulations and
  29.69million neural positions across both runs; about32minutes concurrently.
  Both have completed.json and atomic optimizer/replay/RNG resume bundles.
- Cold selected round24: raw scores90.5% vs its untrained initial network and
  59.5% vs the largest-piece heuristic, but40.75% vs original frontier342.
  With128-simulation search it scores39.75% vs equally searched frontier342.
  Cold start learned useful behavior; it did not become our strongest agent.
- Warm selected round8: raw49% vs initial frontier342, paired95%CI42.5–55.25%.
  With128-simulation search, **61.25% vs equally searched frontier342**,
  115W/15D/70L, paired95%CI55–67.25%. This is an exploratory positive result for
  the searched agent, not evidence of improved raw play or universal strength.
- Outside benchmark is Pentobi31.0 level1, no book/resign, one thread, explicit
  rotation/pass translation and independent final-score checks. Warm raw/search
  scores7%/9.25% on reserved seed2549000, original reference2.5%/6.75% on identical
  openings. Even the improved agent remains much weaker than Pentobi level1.
- Original unchanged frontier342 with512 simulations scored71.5% against its raw
  policy on200 games, paired95%CI66.75–76.5%, and13% against Pentobi on matched
  seed2551000, CI8.5–17.75%. These are different comparisons/openings from the
  trained-agent tests; do not rank their percentages across datasets.
- All held-out cases use100 randomized four-ply openings in both seats. Bootstrap
  resamples whole pairs. One training seed per condition, nominal intervals,
  multiple exploratory comparisons; these are not human ratings or guarantees.

## Completed follow-up and performance work

- `runs/duo-az-warm-search-selection-2559000/`: bounded, zero additional training.
  Eight existing checkpoints were compared using40 validation opening pairs,
  both agents with128 simulations. Round24 selected (58.125% validation score).
  Fresh seed2559000,100 pairs: searched49.5%, CI43.75–55.25%; raw52%, CI45.75–58.01%.
  **Failed** its declared primary criterion. Preserve round8 and this negative
  evidence; do not promote round24. It is not a replication of round8.
- Profiling the original small-model pipeline found about4% of time in neural
  evaluation; most time was duplicate move generation and Python object/array
  conversion. The new native packed observation buffers reuse cached search
  actions and remove those conversions. Same128 decisions and16384 simulations:
  old4.70/4.45seconds, new1.62/1.62seconds, approximately**2.8x faster**.
  The action sequences, search visits/priors/Q values and observations match
  the reference path exactly in the checks. No change to game rules or weights.
- Native feature buffers have independent reference tests across complete games,
  both seats and forced passes. bfloat16 inference crosses NumPy safely.
  **60 Python tests and21 Rust tests pass**; binding/core clippy and formatting pass.
- Regenerate: `.venv/bin/python -m blokus_rl.duo_search_report`.
  Figures: `docs/assets/search-results.{png,svg}`. Data summary:
  `runs/duo-search-summary.json`. Reports preserve both positive and negative results.

## Preservation

- Original certified tic-tac-toe and frontier checkpoints remain byte-identical
  to the pre-search archive; hashes in runs/search-preserved-checkpoints.json.
- `.backups/runs-before-search-20261025.tar.gz`:824 original files, about51MB.
- `.backups/search-study-20261025.tar.gz`:163 completed study/profile files,
  78,286,589bytes, SHA256c3516dfe88ded2924078bac9eb6564bf4989b57de942653ecd8160be2944ef5e.
- `.backups/search-selection-20261025.tar.gz`: completed selection follow-up,
  4,709,648bytes, SHA256a1d01a595c98f292dec203bb9a0f9f13beb259c6a3caf3aaeec5b470c29a352c.
  Backups are local, not off-device protection.

## Hardware decision and next hypothesis

The existing additive search achieves76.74 decisions/s on the Mac versus32.71
on L4 and64.22 on H100 in CUDA fp32. Keep it local. The untrained CNN achieves
21.83/31.06/66.73 respectively; H100 is faster but L4 delivers about1.86x more
CNN search decisions per estimated steady execution dollar. CPU hosts differ,
so these are complete-system measurements, not isolated GPU speed ratios.

The prepared runner executed successfully with the exact allowlisted source and
checkpoint. Both remote runs remained inside their300second limits. No crash,
retry, duplicate app, top-up or model promotion occurred. The provider resource
report, original executed source/script, hashes and cost ledger are saved locally.
New report and benchmark utilities pass lint/format; unchanged model hashes were
verified again. Original60 Python/21 Rust test results still apply to the unchanged
game/training code.

A concrete future scaling hypothesis is multiple CPU search workers sharing GPU
inference, measured against CPU-only workers with the same simulations. A future
CNN learning trial also needs a held-out strength test. Neither is running: this
bounded study is finished, and no indefinite training series was launched.
