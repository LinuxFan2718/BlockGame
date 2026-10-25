# Final overnight handoff — 2026-10-25 01:48 EDT

Automatic continuation `continue-blokus-rl-experiments` was DELETED successfully.
The two last hypotheses (frozen-neural mixture and convolutional screen) failed
their declared improvement criteria. This follows the user's condition to keep
working while ideas keep working, and the screen's explicit stopping rule. Do
not silently restart the automation. No training jobs remain intentionally active;
all four final training sessions returned exit code 0.

## Start here

Read `docs/overnight-summary.md` for findings, playable models and commands.
The Rust engine and training request is complete. Strong earlier models and
all negative evidence are preserved for human review.

## Preserved models

- Tic-tac-toe: `runs/self-play-study/alphazero-seed-42/best.pt` and seed 43's
  `best.pt`. Independently certified never to lose from a normal start, both
  seats, raw greedy policy. Search can regress. `docs/self-play-results.md`.
- Blokus frontier policy: `runs/duo-frontier-training-342/best.pt`.
  Held-out strict wins: random 98.25%, largest 82.75%, frontier 73.25%.
- Blokus largest specialist: `runs/duo-longer-margin-243/best.pt`.
  Held-out strict wins: random 99.15%, largest 91.35% (2,000 games each).
These are opponent-specific results, not optimality or human-level evidence.

## Completed evidence

All three independent staged curricula passed: longer-margin largest wins
86.9%, 91.35%, 82.35%; frontier stage frontier wins 73.25%, 69.75%, 63.5%.
Main tables/curves: `docs/blokus-duo-results.md`, generated with `duo-report`.

The paired neural cross-play probe challenged heuristic-only rankings. Six
models played exactly two deterministic normal-start games per pair. Three
frontier policies also played 100 shared randomized four-ply openings in both
seats. Bootstrap resamples whole pairs. Results/protocol:
`docs/blokus-duo-crossplay-{results,protocol}.md`, data
`runs/duo-crossplay-1790000/`. Original probe predates extra per-pair outcome
retention; aggregate results and replayable examples remain valid.

Frozen-neural matched study: parent held-out score 44.5%; continued-frontier
control 35.5%; mixed opponents 45%. Mixed gain vs parent +0.5 points with paired
95% interval −8.5 to +9.5 points, plus a frontier score loss of 11.5 points.
FAILED. Common parent/test opening hashes verified.
`docs/blokus-duo-frozen-league-{results,protocol}.md` and
`runs/duo-frozen-league-comparison.json`. Both 9,600-game arms completed. Their
original in-memory checkpoint format was 2; later metadata enhancements do not
retroactively apply to them.

Matched cold-start convolution screen: 1,600 games per actor, seed 542,
frontier opponents, terminal margin reward. Selected MLP frontier score 44%,
CNN 18%; CNN random score 94.25%. FAILED its >=5-point frontier gain criterion.
MLP has 126,043 parameters, CNN 52,978, and CNN took more compute.
`docs/blokus-duo-convolution-{results,protocol}.md` and
`runs/duo-convolution-comparison.json`. A small-budget failure is not a general
rejection of convolution. Do not replace the earlier strong policies with it.

## Code and verification

Rust `src/duo.rs` remains the single Blokus rules implementation. Independent
cell-checker tests verify optimized legal moves. PyO3 -> PettingZoo -> PyTorch /
TorchRL PPO remains the common stack. Latest full suite: 52 Python tests passed;
20 Rust tests/core and binding clippy previously passed, with no subsequent Rust
rules changes. Ruff passes.

Observation version 2 includes relative board, inventories, starts, seat, scores
and pass count (634 inputs). Legacy checkpoints gain three zero-weight columns;
four benchmark rechecks gave identical W/D/L and margins after migration.
Checkpoint format 3 records additive/conv architecture and supports older formats
1/2. Default actor is still additive. CNN's native orientation/anchor mapping,
both observer perspectives, cell permutations, legal masking, finite PPO updates
and checkpoint predictions are tested. End-to-end smokes passed.

New cross-play and frozen-league tools are reusable:
`.venv/bin/python -m blokus_rl.duo_crossplay --help`
`.venv/bin/python -m blokus_rl.duo_league --help`
Standard CLI: `.venv/bin/blokus-rl duo-play`, `duo-evaluate`, `duo-train`, `duo-report`.
Select CNN with `--architecture conv`; its interface remains the same.

Use installed .venv directly. Native rebuild only if native sources change:
`env UV_CACHE_DIR=/private/tmp/blokus-uv-cache CARGO_HOME=/private/tmp/blokus-cargo-cache uv sync --locked --offline --reinstall-package blokus-rl`.
The repo started empty; all source files are untracked, and runs/ is ignored.
Preserve both. No paid cloud training, publishing, messages to people, subagents,
commits or PRs were used. Nothing requires approval or user action immediately.
