# Active search work — 25 October 2026

New authorization: Dennis asked to implement and try search and self-play after
Claude's review. The old overnight stop applies to the completed PPO screens.
Read `docs/search-experiment.md`, this file, then current logs before any work.
Do not launch duplicates. No subagents, paid cloud, publishing or messages to people.

## Running

- Cold-start self-play: `runs/duo-az-cold-642.log` and
  `runs/duo-az-cold-642/progress.jsonl`. 32 rounds × 128 games = 4096 games,
  128 simulations per move, 3 workers. Process session 96235.
- Warm-start self-play: `runs/duo-az-warm-643.log` and
  `runs/duo-az-warm-643/progress.jsonl`. Same budget, initial frontier-342.
  Process session 34123. Parent PIDs49482 (cold) and49491 (warm) were checked;
  six one-thread workers were active, without duplicates. These are two
  intentional different arms, not duplicate training jobs.
- Search-budget confirmation completed: `runs/duo-search-confirm-2542000/`;
  128 scored60.25% vs raw; 512 scored71.5% (pairedCI66.75–76.5%).
- External search benchmark completed: `runs/duo-pentobi-search128-2551000.log`;
  100 paired openings vs Pentobi level1: 9% score.
- Matched external raw benchmark completed:
  `runs/duo-pentobi-raw100-2551000.log`, 6.5% score. Difference +2.5points,
  pairedCI+0.25 to +5.0. Independent raw seed2552000 score7.25%.
- Higher-budget outside comparison completed:
  `runs/duo-pentobi-search512-2551000.log`. 24W/4D/172L, score13%,
  paired95%CI8.5–17.75%; gain over matched raw +6.5points,
  paired95%CI+3.25 to +10.0. 391 seconds, not matched compute against Pentobi.
- Matched reserved-seed Pentobi reference evaluations started in this follow-up:
  `runs/duo-pentobi-reference-raw-2549000.log` (session78676),
  `runs/duo-pentobi-reference-search128-2549000.log` (session62975).
  They use unchanged frontier342 weights and the same reserved openings that
  selected self-play checkpoints will face. They cannot affect selection or
  the already-fixed training configurations.

Heartbeat check at15:26EDT found both training parents and all six workers
healthy, both around round24 of32. No resumption or duplicate training was needed.
By the report snapshot both passed3328games. Cold validation vs its random
initial model is high; warm validation against frontier342 fluctuates. Neither
is held-out evidence. Let the remaining bounded rounds finish.

Follow-up schedule `finish-blokus-search-and-self-play-study` was created for
this chat every15minutes. It should finish these bounded runs, evaluations and
report, then delete itself. It must not restart the deleted overnight automation
or silently begin an indefinite series of training jobs.

## Completed

- Rust PUCT tree + batched PyTorch bridge and self-play trainer implemented.
- Native on-demand search; no exhaustive Blokus state table, training oracle or
  heuristic opponent. Exact endgame tests both perspectives pass.
- 57 Python tests pass, including bit-identical resumption after a simulated
  between-round interruption. All21 Rust tests pass; core clippy passes.
- Five old-policy search-budget arms: see run results; 128-sim fresh confirmation
  116W/9D/75L, score60.25%, paired95%CI56–64.5% over200 games.
- Pentobi31.0 source pinned and built. Adapter cross-checks a complete legal game
  and final margin. Raw policy vs level1 on fresh200 games: 12W/5D/183L,
  score7.25%, pairedCI4–11%. This challenges heuristic-only rankings.
- CPU/MPS inference benchmark favors CPU for current tiny model; 6 worker cores
  are used for two training arms. No conclusion about CNNs or large GPU jobs.
- Existing runs backed up locally before search work; certified TTT preserved.
- Pentobi seeding was checked by replaying the identical seeded match with an
  unrelated match in between; the complete move sequence and outcome matched.
  Its set_random_seed command resets existing random generators, not only new ones.

## Next

1. Let both bounded 4096-game runs finish. Check logs for exceptions. They have
   atomic round checkpoints. A fresh run command cannot reuse an existing folder;
   `--resume` requires the exact original flags from protocol.json and loads the
   completed round's weights/replay/optimizer/RNG from resume.pt. Smoke run predates
   this final resume-bundle enhancement; do not resume that obsolete smoke.
2. Freeze best.pt and final.pt; evaluate best on **reserved** test seed2549000 with
   100 four-ply paired openings, raw and search128 vs initial frontier342, plus
   heuristic and Pentobi checks. Keep validation and held-out evidence distinct.
   Compare cold against random initial and strong fixed baseline; warm against
   its initial policy. Empty-board deterministic games are exactly two, no CI.

   ```sh
   .venv/bin/python -m blokus_rl.duo_az_evaluate --run runs/duo-az-cold-642 --out runs/duo-az-cold-heldout-2549000
   .venv/bin/python -m blokus_rl.duo_az_evaluate --run runs/duo-az-warm-643 --out runs/duo-az-warm-heldout-2549000
   ```

   The evaluator refuses unfinished runs, freezes copies/hashes, and tests raw
   against initial/reference/largest/frontier/largest-specialist, search128 against reference, and
   raw/search128 against Pentobi on100openingpairs each. It also records exactly
   two deterministic empty-board games. End-to-end evaluator smoke passed.
   Selected, final, initial and reference checkpoints (including the largest
   specialist) are copied and hashed before any cases run. Updated smoke passed.
3. Record actual learning curves and total simulated positions, not only game
   counts. Make a compact plot/report. Preserve negative results.
   Implemented: `.venv/bin/python -m blokus_rl.duo_search_report` generates
   `docs/search-results.md`, `docs/assets/search-results.{png,svg}` and
   `runs/duo-search-summary.json`. Regenerate after held-out results complete.
4. If raw/search performance improves, next hypothesis may increase games/search
   or try a spatial network on the same loop. A poor 4096-game run doesn't reject
   AlphaZero. Decide next budget from observed failures and throughput.
5. Commit validated code/docs locally, never push. Existing baseline896a7e0 has no
   remote. Runs are ignored; maintain recovery archive/manifest, disk budget.

Notify Dennis only for meaningful results or a failure needing action. No claim
of optimal, superhuman or human-level play is supported. Pentobi is evaluation
only; do not train on its moves without an explicit change of experiment.
