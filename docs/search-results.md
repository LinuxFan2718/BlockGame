# Blokus search and self-play results

Status: **complete**.

Score = win + half a draw. Each reported randomized-opening case uses 100 independently generated four-ply openings, each played in both seats. The 95% intervals resample entire two-game opening pairs. Single training seeds do not establish reproducibility across seeds.

![Results](assets/search-results.png)

## Planning with unchanged weights

| Search budget | Games | W/D/L | Score | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| 128 simulations | 200 | 116/9/75 | 60.25% | 56.00%–64.50% |
| 512 simulations | 200 | 136/14/50 | 71.50% | 66.75%–76.50% |

Opponent is the same frozen frontier-342 network playing raw. Seed2542000.

## Outside opponent: Pentobi 31.0 level 1

| Policy | Games | W/D/L | Score | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| raw | 200 | 10/6/184 | 6.50% | 3.75%–9.75% |
| search 128 | 200 | 15/6/179 | 9.00% | 5.50%–12.75% |
| search 512 | 200 | 24/4/172 | 13.00% | 8.50%–17.75% |

All rows share seed2551000 openings and Pentobi seeds, use one Pentobi thread, no book, no resignation, and default avoidance of symmetric draws. Each terminal score margin is independently checked against Rust. These settings do not imply a human rating.

## Self-play budgets and learning

| Run | Finished games | Search simulations | Neural positions evaluated | Wall minutes* |
| --- | ---: | ---: | ---: | ---: |
| cold | 4,096 | 17,495,936 | 14,739,718 | 31.76 |
| warm | 4,096 | 18,094,848 | 14,949,921 | 31.56 |

*Per-run sum of round wall times; the two runs run concurrently. Both use 128 simulations/move, 3 CPU workers, 32 rounds × 128 games, 100 gradient steps/round and 20,000 replay positions. Cold starts randomly; warm starts from frontier-342. Neither uses heuristic or Pentobi training labels. Validation curves have **different reference opponents**: each run plays its own frozen initial network.

## Reserved held-out self-play evaluation

Seed2549000 is never used for checkpoint selection. The selected checkpoint is frozen after training. Raw and search128 are separate agents.

| Run / opponent | Games | W/D/L | Score | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| cold: raw_vs_initial | 200 | 179/4/17 | 90.50% | 86.25%–94.50% |
| cold: raw_vs_reference | 200 | 78/7/115 | 40.75% | 33.75%–47.75% |
| cold: search_vs_reference | 200 | 98/4/98 | 50.00% | 44.00%–55.75% |
| cold: raw_vs_largest | 200 | 118/2/80 | 59.50% | 53.50%–65.50% |
| cold: raw_vs_frontier | 200 | 63/9/128 | 33.75% | 28.00%–39.75% |
| cold: raw_vs_largest_specialist | 200 | 72/8/120 | 38.00% | 31.50%–44.50% |
| cold: raw_vs_pentobi | 200 | 7/1/192 | 3.75% | 1.50%–6.50% |
| cold: search_vs_pentobi | 200 | 9/1/190 | 4.75% | 2.00%–7.75% |
| warm: raw_vs_initial | 200 | 92/12/96 | 49.00% | 42.50%–55.25% |
| warm: raw_vs_reference | 200 | 92/12/96 | 49.00% | 42.50%–55.25% |
| warm: search_vs_reference | 200 | 137/11/52 | 71.25% | 65.25%–77.00% |
| warm: raw_vs_largest | 200 | 137/4/59 | 69.50% | 63.50%–75.25% |
| warm: raw_vs_frontier | 200 | 95/10/95 | 50.00% | 44.00%–55.50% |
| warm: raw_vs_largest_specialist | 200 | 120/6/74 | 61.50% | 54.25%–68.25% |
| warm: raw_vs_pentobi | 200 | 11/6/183 | 7.00% | 4.00%–10.25% |
| warm: search_vs_pentobi | 200 | 18/1/181 | 9.25% | 5.75%–13.00% |

### Equal search budget: does training help the searched agent?

| Run vs original searched reference | Games | W/D/L | Score | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| cold search128 vs reference search128 | 200 | 74/11/115 | 39.75% | 33.25%–46.50% |
| warm search128 vs reference search128 | 200 | 115/15/70 | 61.25% | 55.00%–67.25% |

Both players get 128 simulations/move. This is an additional diagnostic on the original held-out openings, with already-frozen checkpoints; no model was selected or changed from these results. Other reported intervals are nominal 95%, with no correction for the many exploratory comparisons.

The in-repo reference is the original frontier-342 network. Largest specialist means longer-margin-243; largest/frontier alone mean hand-written baselines. Normal-start deterministic games are saved separately as exactly two games without an uncertainty estimate. See each held-out protocol for hashes and frozen copies. Warm raw_vs_initial and raw_vs_reference use the same opponent and openings; their duplicate rows are not independent evidence.

### Matched Pentobi comparison on reserved openings

| Policy | Games | W/D/L | Score | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| original frontier342 raw | 200 | 3/4/193 | 2.50% | 0.75%–4.50% |
| cold raw | 200 | 7/1/192 | 3.75% | 1.50%–6.50% |
| warm raw | 200 | 11/6/183 | 7.00% | 4.00%–10.25% |
| original frontier342 search128 | 200 | 10/7/183 | 6.75% | 3.75%–10.00% |
| cold search128 | 200 | 9/1/190 | 4.75% | 2.00%–7.75% |
| warm search128 | 200 | 18/1/181 | 9.25% | 5.75%–13.00% |

More search can improve decisions at fixed weights; that does not establish that training produced a better raw network. Stronger internal results do not establish human-level play. Degenerate bootstrap intervals do not prove an exact population score.

Method, commands and preservation details: [rolling experiment](search-experiment.md). Run `python -m blokus_rl.duo_search_report` to regenerate this report.

## Follow-up: select checkpoints using searched validation play

This adaptive follow-up compared rounds 4/8/12/16/20/24/28/32 of the existing warm-start run on 40 validation opening pairs, both agents using 128 simulations. It added **zero training games**. Selection used seed2543000; the test used a fresh seed 2559000, generated only after freezing the selected checkpoint.

Selected round: **24**; validation score **58.13%**. The declared primary success criterion required a fresh-test searched-vs-searched 95% lower bound above 50%.

| Fresh test | Games | W/D/L | Score | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| search_vs_search | 200 | 91/16/93 | 49.50% | 43.75%–55.25% |
| raw_vs_raw | 200 | 98/12/90 | 52.00% | 45.75%–58.01% |

**Failed** that criterion: selecting on searched validation did not produce a confirmed improvement on the new test. Do not promote this round 24 checkpoint over the original selected round 8 model from these results. This is not a replication of round 8 on the new seed, because a different checkpoint was tested.
