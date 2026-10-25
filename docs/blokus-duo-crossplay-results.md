# Neural cross-play probe

All models were frozen before this probe. The deterministic empty-board matrix has exactly two games per pair, one per seat. Entries are row-policy score rates (win + half draw), not confidence estimates.

| Row policy | frontier-342 | frontier-343 | frontier-344 | longer-242 | longer-243 | longer-244 |
|---|---|---|---|---|---|---|
| frontier-342 | — | 0% | 50% | 50% | 50% | 25% |
| frontier-343 | 100% | — | 0% | 50% | 50% | 25% |
| frontier-344 | 50% | 100% | — | 50% | 0% | 50% |
| longer-242 | 50% | 50% | 50% | — | 100% | 50% |
| longer-243 | 50% | 50% | 100% | 0% | — | 100% |
| longer-244 | 75% | 75% | 50% | 50% | 0% | — |

The full matrix is descriptive for these specific greedy policies and the normal start. It does not measure robustness over arbitrary openings.

## Shared randomized openings

Each pair below uses 100 common legal four-ply random openings, replayed in both seats. Intervals bootstrap whole opening pairs to preserve dependence between the seat-swapped games.

| Match | W/D/L | Score rate | 95% paired score interval |
|---|---|---|---|
| frontier-342 vs frontier-343 | 119/9/72 | 61.75% | 55.25%–68.00% |
| frontier-342 vs frontier-344 | 86/8/106 | 45.00% | 37.75%–52.75% |
| frontier-343 vs frontier-344 | 89/10/101 | 47.00% | 40.50%–53.50% |

Frontier-344 had the lowest frontier-heuristic benchmark but was competitive in these neural matches. The evidence does not establish a universal ranking. The known weakness motivates a separately declared frozen-opponent curriculum; it does not retroactively change the heuristic results.

[Probe protocol](blokus-duo-crossplay-protocol.md) · [Next experiment protocol](blokus-duo-frozen-league-protocol.md)

Artifacts: `runs/duo-crossplay-1790000/protocol.json` stores opening moves and model hashes; `results.json` includes replayable loss examples.
