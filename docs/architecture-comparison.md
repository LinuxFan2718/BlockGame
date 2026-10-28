# Our agent, Pentobi and AlphaBlokus — 28 October 2026

Our current evidence points first to unreliable transfer and limited training
coverage. A representational ceiling is plausible, but unproven. The small CNN
is much smaller than AlphaBlokus; however, our earlier, similarly sized residual
network lost clearly to it at the training budget we supplied. Increasing depth
or search alone is therefore not an evidence-backed cure.

This is a source and completed-results comparison. It starts no games, GPU jobs
or training. Local parameter counts were recomputed from our model classes;
AlphaBlokus's count below is calculated from its published layer dimensions,
not from downloaded model weights.

## What is actually being compared

Our production architecture is `DuoConvActorCritic`, used by the preserved
round-112 parent and the seed-1070/1071 descendants. Our alternative
`DuoResidualActorCritic` was tested in the earlier cold-start experiment.
Pentobi is the pinned 31.0 source commit
`d5374ddc20eced902788e733d67ee16bd1836fd3`. AlphaBlokus is the inspected source
commit `b4470ef9667e7c6f000a0542bb3aaa4d5cce1292`, also checked previously against
its published ONNX graph metadata.

| Component | Our current CNN | AlphaBlokus | Pentobi 31.0 |
|---|---|---|---|
| Game considered here | Two-player Duo, 14×14 | Four-player Classic, 20×20 | Supports both; our benchmark uses Duo |
| Main evaluator | Learned policy/value CNN | Learned policy/value residual CNN | Feature-based move priors and heuristic stochastic playouts; no policy/value CNN |
| Network size | **52,978 parameters**; four 3×3 convolutions, width 32 | **909,983 parameters** from published configuration; stem plus ten two-convolution residual blocks, width 64 | Neural parameter count does not apply |
| Spatial inputs | Own/opponent occupancy and active starting-square plane | Four player occupancy planes, plus x/y coordinates | Exact game state with explicit geometric move features |
| Remaining pieces | Both inventories; one late feature modulation and piece bias | All four inventories; modulation in five alternating residual blocks and both heads | Piece availability and piece/attachment features used directly |
| Policy output | 91 oriented-shape channels × 14×14 anchors, plus pass | 91 oriented-shape channels × 20×20 anchors | Feature-weighted move probabilities |
| Value | One scalar expected terminal win/draw/loss result | Four outcome shares, necessary for its four-player game | Results of playouts, with quality bonuses including score and game length |
| Search | PUCT-style neural leaf search, eight pending leaves with virtual loss | PUCT-style neural leaf search; concurrency across many games, without virtual loss | MCTS with RAVE, Last-Good-Reply and game-specific priors/playout knowledge |

Both neural agents mask illegal actions, train policy against search visits, and
train value against eventual outcomes. Our legal-only policy loss already
implements the option AlphaBlokus describes as helpful. A scalar value is valid
for two-player zero-sum Duo: copying its four-player value vector would not
address a missing requirement in our game.

Sources: our [small CNN](../python/blokus_rl/duo_conv.py),
[residual alternative](../python/blokus_rl/duo_residual.py),
[state encoding](../python/blokus_rl/duo_env.py),
[loss](../python/blokus_rl/duo_deep.py), and
[Rust search](../src/duo_search.rs);
AlphaBlokus [network](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/python/src/alphablokus/res_net_film.py),
[configuration](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/configs/train_live/vast_film.toml)
and [method description](https://github.com/ShivamSarodia/AlphaBlokus#adaptations);
Pentobi [move priors](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/PriorKnowledge.h),
[search settings](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/SearchParamConst.h)
and [playout evaluation](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/State.cpp).

## The architectural concern is spatial reasoning

Four stride-one 3×3 convolutions give our local feature path a **9×9 receptive
field** on a 14×14 board. A pooled summary of all squares also reaches every
policy logit through the context/FiLM branch. Thus the network has whole-board
information, but compresses distant spatial arrangements before combining them.
Its value head uses that pooled context directly. Inventories enter after the
four spatial convolutions; there is no later stack of spatial nonlinear layers
that can repeatedly combine inventory with board geometry.

AlphaBlokus has enough convolutional depth for features to depend on its whole
20×20 board. Coordinates are supplied at the stem, selected deeper blocks and
heads, and inventories alter intermediate features repeatedly. This is a more
expressive way to represent access routes, blocking, distant corners and which
remaining shapes can exploit them. The claim that these particular concepts
limit our agent is a hypothesis; we have not measured a causal failure in them.
AlphaBlokus also pools before its value output, but does so after a much deeper,
inventory-conditioned spatial computation.

Our previous residual model already had ten blocks, width 64, coordinates and
repeated inventory modulation: **986,205 parameters**, slightly more than this
AlphaBlokus configuration. With 32,768 cold-start games and about 224,000 recorded
roots per arm, it scored only **25.5% [19.75%, 31.5%]** against our small CNN at
equal 128-simulation playing budgets. It is not an exact AlphaBlokus copy: it
uses GroupNorm rather than BatchNorm, a different head and conditioning design,
and Duo rather than Classic. Within our deep-versus-small experiment, both arms
played Duo; depth, width, normalization, coordinates, conditioning and value
activation changed together, so the experiment cannot isolate depth.
It establishes that the larger bundle did not help under that budget, not that
large networks cannot help. [Completed deep study](deep-results.md).

## Training differs as much as the network

| Setting | Our recent control | Our recent teacher | AlphaBlokus published configuration |
|---|---:|---:|---:|
| Fast moves, 80% of decisions | 32 simulations | 32 simulations | 200 rollouts |
| Recorded full-search moves, 20% | 1,024 | 2,048 | 1,000 |
| Expected search work per decision from that mixture | 230.4 simulations | 435.2 simulations | 360 rollouts |
| Replay window | 100,000 positions | 100,000 | 2,700,000 |
| Training examples per new recorded root | Approximately 3 | Approximately 3 | Configured 3 |
| Initial stochastic move-selection period | 12 plies | 12 plies | 35 turns in a four-player game |

The mixture averages describe requested budgets, not equal-strength or equal-time
searches across different games and implementations. AlphaBlokus's fast moves
receive **6.25×** our search budget. Improving only the full-search teacher leaves
the other 80% unchanged. Those cheap moves determine which positions and terminal
labels the recorded examples eventually encounter. They are an important
untested data-quality hypothesis, not automatically wasted moves.

AlphaBlokus's replay capacity is **27×** ours. The recent seed-1071 run performed
131.4 million simulations and 102.8 million neural position evaluations, but
produced only **114,011 new recorded roots**, with 1,366 optimizer steps and
349,696 gradient examples. Those are distinct units; tree evaluations are not
additional training labels. The parent had substantial earlier training, so the
114,011 figure is an increment, not its lifetime experience.

The published AlphaBlokus filename `026025784` and the exporter's sample-count
naming convention suggest roughly 26 million cumulative recorded samples. This
is an inference from source and filename, not an independently audited training
ledger, and does not establish a matched sample-efficiency comparison. Its
larger replay window itself is directly specified in the configuration.

Our collection starts at the empty board, uses root noise on full searches and
stochastic early moves, and does not apply random symmetry augmentation in the
current fitting path. The observer's fixed 180-degree normalization is not data
augmentation. Wider replay and rules-preserving symmetry augmentation could
increase coverage, but must be tested; old/off-policy replay can also hurt.
The different early-temperature settings cannot be compared as if Duo and
four-player Classic had the same turn scale.

Sources: [our collection](../python/blokus_rl/duo_shared.py),
[replication protocol](control-replication-protocol.json),
AlphaBlokus [self-play configuration](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/configs/self_play/full_mcts_vast.toml),
[training configuration](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/configs/train_live/vast_film.toml)
and [sample-count exporter](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/python/scripts/train_live.py).

## What Pentobi supplies that we must learn

Pentobi's compact priors use explicit features for piece size, attachment
opportunities, opponent attachment points, forbidden squares and recent local
interaction. Some prior weights were fitted from games with its softmax learning
tool; other playout parameters were tuned. Its Duo playout weighting explicitly
favors larger pieces and shapes with more attachment points. RAVE shares evidence
from moves appearing later in simulations, and Last-Good-Reply reuses successful
responses. Playout rewards include score/length quality bonuses in addition to
the result. This supplies useful geometric and strategic structure before any
policy/value neural training.

Our input is not missing the inventories or terminal-score state. But we ask a
small model to infer useful geometric features from occupancy and eventual WDL,
while Pentobi supplies several of them explicitly. Possible experiments include
attachment/mobility input planes or auxiliary score/mobility prediction, keeping
terminal score as the playing objective. They would be declared changes to our
learning protocol, not proven fixes.

Also, **2,048 neural simulations is not 2,048 Pentobi playouts**. Pentobi Duo level
4 starts from a base count of 213 and weights it by move number; its work per
simulation and termination rules differ. Level 4 is not a matched node-count or
wall-clock comparison. Our benchmark disables its book and resignation.
[Pentobi budget implementation](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/Player.cpp).

## What our results actually diagnose

1. **Transfer is unreliable — observed.** Seed 1070's control improved Pentobi-4
   score by +13.75 percentage points [5.5, 22.25]. Seed 1071's improvement was
   −1.25 [−8.25, 5.75]: control 37.75%, retested parent 39%. Yet that control
   scored 58% [51.5, 64.5] against the parent with equal playing search and only
   49% [42.5, 55.25] raw. Head-to-head progress need not transfer to Pentobi.
   This supports investigating opponent/distribution coverage, not a conclusion
   that all learning has stopped. Different books and selected experience prevent
   attributing the discrepancy solely to training seed.
2. **Selection is noisy and mismatched — observed design, causal effect unknown.**
   We select the highest point score against one frozen parent on 32 opening
   pairs at 512 simulations, then test Pentobi at 2,048. The new run selected
   round 16/4,096 games, although all 64 rounds/16,384 games completed. Its
   validation score was 62.5% [50, 75]; round 64 was 60.94%. This does not prove
   the final checkpoint is better, but does make small validation differences a
   weak guide to transferable strength.
3. **More full-root search has not established a further gain.** In seed 1070,
   teacher-minus-control Pentobi-4 difference was −0.25 points [−7.25, 6.75],
   despite 87.7% more actual training simulations. That interval is inconclusive,
   not proof of equivalence. The earlier 256→1,024 continuation did improve
   strength; search is useful, but its marginal return is not guaranteed.
4. **Small-model spatial limitations and narrow replay are credible hypotheses.**
   Both differ materially from AlphaBlokus. Neither has been isolated as the
   cause of the fresh failed replication.
5. **GPU utilization is a throughput issue.** The completed training window
   averaged 17.34% GPU busy and 3.63/8 CPU cores, with ample RAM/VRAM. The host,
   search and inference coordination path limits throughput. Better utilization
   could buy more data per dollar; it does not itself improve target quality or
   explain a strength plateau. The 64-round limit was reached before the time cap.

Evidence: [replication](control-replication.md),
[controlled teacher comparison](integrated-overnight.md),
[earlier continuation](continuation-results.md),
[completed performance profile](control-replication-performance.md).
All quoted intervals are nominal whole-opening intervals for fixed selected
models, not uncertainty across independent training runs.

## Recommended order

First, use saved artifacts for a diagnostic audit: value error/calibration by
game stage and opponent, policy/search disagreement, root-visit concentration,
opening diversity and validation-selection stability. Predictions against
Pentobi are a distribution-shift diagnostic, not calibration under identical
self-play policies. Treat examination of existing test games as exploratory;
do not choose an alternative checkpoint on those outcomes and call it held out.

Then prespecify a more reliable selection/evaluation protocol with broader
validation opponents and enough opening pairs to resolve meaningful differences.
Keep a fresh final test book and report multiple training seeds separately.

The next training ablation I would prioritize is coverage/trajectory quality:
change fast-move search or replay coverage one at a time, with explicit equal-time
and actual-work accounting. Symmetry augmentation is another low-compute
candidate, after exact Rust checks of state/action/target transforms. Do not
simultaneously change these and architecture.

If diagnostics implicate value or global geometry, test a modest residual model
with repeated inventory conditioning and coordinates, with a learning-budget
ablation. Our existing ten-block failure is a reason to measure optimization and
sample needs, not to jump immediately to an even larger network. Auxiliary
geometric/score targets are a separate experiment. Nothing in this report
authorizes or launches another paid run.

AlphaBlokus is a useful design reference, not a direct strength baseline: it plays
four-player Classic. Its public win/cost claims have not been reproduced here.
No human Elo, frontier ranking or architectural impossibility is inferred.
