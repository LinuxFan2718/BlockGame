# Why our current Blokus agent trails Pentobi

26 October 2026. This note separates measured results, inspected implementation
details and hypotheses. The levels 1–9 benchmark is still running; its partial
results are not used here.

## Have we played AlphaBlokus?

**No. None of our agents has played AlphaBlokus.** We inspected its source,
configuration and published ONNX network graph. We did not execute its agent or
download the separate network-weight data file.

The inspected AlphaBlokus targets four-player Classic on a 20×20 board. Our
trained agents play two-player Duo on a 14×14 board, with different starting
positions and scoring. Its four-player policy/value outputs cannot be plugged
into our Duo evaluator. A meaningful match needs a shared game variant and a
compatible agent, rather than simply connecting the two programs. The repository's
`DuoArena` name in a Classic benchmark denotes two software opponents; it does
not establish support for the 14×14 Duo game.

The checked AlphaBlokus playing configuration uses 2,000 search simulations per
move. That is consistent with a substantial thinking budget, but I have not
measured the browser's speed or independently reproduced its playing strength.
The author reports results against strong Pentobi; those are the author's
results, not matches we ran. See our [architecture study](alphablokus-study.md)
and the [author's repository](https://github.com/ShivamSarodia/AlphaBlokus).

## What we have actually established

The best model from our completed study is the **small CNN**, with 52,978
parameters. It was trained from random initialization on 32,768 self-play games.
Its selected checkpoint is round 128. Pentobi supplied no training moves or
labels. The deep model has 986,205 parameters and the same game budget; its
selected checkpoint is round 120.

Against Pentobi 31.0 level 1, the completed held-out test used 100 opening
positions, each played with seats swapped: 200 games per agent.

| Agent | Wins / draws / losses | Score |
| --- | ---: | ---: |
| Small CNN, direct policy | 27 / 5 / 168 | 14.75% |
| Small CNN, 128 search simulations/move | 58 / 9 / 133 | 31.25% |
| Deep CNN, direct policy | 12 / 1 / 187 | 6.25% |
| Deep CNN, 128 search simulations/move | 21 / 2 / 177 | 11.00% |

Score counts a draw as half a win. The small searched agent's nominal paired
95% interval is 25.25%–37.50%. This supports a substantial gap even against
level 1. These are the completed study's results, not the fresh levels 1–9
benchmark. Full methods and limitations are in [deep-results.md](deep-results.md).

## My leading explanations

### Pentobi retains useful game knowledge even at level 1

Pentobi's inspected source includes explicit features for piece size, available
corner connections, surrounding space and interaction with the opponent's
connection points. Its Duo-specific feature weights are marked as learned using
its learning tool. Calling it simply a collection of hand-written rules misses
that fitted knowledge.

Lowering its level reduces search; it does not remove that knowledge. Its Duo
level-1 base simulation count is only three. Thus our loss at level 1 cannot be
explained by assuming Pentobi simply searches more nodes than our 128.

Primary sources, pinned to the version we ran:
[Duo priors and learned weights](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/PriorKnowledge.cpp),
[feature definitions](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/PriorKnowledge.h),
[level budgets](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/Player.cpp).

### The two engines' simulations mean different things

Pentobi's search uses guided simulated continuations, normally through the rest
of a game, with early termination in some cases. Our search normally stops at a
new position and asks the neural network to estimate the eventual outcome.
Both approaches are legitimate; their simulation counts are different units.

This makes our learned **policy** (which moves deserve attention) and **value**
(how promising a position is) especially important. An overconfident value can
favor a losing branch; an inaccurate policy can leave a good move underexplored
within 128 simulations. More search helps in the measured results, but does not
eliminate dependence on the network.

Pentobi implementation:
[playout loop](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libboardgame_mcts/SearchBase.h),
[playout rules](https://github.com/enz/pentobi/blob/d5374ddc20eced902788e733d67ee16bd1836fd3/libpentobi_mcts/State.h).
Our implementation: [neural search driver](../python/blokus_rl/duo_mcts.py),
[Rust traversal and backup](../src/duo_search.rs).

### Our useful training dataset is still modest

The small model's run produced 225,539 recorded full-search positions and about
692,000 sampled training examples over 2,702 optimizer steps. It evaluated about
75 million positions inside search, but those are **not 75 million supervised
training examples**. Only selected search roots entered the training buffer.

That is a meaningful learning experiment, but it leaves substantial room to
test additional experience, better search targets and optimization settings.
We have not measured the learning curve far enough to know whether more games
alone would close the gap. Nor have we established that the present training
ratio is suitable for the larger network.

### Self-play can preserve shared blind spots

Both sides initially make poor decisions. Search needs to produce a better
move distribution than the current policy; the next network learns from that
improvement. If the current policy and value miss a strategic resource, modest
search may fail to discover its importance, so both players continue overlooking
it. This is a plausible weakness of our current setup, not a demonstrated
diagnosis of a particular lost game.

Blokus makes delayed consequences especially relevant: spending a piece or
losing access to a region can matter many moves later. Our value target is the
final win/draw/loss. The network must infer those connections from experience.
An expert opponent could expose mistakes, but learning from one is optional;
it is not a prerequisite for self-play to work.

### More network capacity did not solve this particular experiment

The deep searched agent scored only 25.50% head-to-head against the small
searched agent. That establishes which trained agent was stronger here. It
does not establish why: depth, width, normalization, coordinate inputs,
inventory conditioning and value-output design changed together, and we ran
only one training seed per setup.

My working hypothesis is that the larger model needs a better-matched data and
optimization regime, or a better implementation/design choice. That hypothesis
could be wrong. We should not treat this as evidence that deep CNNs cannot play
Blokus, or assume a larger GPU would fix it.

## What is reassuring, and what remains open

The small raw policy scores 99.5% against its untrained initial network and
80% against our largest-piece heuristic. Its searched version also beats the
previous warm-start searched agent with a 72.75% score. Learning and search are
producing useful improvements.

The independent rules checker and Pentobi game/score cross-checks reduce concern
about a basic rules mismatch. They do not prove that the learning algorithm,
value calibration or search choices are optimal or free of subtle problems.
We have not yet isolated whether wrong move rankings, wrong value estimates,
insufficient exploration or training settings contribute most to the gap.

## What I would investigate next

1. Finish the frozen levels 1–9 evaluation to establish the performance curve.
2. Inspect representative losses and compare the network's values and preferred
   moves with stronger search. Test whether additional search actually changes
   the decisions that lose games, rather than assuming it will.
3. Replicate the small setup across independent seeds, then change one factor
   at a time: training examples per new position or self-play search budget.
4. Use Pentobi imitation as a separate diagnostic if useful: can the network
   represent and learn stronger decisions when given them? Keep a pure self-play
   control so that we can distinguish teacher-assisted progress from learning
   from scratch.

These are proposed investigations, not newly launched training runs. My leading
explanation is a gap in learned game knowledge and the quality of the search
targets used to acquire it. The current evidence does not show a fundamental
limit of reinforcement learning or establish that compute alone is the answer.
