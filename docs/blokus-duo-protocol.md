# First Blokus Duo learning experiment

Written before the first training run. Tic-tac-toe cold-start learning succeeded
with the search-guided method in two of three seeds; it did not establish that
PPO self-play alone reliably bootstraps. Blokus is a new experiment.

## Rules and architecture

The reference is [Mattel's official FWG43 instructions](https://service.mattel.com/instruction_sheets/FWG43-Eng.pdf),
including its starting-point diagram. The board has 14×14 squares; each player
has the same 21 pieces. The first placement covers either marked start, and the
second player uses the other. Later placements touch one's own pieces at corners
and never edges. Other-color contact is unrestricted. A player with no placement
must pass. Both blocked ends play. Scores subtract unplayed squares; using every
piece scores +15, or +20 when the monomino was last. Equal scores draw.

Rust is the single rules implementation. An independent cell checker tests its
cached-mask move generator. PyO3 exposes games and legal candidates. PettingZoo
exposes a fixed integer action space (orientation × top-left square, plus pass).
Training uses the compact legal candidate list for each position.

The neural network is an ordinary PyTorch feedforward actor/critic. Input:
relative board planes, both remaining inventories, required starts and seat.
It predicts contextual square and piece scores, adding them to score each legal
placement. This is an architectural assumption about how to represent a move,
not a supplied strategy. No demonstration moves or heuristic training targets.

## Hypothesis and selection

First hypothesis: terminal-outcome PPO against uniform random legal play can
learn a better policy than its initial weights. Unlike tic-tac-toe, a sampled
finite evaluation cannot certify perfect play in Blokus.

Initial smoke run: small budget to verify the pipeline. Main pilot: seed 42,
100 updates × 16 complete learner games = 1,600 games. Both learner seats
alternate during collection. Three PPO epochs, Adam 0.0003, entropy coefficient
0.02, gamma 1; complete-episode Monte Carlo returns. TorchRL supplies clipped
PPO. Policy is frozen during each collection, and saved old log probabilities
are checked against it before optimization in tests.

Validation every 10 updates: 20 games per seat, seeds beginning 70000, against
uniform random and a largest-area legal-piece baseline (ties random). Select
the highest validation score rate against random, then largest. Keep initial,
selected, final and latest checkpoints separately. Test data never select a model.

Held-out test: 100 games per seat, separate seeds beginning 90000. Compare
selected and initial against both baselines, and selected directly against initial.
Report W/D/L, both seats, score margins and strict-win Wilson intervals.
The evaluated neural learner is greedy; frozen neural opponents sample their
saved distribution. Otherwise two deterministic policies would repeat the same
two games, making a binomial confidence interval misleading. This correction
was made during the pilot; the direct initial-model match is rerun accordingly.

Evidence of improvement requires beating the untrained policy directly with a
95% strict-win lower bound above 50%, alongside improvement against random.
Beating random is an initial milestone; largest-piece performance is a stronger
baseline and still does not establish human-level skill. Replicate promising
methods in other seeds. Separate validation tuning from fresh final test seeds
if later experiments reuse the initial test results to inform a new method.

## Follow-up hypotheses

If terminal wins give poor credit assignment, compare terminal score margin/109
as an explicitly different objective. If random-opponent training improves, try
a frozen-snapshot league with 40% random opponents and 60% uniformly chosen
historical models, fixed for each episode. A snapshot is added every 10 updates,
up to 16. This differs from the unsuccessful tic-tac-toe mixture and must be
evaluated on its own merits. Neither change is evidence until tested.

Warm starts use weights only with a new optimizer and seeded RNG. They are not
exact resumption, and parent checkpoint hashes must be recorded.
