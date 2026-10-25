# Follow-up: can harder opponents teach more than random play?

Declared after the first seed-42 pilot showed 93% wins against random but only
49.5% against largest-piece play on 200 held-out games. Those pilot test results
are exploratory evidence for this follow-up; use fresh final test seeds here.

All arms warm-start the SAME seed-42 pilot selected checkpoint, resetting Adam
and the RNG. Each gets seed 142, 200 updates × 32 episodes = 6,400 additional
games. Hidden size 128, three PPO epochs, terminal win reward, otherwise the same
PPO defaults. Training opponents:

- random: keep training against uniform legal play (compute/game control).
- league: 40% random and 60% sampled immutable snapshots, fixed per game.
- mixed: 25% random, 25% largest-piece, 50% sampled immutable snapshots.

Snapshots include the initial warm-start model, add one every 10 updates, retain
16. Largest-piece opponents give outcomes, not demonstration targets; this arm
uses a human-designed opponent curriculum, so do not call it pure self-play.

Validation every 20 updates uses 40 games per seat starting at seed 170000.
Select highest mean of the random and largest-piece score rates, breaking ties
with largest-piece performance. Unlike the pilot, this explicitly values both
benchmarks equally. Test selected and initial on 200 games per seat at seed
190000, and evaluate selected greedy against the sampled initial policy.
Save final independently of selected.

An encouraging result is a strict-win 95% lower bound above 50% against largest
while retaining at least 90% score rate against random. Prefer gains in BOTH
seats. Even a pass means beating one heuristic; it does not establish strong
human-level play. Replicate promising methods with other training seeds before
claiming reliability. Record failures as well as gains.

## Extension declared before its runs

While the three original arms were running, add a direct largest-piece opponent
arm with the same parent, seed and 6,400-game budget. It tests whether a known
weakness benefits from concentrated practice. Also run the same arm with
`reward=margin`, an explicitly different objective (terminal score difference
divided by 109). Its more informative outcomes might reduce the noisy
win-only credit assignment. These are follow-up hypotheses, not part of the
original three-arm declaration. Use test seed 290000 for these additional arms.

The league's first held-out result was 63.5% strict wins against largest-piece
play (400 games, 95% interval 58.7–68.1%). Replicate the league using training
seeds 143 and 144, starting respectively from the independent seed-43 and
seed-44 pilots. Budgets and validation criterion stay the same; final test seed
390000. Thus the two-stage pipelines use independently trained parents, rather
than merely fine-tuning one lucky parent with different random seeds.

The largest-opponent margin arm then won 69% against largest on 400 fresh games,
versus 58.25% for its win-reward counterpart. Replicate the margin arm with
seeds 143/144 and independent pilot parents 43/44, unchanged budgets and selection,
using final test seed 490000. This tests replication, not a statistically definitive
claim that the two reward objectives differ from a single pair of runs.
