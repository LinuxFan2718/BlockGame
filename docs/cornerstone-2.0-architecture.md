# Cornerstone 2.0 — architecture

Cornerstone is this project's Blokus Duo agent. In Blokus your pieces may
touch only at their corners, and the agent's play revolves around claiming
corners. Releases:

- **Cornerstone 1.0** (`7a8ed0c`, 28 October 2026): a 53k-parameter CNN with
  Rust tree search. It beat Pentobi level 3 and lost to level 4.
- **Cornerstone 2.0** (this document, 29 October 2026): built in eight phases
  recorded in [the v2 roadmap](v2-roadmap.md).

## Family tree

Cornerstone 2.0 is a **custom network in the AlphaGo Zero / AlphaZero family**,
not a copy of any published network.

| Idea | Source |
|---|---|
| One residual network with a move-policy output and a position-value output; tree search guided by it; trained on the search's visit counts and on game results | Silver et al. 2017 (AlphaGo Zero, AlphaZero) |
| Start by imitating an expert, then improve by self-play | Silver et al. 2016 (the original AlphaGo) |
| Global pooling in every block; win/draw/loss value; auxiliary score-margin and ownership targets; mixed fast/full searches; first-play-urgency reduction; averaged (EMA) weights | KataGo (Wu 2019) |
| Move output of 91 piece orientations × board squares | AlphaBlokus |
| Inputs computed from the rules: attachment corners, forbidden cells | Concepts behind Pentobi's move features |
| Exact endgame solver | Chess-engine practice (alpha-beta, transposition tables) |

The main differences from Silver's network are size and the KataGo-style
additions. AlphaGo Zero used 20–40 residual blocks of 256 channels;
Cornerstone 2.0 uses **6 blocks of 64 channels (571,006 parameters)**. That was
the most cost-effective size for this project's Modal budget.

## 1. What the network sees

Every position is shown from the point of view of the player to move. Player 1
sees the board rotated 180°, so both players' openings look alike.

**Nine 14×14 input planes**, all computed by the Rust rules engine:

| # | Plane |
|---|---|
| 0 | Cells I occupy |
| 1 | Cells my opponent occupies |
| 2 | My attach cells: empty cells where a new piece of mine could touch my pieces corner-to-corner (my start square before my first move) |
| 3 | The opponent's attach cells |
| 4 | Cells I may never fill (they touch my own pieces along an edge) |
| 5 | Cells my opponent may never fill |
| 6 | Cells covered by at least one of my legal moves |
| 7 | Cells covered by at least one of the opponent's legal moves |
| 8 | All ones (lets the network see the board edge through zero padding) |

**48 numbers about the position:**
- which of my 21 pieces remain, and which of the opponent's 21 remain
- both current scores
- the number of consecutive passes
- whether I moved first
- the log of each side's number of legal moves

None of these are strategy heuristics. They are facts the rules make cheap to
compute but a small network would otherwise have to work out itself.

## 2. The network

```
scalars (48) ─ Linear → 64, ReLU ─────────────────────────────┐ context
                                                               │
planes (9×14×14) ─ 3×3 conv → 64 ─ + Linear(context) bias ─ BN ─ ReLU
                    │
                    ├─ 6 × residual block ─────────────────────┤
                    │      3×3 conv → BN → ReLU
                    │      + global bias: mean & max over the board, plus
                    │        context → Linear → one bias per channel
                    │      3×3 conv → BN (starts at zero) → + skip → ReLU
                    │
                    ├─ policy head: 1×1 conv → 32, ReLU, global bias
                    │     1×1 conv → 91 logits per square (orientation × anchor)
                    │     + per-orientation bias and a pass logit (from pooled features)
                    │     = 17,837 move logits; illegal moves are masked out
                    ├─ value head: 1×1 conv → 32, ReLU, mean & max pooling + context
                    │     → MLP (128) → win/draw/loss logits + final score margin (÷20)
                    └─ ownership head: 1×1 conv → 3 classes per cell
                          (empty / mine / theirs at the end of the game), training only
```

A move is identified by its oriented shape (91 of them, counting the 21 pieces'
rotations and reflections) and its top-left anchor square, so one convolution
scores every placement at once. The engine supplies the legal list, and the
softmax runs over legal moves only.

**Global pooling** gives every block a summary of the whole board and of both
inventories, so local features can depend on distant regions and on which
pieces remain. This is the main fix for Cornerstone 1.0's 9×9 field of view.

## 3. Search

PUCT Monte Carlo tree search in Rust:

- Node priority = Q + 1.5 · P · √N_parent / (1 + N_child).
- **First-play urgency:** an unexplored move is valued at the parent's value
  minus 0.25 × √(prior mass already visited) (0 at the root), rather than as a
  draw.
- **Value used by the search** = P(win) − P(loss) + 0.05 · tanh(predicted
  margin / 20).
- Several leaves per tree are evaluated together using virtual loss; hundreds
  of games are batched for the GPU.
- The subtree under the chosen move is kept for the next move.
- **Exact endgame solver:** alpha-beta over the true final score margin, with
  a Zobrist transposition table and largest-piece-first move ordering. When
  both sides together have few legal placements left (20 in self-play, 30 in
  play), the solver picks the move outright, so endgames are played perfectly.
  On its own it scored 61% against the same search without it.
- In play: 800–2,048 simulations per move, no noise, best-visited move.

## 4. How it was trained

1. **Imitation (supervised).** 32,358 Pentobi-vs-Pentobi games: 21,769 at
   level 6 (generated free on the Mac) and 10,589 at levels 7–8 (on a Modal CPU
   container). The network learned Pentobi's move (one-hot), each game's result
   and margin, and the final owner of every cell. It used random board
   symmetries, 15,000 steps of 1,024 positions, AdamW with a one-cycle learning
   rate up to 2e-3, and EMA weights. It matches Pentobi's move on about 38% of
   held-out positions.
2. **Self-play reinforcement learning** on one L40S with 32 CPU cores at about
   22 games/s:
   - **Search budget:** 25% of moves get a full 800-simulation search with
     exploration noise; only these are recorded for training. The rest get 128
     simulations.
   - **Exploration:** the first 10 plies are sampled from the visit counts.
   - **Openings:** games start after 0–4 random plies. This fixed a collapse
     caused by Duo's large first-move advantage.
   - **Losses:**
     - policy: cross-entropy against visit counts
     - value: win/draw/loss cross-entropy
     - margin: smooth L1, weight 0.5
     - ownership: cross-entropy, weight 0.5
   - **Batches:** 512 positions under random symmetries, drawn from the newest
     half of a 2-million-position replay buffer, with 10–20% Pentobi positions
     mixed in.
   - **Optimizer:** AdamW, learning rate 5e-4 → 3e-4 → 2e-4 → 1e-4, EMA decay
     0.995.

   The run: 823,673 self-play games, 74,000 gradient steps, about $37 of Modal
   credits for the RL itself.

## 5. Results

The release network (`runs/v2-release/v2.pt`, 823,673 self-play games) at
800 simulations per move. Each level uses 100 fresh openings of 4 random plies,
each played from both seats (200 games):

| Pentobi 31.0 level | Cornerstone 1.0 | Cornerstone 2.0 |
|---|---|---|
| 4 | 34–44% at 2,048 sims (did not clear) | 89% (imitation network) |
| 5 | not reached | 80.5% (imitation network) |
| 6 | — | **76.0%** [71.25, 80.75] |
| 7 | — | **58.25%** [54.0, 62.5] |
| 8 | — | **52.25%** [49.0, 55.5] |
| 9 | — | 45% in a 20-game sample (earlier RL network) |

Head-to-head, the release network beats the imitation network 65.25%
[61.0, 69.75], about +110 Elo from self-play.

In Duo the first player wins most games at high levels: Pentobi beats itself
95% of the time as first player at level 8. Against level 8, Cornerstone wins
81.5% of its first-seat games but only 23% from the second seat. Winning from
the second seat is its remaining weakness.

## 6. Code map

| Part | File |
|---|---|
| Rules, move generation | `src/duo.rs` |
| Input planes, symmetries | `src/duo_v2.rs` |
| Search tree | `src/duo_mcts.rs` |
| Batched self-play engine | `src/duo_engine.rs` |
| Endgame solver | `src/duo_solver.rs` |
| Compact replay buffer | `src/duo_replay.rs` |
| Network | `python/blokus_rl/v2/model.py` |
| Inference and engine driver | `python/blokus_rl/v2/engine.py` |
| Self-play training | `python/blokus_rl/v2/train.py` |
| Pentobi imitation | `python/blokus_rl/v2/pentobi_data.py`, `imitate.py` |
| Pentobi ladder, head-to-head | `python/blokus_rl/v2/ladder.py`, `arena.py` |
| Modal jobs and credit guard | `cloud/modal_v2.py`, `cloud/v2_budget.py` |
