# AlphaBlokus architecture study — 25 October 2026

Scope: read the public source and inspect the published ONNX graph at commit
`b4470ef9667e7c6f000a0542bb3aaa4d5cce1292`. No AlphaBlokus training, game execution or independent strength
comparison was performed. Public source is in the ignored
`.local/alphablokus-study/` checkout; no repository scripts were executed. The
separate model-weight data file was not downloaded.

The answer is **yes to both: a residual convolutional neural network and
neural-guided Monte Carlo tree search**, trained using an AlphaZero-style
self-play loop. This implementation targets four-player Classic on a20×20 board.
It should not be confused with our14×14 Blokus Duo environment.

## The actual published network

The README leaves its architecture diagram unfinished, so this study checked
`res_net_film.py`, live configuration and the published browser model's ONNX
metadata. `static/browser/026025784.onnx` is a Git LFS pointer. The actual168,445byte
graph was downloaded separately and verified against the pointer's SHA256:
`43dce8062bcfcb57e8819db29bfb340545be1c368404a3c979085703d62831fa`.
Saved inspection results: [model metadata](assets/alphablokus-model-metadata.json).

| Component | Confirmed representation |
| --- | --- |
| Board input | batch×4×20×20, one occupied-square plane per player |
| Coordinates | two internally generated x/y planes, producing six initial channels |
| Remaining pieces | batch×4×21, supplied separately |
| Initial convolution | 6→64 channels,3×3, batch normalization, ReLU |
| Main spatial body | 10 residual blocks, two3×3 convolutions each,64 channels |
| Policy output | batch×91×20×20 placement logits |
| Value output | batch×4 outcome logits |

Board and inventory inputs are normalized to the current player's perspective.
The board rotates and player channels are reordered, allowing the same network
to serve all four seats. Coordinate channels preserve explicit information about
location, including edges and starting corners.

A residual block adds its input back to a learned transformation before applying
ReLU. This skip connection lets a deeper network preserve features and makes it
easier for training signals to reach earlier layers. Each output channel is an
entire feature map, not one scalar and not one player.

The84 remaining-piece indicators pass through an84→64 dense encoder, followed
by a64→896 projection. Those896 numbers supply channel-wise scales and offsets
for five alternating residual blocks and both heads. This is feature-wise linear
modulation, or **FiLM**. Inventory can therefore change how spatial features are
interpreted: a cavity may be useful with a matching piece left and useless once
that piece has been played. That example is an interpretation of the architecture,
not an independently observed learned feature. Coordinates are also reintroduced
in blocks2/5/8 (zero-indexed) and the two heads.

The policy gives one score for each oriented shape and anchor location. The
91×20×20 tensor has36,400 slots; these are not all legal moves. Rust generates
legal placements, and inference normalizes the corresponding logits. Policy
training masks illegal placements before cross-entropy.

The value head pools spatial features, uses dense layers and outputs four logits.
Inference turns them into four outcome shares. Terminal code assigns1 to a sole
winner, divides1 equally among tied winners and assigns0 to others. For example,
a tie between two players has target[0.5,0.5,0,0]. The inspected result function
ranks occupied-square counts; it is not our Duo bonus-scoring implementation.
At each search node, the player moving there maximizes their own value component.
This avoids treating a four-player game as alternating two-player sign flips.

Sources: [network implementation](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/python/src/alphablokus/res_net_film.py),
[live model configuration](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/configs/train_live/vast_film.toml),
[published graph pointer](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/static/browser/026025784.onnx),
[perspective normalization](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/crates/game-core/src/game/board.rs#L44),
[terminal outcome](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/crates/game-core/src/game/state.rs#L117).

## Search and the learning loop

1. Start from a live Rust game state and build a new search tree for the move.
2. Select branches using accumulated value plus a policy-weighted exploration
   bonus (PUCT-style selection).
3. Apply moves in cloned states. At a new nonterminal leaf, request policy and
   value from the network. At a terminal leaf, use the actual outcome.
4. Back up all four value components and visit counts through the explored path.
5. Choose the move from root visit counts, using configured sampling temperature.
6. Save selected root positions, their visit distributions and final outcomes.
   PyTorch trains the network to predict those targets, exports another ONNX
   model, and self-play workers load it for subsequent games.

A "rollout" in this code is a search simulation that normally ends in a neural
leaf evaluation. It does not necessarily play an entire random game to the end.
The published benchmark/web configuration uses2,000 simulations per move;
that is a configured playing budget, not a fixed property of MCTS.

The checked live training configuration uses1,024 concurrent games and GPU
batches of128 positions. Each tree awaits one evaluation at a time; batching
comes from many independent games, without virtual loss inside each tree.
80% of moves use200 simulations and20% use1,000:360 on average. Only full-search
moves receive root exploration noise and are recorded as training examples.
Fast moves cheaply advance games; their outcomes still provide terminal targets
for the full-search positions recorded in those games.

The FiLM training configuration uses a2.7million-position recent replay window,
sampling ratio3.0, minibatch128 and learning rate0.001. The default live optimizer
is Adam. The loss is outcome cross-entropy plus weighted policy cross-entropy
(policy weight0.374 in this configuration). The default outcome target is the
final result, rather than the search's Q estimate. Other model and optimizer
experiments also exist in the repository; these values are a specific checked
configuration, not universal recommendations.

The exporter names models by cumulative recorded-sample count. The published
model name026025784 therefore suggests about26million recorded training positions.
That is an inference from the naming code, not an audited run ledger. Training
positions, games, neural leaf evaluations and optimizer updates are different
units; our earlier millions of neural evaluations cannot be compared directly
to that recorded-sample number.

Sources: [search traversal](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/crates/mcts-core/src/search.rs),
[selection/value logic](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/crates/mcts-core/src/node.rs),
[training self-play configuration](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/configs/self_play/full_mcts_vast.toml),
[full-search recording](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/src/agents/mcts/agent.rs#L51),
[loss](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/python/src/alphablokus/train_utils.py#L243),
[live trainer](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/python/scripts/train_live.py).

## Making the GPU useful

Rust runs concurrent games and move generation. A shared asynchronous batcher
collects evaluation requests. Native NVIDIA inference uses ONNX exported from
PyTorch and executed through TensorRT, with overlapping host-to-device transfer,
inference and device-to-host transfer. The README describes three CUDA streams.
The example configuration requests batches128, a pool32 and an inference cache
with up to4million entries. These are configured capacities, not measured
occupancy or throughput from our study.

Training and self-play exchange compressed data/model files through object
storage. The README says Cloudflare R2; source uses S3-compatible paths. The
browser version shares Rust MCTS through WebAssembly and uses browser inference,
so the native TensorRT training setup is not required for playing in a browser.

Sources: [request batcher](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/src/inference/batcher.rs),
[browser MCTS integration](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/crates/browser-wasm/src/lib.rs#L272),
[author's system description](https://github.com/ShivamSarodia/AlphaBlokus/blob/b4470ef9667e7c6f000a0542bb3aaa4d5cce1292/README.md).

## What this means for our project

The useful differences are a substantially deeper spatial model, explicit piece
inventory conditioning throughout it, a much larger training-data budget, and
many concurrent games feeding batched GPU inference. Our existing CNN's trunk
has four convolutions with32 channels; the checked AlphaBlokus body has ten
residual blocks with64 channels. This motivates an experiment, but does not prove
that these exact settings are best for Duo.

Its published training configuration is self-play rather than a Pentobi teacher.
Pentobi is used in separate evaluation configurations. We can study this approach
without copying expert moves or weights, while retaining Pentobi imitation as a
separate control for our learning pipeline. The author's under-$100 cost and
strength claims have not been independently reproduced here. The file named
strongest_alphablokus_v_pentobi uses20×20 Classic; its "DuoArena" label denotes
two software opponents in that arena, not14×14 Blokus Duo.
