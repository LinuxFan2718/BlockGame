# Blokus Duo inference and checkpoint speedups

Dennis requested stopping the running study and implementing the measured
performance improvements on 27 October 2026. The training app is stopped with
zero tasks, its follow-up is deleted, and all committed state is verified and
backed up. See [the stop record](teacher-search.md) and
[the live profile](teacher-performance-profile.md). Training is not resumed.

## Implemented behavior

- A read-only action-index table replaces per-leaf NumPy geometry conversion.
  It is derived from the Rust placement catalog for both player perspectives,
  includes forced pass, and occupies 71,348 bytes using uint16. Search supplies
  the indices directly to inference and replay; legacy candidate encodings
  remain supported.
- Shared-memory request/response buffers replace pickled inference arrays.
  Queues carry bounded request counts and sequence metadata. Each worker has
  one outstanding request, responses are copied before reuse, and the owner
  unlinks buffers after workers exit. The queue implementation remains available
  for comparison and fallback.
- An optional bounded neural-evaluation LRU also deduplicates identical rows
  within a batch. Keys contain the entire observation plus ordered legal
  indices. Cached outputs precede root noise and include no search statistics.
  A fresh cache is created for each frozen collection round, preventing stale
  outputs after a gradient update. Counters separate logical requested neural
  positions, cache hits and actual GPU-evaluated positions. Cache use remains
  explicit until its end-to-end benefit is measured.
- Atomic replay checkpoints now use flat tensors and lossless bit packing of
  the binary board/inventory/start/player features. Float32 score/pass features,
  float64 terminal targets, ordered indices and uint16 visit counts are retained
  exactly. Old list-format checkpoints remain readable. Optimizer, RNG, model
  and selected-checkpoint state remain in the atomic round file. A stale
  original replay is released after restore rather than pinned in memory.

No rules, rewards, architecture, search counts or expert demonstrations changed.
CUDA batching can already vary floating-point results across scheduling; exact
GPU trajectory identity across implementations is not claimed.

## Validation completed locally

The complete Python suite passed: **89 tests**, 49.64 seconds. This includes
all 27,460 valid placement/perspective/pass checks, queue/shared/cache collection
equivalence, key separation for inventory/player/score/pass/action order,
response lifetime and buffer cleanup, legacy and packed replay recovery,
selected-model initialization and certified tic-tac-toe/environment tests.
The PettingZoo tests emitted their existing API-advice warnings.

A separate frozen best-CNN preflight ran the archived original queue code and
the new shared-memory paths, with and without a cache, on identical seeds and
4 games each. All retained training-record hashes and all 4,480 logical
simulation counts matched exactly on CPU. Its small timing differences are
preflight evidence, not a meaningful estimate of Modal speedup.

On the actual stopped teacher checkpoint's **100,000 replay records**, every
observation, index, visit and target matched after packing and unpacking:

| Local checkpoint measurement | Original | Packed |
| --- | ---: | ---: |
| Bytes | 403,395,919 | 106,659,835 |
| Save seconds | 3.449 | 0.171 |
| Additional packing seconds | — | 0.349 |
| Load seconds | 1.238 | 0.016 |
| Additional unpacking seconds | — | 0.130 |

The artifact is **3.78× smaller**; saving including packing is **6.64× faster**
in this single local measurement. These are storage measurements, not an
end-to-end cloud training multiplier. Original checkpoints were never rewritten.

## Bounded GPU implementation benchmark

Prepared `cloud/modal_speed_benchmark.py` and `cloud/duo_speed_benchmark.py`.
This tests collection throughput with frozen weights and **zero gradient
updates**, not playing strength or continuation of the stopped study.

- Immutable original image: im-BnyLYweX6vlwu6iCpDnqrC; original installed module
  hashes are checked. The optimized overlay contains only six project modules,
  its package-path initializer and the benchmark script, approximately 51 KB.
- The existing private round-112 parent is used in place, SHA256
  95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512.
  No model/replay is uploaded.
- Four cases: original queue, lookup with queue, lookup with shared memory,
  and shared memory plus 8,192-entry cache. Two repetitions, reversed case
  order on the second; 192 measured games plus 12 warmup games per case.
- Same small CNN, seeds 1067 and rounds 1/2, six workers, 32 live games,
  full1024 on recorded 20%, fast32 on unrecorded 80%, inference256/leaf8,
  CUDA bfloat16 and TF32 permitted. Report games, simulations, requested versus
  actual neural work and wall time; differing GPU trajectories must not be
  mistaken for identical work.
- One L40S, eight capped cores, 24 GiB, 1,200-second hard limit, retries0,
  max1container. Reservation including startup is **$0.90314333** under a
  **$1.25 phase cap**, from the existing $130 authorization with $50 retained.
  No new training or automatic retry.

The launcher checks current apps, billing and a one-shot cost ledger. It freezes
the source commit and every uploaded member hash. Results must be downloaded
and checked after stopped compute, then actual metering compared with the
reservation. No end-to-end GPU result is claimed before those cases complete.

## Runtime and compatibility

`python -m blokus_rl.duo_deep` accepts `--inference-transport queue|shared_memory`
and `--inference-cache-entries N`. These execution options are recorded in new
atomic checkpoints and must match on exact recovery. A new branch chooses its
own runtime; old legacy checkpoints have no runtime declaration and remain
loadable. A source-upgraded continuation is a new implementation lineage and
must be recorded explicitly rather than relabeled as the original study.

Compiled/CUDA-graph inference, persistent subtree reuse and asynchronous Volume
commits remain candidates, not claimed implemented gains. Subtree reuse changes
budget accounting; asynchronous persistence needs stronger failure tests.
Compact atomic checkpoints address the measured storage cost without weakening
recovery guarantees.
