# Frozen CUDA Graph and H100 throughput evaluation

Dennis requested a Modal GPU evaluation on 28 October 2026, emphasizing useful
speed and the possibility of saturating an H100. Training remains stopped.
This is a frozen-weight performance study, not a playing-strength experiment.

## Prespecified execution

- Serial L40S, then strict H100 (`H100!`), each one 30-minute hard-limit call,
  eight capped CPU cores, 24 GiB RAM, retries0 and max1container. No duplicates
  or automatic study repeats. H100 starts only after verified, stopped L40S.
- Combined cap **$5**; combined reservation **$4.19642**, including $0.50 image
  startup/build allowance. CLI rates: L40S total $2.5204/hour, H100 total
  $4.5204/hour with these CPU/RAM limits. Existing $130 credit authorization;
  retain $50 reserve. No top-ups, plan changes or training launch.
- Immutable base image `im-BnyLYweX6vlwu6iCpDnqrC`; source-only overlay and
  separately compiled native prototype. No additional model/replay upload.
  Private Volume `blokus-deep-selfplay-20261026`, profile `linuxfan2718`.
- Frozen selected parent SHA256
  `95569be7c179180c12681255dbcee9884f465e713f418e958a508907beb4d512`.
  Its existing latest replay-state SHA256 is
  `df74afa8bf699eb63f9dcf4d32cd13530e7b276f66c63ddd28368049edf75911`.
  Both remain unchanged. A deterministic 2,048-position input fixture is generated
  inside the private Volume from the existing replay, seed1069.

## Comparisons

First, verify the isolated Linux native implementation against the image's
original extension on 32 random games and search roots with arbitrary neural
priors/values. Compare all ordered decisions and complete root statistics.
This supplements the earlier local saved-trace tests.

CUDA Graphs capture network, legal-logit gather, softmax and clamped values.
Pinned host buffers and device tensors persist for each shape. Transfers are
ordered before graph replay; host output is read only after stream completion,
then copied into caller-owned arrays. Batch buckets 8/16/32/64/128/256 and legal
width buckets 64/256/1024/2048 bound shape variety. Unsupported shapes use the
original eager evaluator with explicit fallback accounting. No neural cache.
Record requested versus padded physical neural positions separately.

Correctness compares eager and captured outputs at identical padded shapes
(maximum absolute error <=1e-6) and ordinary unpadded bf16 inference (probability
error <=0.002, value error <=0.02). Report actual maxima, not just pass/fail.
Check normalization, finite outputs, changed inputs and output lifetime.
Padding/batching can change CUDA floating-point trajectories; bitwise search
identity across eager/graph collection is not presumed.

Microbenchmark: real frozen inputs already on GPU, batches32/128/512/2048,
eager versus captured execution, three alternating-order repetitions. Calibrate
iteration count to roughly 0.7 seconds, cap20,000 calls, and report wall/CUDA-event
time, actual iterations and positions/s. A separate ten-second sustained run at
batch2048 measures each mode with sufficient telemetry coverage. This is an
inference-service ceiling, not achieved self-play throughput.

End-to-end cases, twice in reversed order on repetition2:

1. Current engine, eager inference.
2. Current engine, CUDA Graphs.
3. Corner-index/PUCT prototype, eager inference.
4. Corner-index/PUCT prototype, CUDA Graphs.
5. Native prototype + graphs, 64 rather than 32 live games per worker.

Each case has 12 warmup games and **384 measured games**, six workers, full1024
on recorded20% of roots, fast32 on unrecorded80%, inference256/leaf8, seed1069,
CUDA bf16/TF32. Exact same weights and budgets; no gradients, model promotion or
new Pentobi games. There are 3,840 measured games plus120 warmups per GPU.
Record actual games, roots, simulations, requested/padded neural work, timing,
source/native/model hashes and record digests. Changed live-game grouping can
change execution/trajectories; actual work must accompany throughput.

## Measurements and decision

Sample GPU busy-time indicator, memory-controller busy, VRAM, power and clocks;
also cgroup CPU usage and host RAM. Join samples with explicit phase markers.
Report game/s, simulations/s, neural positions/s, end-to-end time, cold capture
overhead, GPU activity and provider cost per fixed workload. NVIDIA GPU busy
time is not percent of peak FLOPs or proof of Tensor Core saturation.

The primary decision is whether graphs increase actual collection throughput;
compare current-graph with current-eager and native-graph with native-eager.
Separate the native and concurrency effects. Compare H100/L40S throughput and
cost efficiency at matched settings. High GPU-only activity with low collection
activity identifies supply/coordination limits; do not inflate utilization with
padding and call it useful speed. These finite tests cannot establish a global
hardware maximum.

Execution ledger and exact payload hashes will be saved before launch. Preserve
all failures and atomic results; download only after each finished marker and
stopped app with zero tasks. Audit all records, costs and preserved states;
back up locally with member hashes, commit validated code/docs without pushing,
then report results. No follow-on training is part of this authorization.

References: [PyTorch CUDA Graph semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html),
[PyTorch pinned transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
[Modal GPU selection](https://modal.com/docs/guide/gpu).

## Corrected loader attempt

The first L40S app `ap-yGUCk4BBAXkUA1RrJmyJze` passed the native and graph
numerical checks and completed the inference microbenchmarks, then its first
collection subprocess hit the 160-second limit. Diagnosis found a benchmark
loader error: indexing lazy `NpzFile` members once per row repeatedly decompressed
whole arrays and retained duplicate backing arrays. The fix materializes each
array once and returns shared views; a regression test checks their shared
backing storage. Scientific settings, graph code, native code and weights stay
the same. The original protocol/source/payload/ledger are retained unchanged.

The original app is stopped with zero tasks, metered **$0.19414255**. Its complete
microbenchmarks, failure and telemetry are preserved in
`runs/duo-cuda-l40s-interrupted-20261028T043915Z/`. This is included in the same
$4.19642 reservation/$5 cap: $3.69642 for the two remaining GPU hard limits,
$0.25 allowance for the stopped attempt and $0.25 for image build/startup.
The corrected run is `duo-cuda-graphs-1069-r1`, with separate `-r1` ledgers and
payload archive. It repeats the comparisons with the corrected loader; initial
microbenchmarks remain diagnostic evidence, not mixed into its repetitions.
