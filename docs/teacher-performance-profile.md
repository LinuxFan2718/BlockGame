# Live performance profile of the teacher-search study

Measured 27 October 2026, about 23:06–23:15 EDT, while the existing Modal
recovery continued. No new instance, model upload, training change or retry.

**The hardware is not saturated. The leading bottleneck is the host-side
search/inference pipeline, with additional checkpoint/commit overhead.** This
is a resource and application-timing profile, not a native/CUDA stack trace.

| Resource | Measurement | Interpretation |
| --- | --- | --- |
| GPU activity | 5.09% sample mean, median 6%, 95th percentile 7%; 2,618 existing samples | The L40S spends most of its time without active kernels. This is NVIDIA's busy-time indicator, not a percentage of peak FLOPs. |
| GPU memory capacity | Peak 746 / 46,068 MiB, 1.62% | VRAM capacity is far above this model's needs. Filling VRAM is not itself a performance goal. |
| CPU | 3.41 of 8 allocated cores on average over 181.28 seconds; median 3.79, sampled peak 4.91 | About 43% average utilization of the allocation. More cores alone are not the first remedy. |
| System RAM capacity | 13.62–14.06 GiB of 24 GiB during that sample | Roughly 57–59% occupied; capacity is not exhausted. Host DRAM bandwidth was not measured. |

The longer CPU sample covers teacher progress from completed round 25 to 26.
One shorter 30-second window averaged only 0.91 cores, illustrating why short
windows, potentially including a round tail or checkpoint, should not be
treated as typical CPU utilization.
Per-process mean CPU use during the longer window was 0.71 cores for the main
process and about 0.42–0.48 for each of the six search workers. Process RSS values
share pages and must not be summed as physical RAM; the table uses the container
memory counter instead. Swap was zero in the available `/proc/meminfo` snapshot.

Ten additional GPU readings over 45 seconds showed 0–8% activity, 746 MiB VRAM,
and a rounded 0% memory-controller busy indicator. These corroborate low GPU
use, but do not measure detailed memory bandwidth, occupancy or kernel latency.

## Where the time goes

Existing counters cover the complete 64-round control and the first 24 teacher
rounds at the downloaded snapshot. The running teacher continued afterward.

| Counter | Control 1,024 | Teacher 2,048, first 24 rounds |
| --- | ---: | ---: |
| Retained additional self-play games | 16,384 | 6,144 |
| Search simulations | 131,407,680 | 92,507,552 |
| Neural positions | 102,746,716 | 69,721,981 |
| GPU inference batches | 1,320,157 | 912,186 |
| Mean actual GPU batch / configured maximum | 77.83 / 256 | 76.43 / 256 |
| Mean worker request batch | 32.70 | 31.42 |
| Collection wall seconds | 6,323.11 | 4,340.04 |
| Inference-service wrapper seconds | 2,789.87 | 1,916.53 |
| Gradient-update seconds | 11.41 | 4.08 |
| Validation seconds | 512.48 | 130.76 |
| Sum of timed round seconds | 7,159.01 | 4,594.67 |
| Training elapsed seconds | 7,984.06 completed | 4,895.65 through saved round 24 |

Gradient updates consume less than 0.2% of timed round wall time. Generating
self-play experience dominates. The inference-service wrapper occupies about
44% of collection wall time and averages about 2.1 milliseconds per GPU batch.
It includes NumPy/Torch packing, host/device copies, Python result handling and
synchronous device-to-host results; it is **not pure GPU computation time**.

Summed worker search seconds overlap across processes and include waiting for
remote inference. They must not be added to collection wall time. Similarly,
the 44% wrapper fraction cannot be added to worker search time.

About 825 seconds for the completed control and 301 seconds through teacher
round 24 fall outside the sum of timed rounds, approximately 10.3% and 6.1% of
elapsed time. The code places Volume commits after each timed round. This
residual also includes setup, shutdown and timing-boundary effects, so it is
not an exact isolated measurement of commit time. Another 312 and 120 seconds,
respectively, remain inside rounds after subtracting collection, gradient work
and validation; checkpoint serialization is in this region.

## Bottleneck diagnosis

Measured: idle GPU, unused aggregate CPU capacity, partly filled inference
batches, and millions of small inference exchanges. Source inspection shows
six workers repeatedly sending NumPy arrays through multiprocessing queues,
blocking for replies, and a single parent packing and serving the requests.
Each small batch can wait up to one millisecond to coalesce. Every move starts
a fresh Rust search tree; there is no cross-position neural cache or subtree
reuse. Each neural leaf also repeats a geometry-to-policy-index conversion.

The strongest inference is **latency and coordination in the host/search/GPU
pipeline**, rather than a shortage of GPU arithmetic, VRAM or allocated cores.
The counters do not establish the exact split between Rust move generation,
Python conversions, pickling, scheduling, copies and CUDA launch overhead.
`py-spy`, `perf` and `nsys` were absent, and this container did not expose
`wchan`, `schedstat` or cgroup CPU-throttling statistics. No invasive profiler
was installed or attached and the running experiment was not instrumented.

## Concrete cache test

A local, separate component benchmark precomputed the mapping from each engine
action to its policy-output index for each player perspective. This mapping is
independent of game position and learned weights. The current code recomputes
it from candidate cell geometry for every evaluated leaf.

- Table size: **142,696 bytes / 139.35 KiB**; build time about 19 milliseconds.
- Exhaustive equality checks: **27,460**, covering all 13,729 valid placements
  plus pass in both player perspectives.
- Real workload shapes: 425 decisions / 91,478 legal candidates, replayed from
  12 already completed benchmark games. No new opponent games were played.
- Seven timing repetitions, each 2,125 calls: current median **75.47 ms**,
  cached lookup **2.08 ms**, a **36.36× component speedup** on this Mac.
- Integer outputs matched exactly. **This is not a measured end-to-end cloud
  speedup**, and does not determine what fraction of the running job this
  conversion consumes. No production code was changed.

## Recommended optimization order

1. **Precompute the deterministic action-index table**, preferably emit compact
   policy indices directly from Rust. This removes repeated geometry work and
   can reduce the bytes and objects crossing Python boundaries. This is a
   correctness-preserving optimization with a cheap exhaustive check.
2. **Reduce inference-exchange overhead.** Test shared-memory buffers instead
   of repeatedly pickled arrays, fewer larger requests, and more asynchronous
   work in flight. Simply making the batching delay longer can hurt latency;
   benchmark throughput and tail behavior at equal work. More workers require
   rechecking RAM and game/search accounting.
3. **Benchmark compiled or captured GPU inference** with a few fixed batch-size
   buckets and reusable buffers. CUDA Graphs reduce repeated kernel-launch
   overhead; they do not remove Python queue traffic or all host packing.
   [PyTorch's explanation](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
   and [tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).
4. **Measure a bounded neural-evaluation cache's hit rate.** Different move
   orders can reach the same position. Cache raw policy/value evaluations,
   keeping search visit statistics separate, and invalidate/version entries
   whenever weights change. Keys must cover inventories, player, scores and
   pass/start state as well as board occupancy; if caching legal probabilities,
   action ordering must also match. Hit rates and hashing overhead are unknown
   here. [Batch Monte Carlo Tree Search](https://arxiv.org/abs/2104.04278)
   describes this separation of inference cache and search statistics.
5. **Reduce checkpoint serialization/commit cost** with compact replay arrays,
   immutable snapshots and safe overlap or incremental persistence. Keep atomic
   recovery semantics and verify optimizer/replay/RNG restoration. Do not trade
   away the recovery guarantees that preserved this study.

A cheaper GPU could plausibly deliver comparable throughput after measurement;
an H100 is not justified by this profile alone. Any hardware or pipeline
comparison should report fixed-work throughput and cost per game, not VRAM
occupancy alone. Subtree reuse is another future search option, but it changes
effective search budgets and target accounting and needs its own controlled
validation. No new paid benchmark or training job was launched.

## Reproduction and raw evidence

Metadata, read-only resource samples, probe failures, timing analysis and the
local cache benchmark script/results are preserved under
`runs/teacher-live-profile-20261028T030627Z/`. Profiling used the existing
container `ta-01M5ZVVMQVKQXNSG960GW4V8YR`, app
`ap-PxLumtPesHAVP1q9Gz82TA`, via `modal container exec --no-pty ... -- python -c`.
Existing telemetry/progress files were downloaded using `modal volume get`.

The first diagnostic needed two compatibility corrections: this container uses
cgroup v1, and large single-line exec output was truncated at 40,960 bytes by
the client path. The successful probe used v1 counters and compact JSON; large
existing files were fetched through Volume downloads. Raw failed outputs are
retained and are not treated as measurements. The extra sampler CPU work was
small but included in the container's aggregate CPU counter. No project source
or model file upload, checkpoint reset, search change or additional instance
occurred.
