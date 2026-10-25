# Modal hardware experiment — 25 October 2026

Dennis offered up to **$130 of existing Modal credits** if useful, and explicitly
selected profile **linuxfan2718** (currently active). This supersedes the earlier
blanket no-paid-cloud condition for this project. No subscriptions, top-ups or
spending beyond that budget are authorized. Local baseline training remains separate.

## Completed outcome

The explicitly approved L4 and H100! benchmarks both succeeded. A matched local
CPU workload also finished. No training checkpoint was changed or promoted, and
both Modal apps are stopped with zero running tasks. This completes the bounded
hardware assessment; no larger cloud training job or recurring follow-up is active.

| Complete search, decisions/second | Mac CPU | L4 CUDA fp32 | H100 CUDA fp32 |
| --- | ---: | ---: | ---: |
| Existing small additive network | 76.74 | 32.71 | 64.22 |
| Existing small CNN, untrained | 21.83 | 31.06 | 66.73 |

Each row uses32 live games,128 decisions and16,384 search simulations. Mac values
are medians of two executions; each cloud value is one execution. No uncertainty
interval is implied. Full measurements and a regenerable plot are in
[the hardware results](modal-results.md).

**Decision:** keep the current small-network search local. For a future CNN
learning experiment, start with L4 if cost matters. H100 was about2.15x faster
than L4 for this CNN search workload but its requested resource rate is about4x
higher. L4 therefore delivered about1.86x more search decisions per dollar.
At account API rates, projected steady execution cost for1,000 decisions of
this particular workload is $0.0094 on L4 versus $0.0175 on H100. These omit
startup, builds and idle time and are not a price forecast for complete training.

The cloud CPU baselines differed substantially: additive28.72/56.05 decisions/s
on the L4/H100 hosts. Therefore the cross-host gain cannot all be attributed to
the GPU. Within each host, CUDA accelerated CNN search2.61x/3.05x and the additive
network only1.14x/1.15x. H100 bf16 reached67.42 additive decisions/s, still below
the Mac's76.74. bf16 did not help this CNN workload. The benchmark does not test
larger models, many-worker inference, later-game positions or playing strength.

The CNN's batch128 forward pass improved31.5x on L4 and39.2x on H100 relative
to their host CPUs. Full search gained much less because the remaining work is
CPU traversal, observation/candidate packing and coordination. The next useful
scaling hypothesis is several CPU search workers sharing a GPU inference queue,
with a CPU-only multiworker control and a fixed simulation budget. A subsequent
CNN learning trial must compare held-out playing strength, not just speed. Neither
hypothesis was silently launched; the completed bounded study stops here.

## Recorded cost and preservation

Modal's per-app resource report after completion recorded **$0.05911950 total**:
L4 $0.01976449 and H100 $0.03935501, including the reported CPU, GPU, memory and
network entries. This is provider-reported metered usage, subject to billing
updates, rather than a final invoice. The earlier account summary lagged behind
the resource report; the final summary records $0.06 applied credits and $0
billed after adjustments. No top-up, subscription or plan change occurred.

The original rounded-rate ledger reserved $0.52535520 total against the $5
initial ceiling. Measured workload execution estimates alone totaled $0.04277;
startup/build/idle and rounding explain why that differs from reported usage.
The runner now uses the account's exact hourly API rates: L4 $0.80, H100 $3.95,
CPU $0.0473/core and RAM $0.008/GiB. Original result estimates remain unchanged.

Saved under runs/: both benchmark JSON/log pairs, matched Mac timings, initial
and final billing snapshots, resource-level report, stopped-app listing and the
reservation/completion ledger. The exact uploaded source archive and executed
benchmark script are retained as modal-uploaded-source-260642.tar.gz and
modal-uploaded-benchmark-260642.py. The executed script is the version in37cbeea;
subsequent changes only improve progress logging, rate precision and ledger
completion records. All original certified tic-tac-toe and frontier model hashes
were checked again and remain unchanged. A local recovery archive of29 files
is `.backups/modal-study-20261025.tar.gz` (284,108bytes; SHA256
fae4b73c1a142ca5100a46b6a8c4fb03bfab81c9ee4675ac3cee2aab55e19c6b).
This is on the same Mac, rather than an off-device backup.

Reproduce the report without launching cloud work:

```sh
MPLCONFIGDIR=/private/tmp/blokus-matplotlib .venv/bin/python cloud/report_modal.py runs/modal-benchmark-l4-260642.json runs/modal-benchmark-h100-260642.json --local-cpu runs/modal-matched-mac-cpu-260642.json
```

## Original bounded measurement plan

Initial allowance: at most **$5 estimated total** for bounded benchmarking,
reserving the rest. Read-only authentication succeeded; the account had no active
apps. Modal SDK1.6.2 is isolated in `.venv-modal/`; training dependencies are unchanged.

Executed runner: `cloud/modal_blokus.py`. It requests one L4 or one H100!, four
physical CPU cores with a matching hard CPU limit, 8GiB requested/capped RAM,
300second function timeout, one container, no automatic retries and 2second
scale-down. H100! avoids an automatic H200 upgrade during the hardware comparison.
The script records a conservative compute reservation in
`runs/modal-cost-ledger.jsonl` before each remote call and refuses reservations
over $5. It measures both the additive policy and existing small CNN:

- Inference batches1/8/32/128, including returning results to CPU.
- Live Rust search with8/32 games, four plies each,128 simulations/move. This
  includes legality, observations, candidate packing, transfers and search itself.
- Twelve synthetic optimizer steps at batch128, timing only; no benchmark-trained
  checkpoint is saved or promoted.
- CPU float32, CUDA float32 and CUDA bfloat16 autocast modes. A shared inference
  server for many CPU workers is not implemented yet, so this does not measure
  the best possible GPU throughput.

The pricing snapshot is [Modal's public list](https://modal.com/pricing): L4
$0.000222/second, H100 $0.001097/second, CPU $0.0000131/core/second, RAM
$0.00000222/GiB/second. At these requested resource limits, a300second call plus
60seconds allowance reserves roughly **$0.105 for L4** or **$0.420 for H100**.
Image builds/startup and actual metering can differ. Estimates are not verified
account charges, and the ledger does not see unrelated account usage.

## Upload and approval status

Only an explicit allowlist of Rust/Python source and build manifests is bundled:
`.local/modal-source.tar.gz`, about71KB after the native interface improvement. The runner also
passes one existing checkpoint, frontier342 best.pt,508,195bytes. The benchmark
script itself is copied separately. It does not upload home configuration,
credentials, `.git/`, the run collection or local backups. Modal's SDK handles its
own account authentication normally.

**Automatic approval review rejected the launch** because cloud credit permission
did not explicitly authorize exporting private project source/model data. Dennis
subsequently answered **“Yes, upload the code and model”** to the specific
upload question in this chat. That explicit human approval resolves the export
block for the allowlisted archive and checkpoint. No launch had occurred before
that approval. The bounded L4 benchmark is now authorized.

Authorized first command:

```sh
.venv-modal/bin/modal run --profile linuxfan2718 cloud/modal_blokus.py --gpu l4 --out runs/modal-benchmark-l4-260642.json
```

The successful L4 CNN result justified one matched H100 comparison, now also
complete. Further GPU work needs a concrete model or batching hypothesis. Keep
per-run limits, save local results and stop idle resources. All local self-play
and held-out evaluations are already complete; do not restart them.

## Local work completed while permission was pending

The original small-network search profile spent about4% in policy evaluation.
Native observation buffers removed duplicate legal generation and Python list
conversion, reducing the same representative search workload from4.70/4.45seconds
to1.62/1.62seconds: approximately**2.8x faster**, without cloud spending. Tests
verify identical observations, action lists, visit counts, priors and values.
Timing data: runs/modal-native-features-benchmark.json; original profile:
runs/modal-local-pipeline-profile.{txt,pstats}.

A matched CPU profile after that optimization used16 games, four random opening
plies and two searched plies,128 simulations/move. The MLP spent about**10.5%**
in policy evaluation; the CNN about**73.1%**. Both evaluated4128 neural positions.
Total profiled times were0.456/1.419seconds. The models can explore different
positions, and profiling overhead is included, so this is a workload diagnostic,
not a claimed GPU speedup. The CNN used untrained weights for timing only.
Data: runs/modal-matched-cpu-profiles.json. This gives a concrete reason to test
a GPU for the spatial network; it does not establish that the CNN plays better.

Install the isolated client reproducibly with:

```sh
env UV_CACHE_DIR=/private/tmp/blokus-uv-cache uv venv --python .venv/bin/python .venv-modal
env UV_CACHE_DIR=/private/tmp/blokus-uv-cache uv pip install --python .venv-modal/bin/python -r cloud/requirements-modal.txt
```

The upload permission comes from Dennis's explicit human reply, not profiling
or heartbeat messages. The bounded L4 and H100 benchmarks have now completed.
