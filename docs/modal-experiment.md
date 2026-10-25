# Modal hardware experiment — 25 October 2026

Dennis offered up to **$130 of existing Modal credits** if useful, and explicitly
selected profile **linuxfan2718** (currently active). This supersedes the earlier
blanket no-paid-cloud condition for this project. No subscriptions, top-ups or
spending beyond that budget are authorized. Local baseline training remains separate.

## First stage: measure value

Initial allowance: at most **$5 estimated total** for bounded benchmarking,
reserving the rest. Read-only authentication succeeded; the account had no active
apps. Modal SDK1.6.2 is isolated in `.venv-modal/`; training dependencies are unchanged.

Prepared runner: `cloud/modal_blokus.py`. It requests one L4 or one H100!, four
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
has been asked specifically to authorize that upload. No launch, source upload,
image build or paid Modal compute has occurred. Do not bypass the rejection,
change transport or retry until the human authorizes the upload in this chat.

Prepared command, to use only after authorization:

```sh
.venv-modal/bin/modal run --profile linuxfan2718 cloud/modal_blokus.py --gpu l4 --out runs/modal-benchmark-l4-260642.json
```

After a successful L4 run, inspect full pipeline throughput and cost before the
H100 comparison. A larger GPU run requires measured benefit or a concrete model
experiment whose compute requirements justify it. Do not spend the full balance
merely because credits are available. Keep per-run limits, save local results,
and stop idle resources. Existing local self-play and held-out evaluations can
continue while upload approval is pending.

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

No project export permission has been inferred from the profiling work or from
the heartbeat messages. All local study/evaluation jobs are complete. The next
cloud step remains waiting for the human's explicit upload approval.
