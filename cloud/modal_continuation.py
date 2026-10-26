"""One bounded paired continuation study; all parent artifacts stay in Modal."""

import hashlib
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import modal
import modal_deep as prior

ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "duo-continuation-865"
TIMEOUT = 32400
PHASE_CEILING = 25.0
TOTAL_CREDITS, RESERVE = 130.0, 50.0
OTHER_ACTIVE_RESERVATION = 3.43666  # Existing CPU-only levels 1–9 benchmark.
HOURLY = 1.95 + 8 * 0.0473 + 24 * 0.008
PARENT_SHA = "0ba92438d27fb72f23364547ba3717729464ace9af23a414cfdff1cbd3298f86"
STATE_SHA = "2df38e9b7585c23fe0b8159cc4511db1811c0b745534d8a9223e885dec1db52d"
PARENT_ROOT = Path("/data/duo-deep-study-764")
image = (
    prior.image.add_local_file(__file__, "/opt/blokus/cloud/modal_continuation.py", copy=True)
    if modal.is_local()
    else None
)
app = modal.App("blokus-controlled-continuation")


@app.function(
    image=image,
    gpu="L40S",
    cpu=(8, 8),
    memory=(24576, 24576),
    timeout=TIMEOUT,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    include_source=False,
    volumes={"/data": prior.volume},
)
def study(execution):
    import torch
    from blokus_rl.duo_continuation import initialize_branch, sha256
    from blokus_rl.duo_crossplay import opening_book
    from blokus_rl.duo_deep import DeepConfig, atomic_json, train
    from blokus_rl.duo_deep_evaluate import evaluate_many, evaluate_pentobi
    from blokus_rl.duo_learning import load_duo
    from blokus_rl.learning import configure_torch

    root = Path("/data") / RUN_NAME
    if root.exists():
        raise ValueError("existing study; inspect checkpoints, never automatically repeat")
    parent_path = PARENT_ROOT / "frozen-conv.pt"
    state_path = PARENT_ROOT / "conv/resume.pt"
    if sha256(parent_path) != PARENT_SHA or sha256(state_path) != STATE_SHA:
        raise ValueError("preserved parent weights/optimizer/replay hash changed")
    root.mkdir()
    started = time.monotonic()
    atomic_json(root / "execution.json", execution)
    configure_torch(865)
    torch.set_float32_matmul_precision("high")
    config = DeepConfig(
        architecture="conv",
        seed=865,
        rounds=128,
        eval_every=16,
        validation_openings=32,
        validation_seed=2583000,
        test_seed=2589000,
        max_seconds=7200,
    )
    configs = {
        "standard256": config,
        "strong1024": replace(config, full_simulations=1024, max_seconds=18000),
    }
    atomic_json(
        root / "study-protocol.json",
        {
            "parent_sha256": PARENT_SHA,
            "source_resume_sha256": STATE_SHA,
            "configs": {key: asdict(value) for key, value in configs.items()},
            "primary": "strong1024 vs standard256, each search128, 100 fresh opening pairs",
            "heldout_seed": 2589000,
            "selection": "32 validation pairs against frozen parent; parent retained unless improved",
            "search_diagnostic_seed": 2581000,
            "search_diagnostic_pairs": 24,
            "no_expert_labels": True,
            "hardware": {"gpu": torch.cuda.get_device_name(), "cpu_cap": 8, "ram_gib_cap": 24},
        },
    )
    prior.volume.commit()

    def evaluate_case(name, policy, opponent, book, budget, opponent_budget, seed):
        result = evaluate_many(
            policy,
            opponent,
            book,
            simulations=budget,
            opponent_simulations=opponent_budget,
            seed=seed,
            precision="bfloat16",
        )
        atomic_json(root / f"{name}.json", result)
        prior.volume.commit()
        print(json.dumps({"case": name, "score": result["score_rate"]}), flush=True)

    try:
        with prior.Telemetry(root / "telemetry.jsonl"):
            # An actual CUDA train/recovery screen; smoke weights are never promoted.
            smoke = replace(
                config,
                seed=2591000,
                rounds=2,
                games_per_round=8,
                workers=2,
                live_games=4,
                full_simulations=16,
                fast_simulations=4,
                eval_every=2,
                validation_openings=1,
                validation_simulations=8,
                validation_seed=2593000,
                max_seconds=300,
            )
            initialize_branch(smoke, root / "smoke", parent_path, state_path)
            train(smoke, root / "smoke", parent_path, resume=True, stop_after_round=1)
            train(smoke, root / "smoke", parent_path, resume=True, commit=prior.volume.commit)
            parent = load_duo(parent_path).to("cuda").eval()
            diagnostic_book = opening_book(24, seed=2581000)
            for budget in (512, 2048):
                evaluate_case(
                    f"diagnostic-parent-search{budget}-vs-search128",
                    parent,
                    parent,
                    diagnostic_book,
                    budget,
                    128,
                    2581000,
                )
            for name, selected in configs.items():
                out = root / name
                initialize_branch(selected, out, parent_path, state_path)
                prior.volume.commit()
                train(selected, out, parent_path, resume=True, commit=prior.volume.commit)
            hashes = {}
            for name in configs:
                frozen = root / f"frozen-{name}.pt"
                frozen.write_bytes((root / name / "best.pt").read_bytes())
                hashes[name] = sha256(frozen)
            # Freeze before creating the reserved test opening book.
            atomic_json(
                root / "heldout-protocol.json",
                {
                    "checkpoint_sha256": hashes,
                    "parent_sha256": PARENT_SHA,
                    "seed": 2589000,
                    "pairs": 100,
                    "simulations": 128,
                    "precision": "bfloat16",
                },
            )
            prior.volume.commit()
            book = opening_book(100, seed=2589000)
            models = {
                name: load_duo(root / f"frozen-{name}.pt").to("cuda").eval() for name in configs
            }
            for budget, mode in ((128, "search128"), (0, "raw")):
                evaluate_case(
                    f"heldout-strong-vs-standard-{mode}",
                    models["strong1024"],
                    models["standard256"],
                    book,
                    budget,
                    budget,
                    2589000,
                )
                for name, model in models.items():
                    evaluate_case(
                        f"heldout-{name}-vs-parent-{mode}",
                        model,
                        parent,
                        book,
                        budget,
                        budget,
                        2589000,
                    )
            # Pentobi is evaluation-only. Include the parent on this same new book.
            for name, model in {"parent": parent, **models}.items():
                for budget, mode in ((0, "raw"), (128, "search128")):
                    result = evaluate_pentobi(
                        model,
                        "/opt/pentobi-build/pentobi_gtp/pentobi-gtp",
                        book,
                        root / f"heldout-{name}-vs-pentobi-{mode}.json",
                        simulations=budget,
                        seed=2589000,
                        precision="bfloat16",
                        commit=prior.volume.commit,
                    )
                    print(
                        json.dumps(
                            {"case": f"{name}-pentobi-{mode}", "score": result["score_rate"]}
                        ),
                        flush=True,
                    )
        atomic_json(
            root / "finished.json",
            {
                "seconds": time.monotonic() - started,
                "finish_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_sha256": hashes,
                "training": {
                    name: json.loads((root / name / "completed.json").read_text())
                    for name in configs
                },
            },
        )
        prior.volume.commit()
        return {
            "directory": RUN_NAME,
            "seconds": time.monotonic() - started,
            "checkpoint_sha256": hashes,
        }
    except BaseException:
        atomic_json(
            root / "failure.json",
            {"traceback": traceback.format_exc(), "seconds": time.monotonic() - started},
        )
        prior.volume.commit()
        raise


@app.local_entrypoint()
def main():
    ledger = ROOT / "runs/modal-continuation-cost-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    reservation = HOURLY * (TIMEOUT + 90) / 3600
    if sum(row["reserved_usd"] for row in rows) + reservation > PHASE_CEILING:
        raise ValueError("continuation phase reservation cap reached; no repeated launches")
    billing = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "modal",
                "billing",
                "summary",
                "--json",
                "--profile",
                "linuxfan2718",
            ],
            text=True,
        )
    )
    # Conservative floor covers earlier finished work plus accounting lag.
    committed = max(15.0, float(billing["metered_cost"])) + OTHER_ACTIVE_RESERVATION
    if committed + reservation > TOTAL_CREDITS - RESERVE:
        raise ValueError("launch would encroach on the declared $50 reserve")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = ROOT / f"runs/{RUN_NAME}-{stamp}"
    payload = ROOT / ".local/modal-source.tar.gz"
    Path(str(prefix) + "-source.tar.gz").write_bytes(payload.read_bytes())
    Path(str(prefix) + "-runner.py").write_bytes(Path(__file__).read_bytes())
    Path(str(prefix) + "-billing-before.json").write_text(json.dumps(billing, indent=2) + "\n")
    execution = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "profile": "linuxfan2718",
        "reserved_usd": reservation,
        "phase_ceiling_usd": PHASE_CEILING,
        "reserve_usd": RESERVE,
        "hourly_resource_estimate_usd": HOURLY,
        "timeout_seconds": TIMEOUT,
        "prior_committed_floor_plus_active_reservations_usd": committed,
        "source_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "parent_sha256": PARENT_SHA,
        "source_resume_sha256": STATE_SHA,
        "checkpoint_source": "existing private Volume; no model or replay upload",
    }
    with ledger.open("a") as handle:
        handle.write(json.dumps(execution) + "\n")
    result = study.remote(execution)
    (ROOT / f"runs/{RUN_NAME}-modal-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
