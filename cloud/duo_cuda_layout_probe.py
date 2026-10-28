"""Run immutable-image benchmarks with explicit delay/layout overrides."""

import argparse
import copy
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "micro", "collection"), required=True)
    parser.add_argument("--layout", choices=("nchw", "channels_last"), default="nchw")
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--tag", default="layout")
    args = parser.parse_args()
    sys.path.insert(0, "/opt/cg/payload")
    import duo_cuda_benchmark as benchmark

    if args.mode == "check":
        benchmark.configure_torch(1069)
        torch.set_float32_matmul_precision("high")
        model = benchmark.load_duo(args.model).cuda().eval()
        converted = copy.deepcopy(model).to(memory_format=torch.channels_last).eval()
        for name, value in model.state_dict().items():
            assert torch.equal(value.cpu(), converted.state_dict()[name].cpu())
        rows = benchmark.fixture(args.fixture)
        checks = []
        with torch.no_grad():
            for count in (1, 33, 128, 256):
                expected = benchmark.evaluate_indexed(model, rows[:count], "bfloat16")
                actual = benchmark.evaluate_indexed(converted, rows[:count], "bfloat16")
                error = [
                    float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)
                ]
                checks.append({"rows": count, "max_abs_error": error})
        result = {
            "weights_exact": True,
            "checks": checks,
            "passed": all(
                r["max_abs_error"][0] <= 0.002 and r["max_abs_error"][1] <= 0.02 for r in checks
            ),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        }
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    benchmark.DeepConfig = functools.partial(benchmark.DeepConfig, batch_wait_ms=args.delay)
    if args.layout == "channels_last":
        original = benchmark.load_duo
        benchmark.load_duo = lambda path: original(path).to(memory_format=torch.channels_last)
    sys.argv = [
        "duo_cuda_benchmark.py",
        "--model",
        str(args.model),
        "--fixture",
        str(args.fixture),
        "--out",
        str(args.out),
        "--repeat",
        str(args.repeat),
        "--tag",
        args.tag,
        "--variant",
        "micro" if args.mode == "micro" else "native-graph-wide",
    ]
    benchmark.main()
    result = json.loads(args.out.read_text())
    result.update({"layout": args.layout, "delay_ms": args.delay})
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
