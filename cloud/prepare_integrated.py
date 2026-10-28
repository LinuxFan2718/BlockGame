"""Freeze the allowlisted source-only payload for integrated GPU validation/training."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/integrated-payload-1070"
MODULES = (
    "duo_deep",
    "duo_shared",
    "duo_mcts",
    "duo_indices",
    "duo_inference",
    "duo_replay",
    "duo_cuda_graphs",
    "duo_env",
    "duo_deep_evaluate",
    "duo_continuation",
)


def main():
    OUT.mkdir(exist_ok=False)
    for name in MODULES:
        target = OUT / "python/blokus_rl" / (name + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "python/blokus_rl" / (name + ".py"), target)
    for name in (
        "src/duo.rs",
        "src/duo_search.rs",
        "bindings/python/src/lib.rs",
        "bindings/python/src/duo_features.rs",
    ):
        target = OUT / "native" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in (
        "build_integrated.py",
        "validate_integrated.py",
        "duo_cuda_native_check.py",
        "run_integrated_study.py",
    ):
        shutil.copy2(ROOT / "cloud" / name, OUT / name)
    (OUT / "python/blokus_rl/__init__.py").write_text(
        'import sysconfig\n__path__.append("/opt/integrated/native-extension")\n'
        '__path__.append(sysconfig.get_paths()["purelib"] + "/blokus_rl")\n'
    )
    hashes = {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.rglob("*"))
        if p.is_file()
    }
    (ROOT / "runs/integrated-payload-1070-sha256.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "files": len(hashes),
                "bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
            }
        )
    )


if __name__ == "__main__":
    main()
