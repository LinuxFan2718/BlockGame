"""Prepare an exact, source-only payload for the authorized CUDA Graph study."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from benchmark_duo_native import corner_source, priority_source

ROOT = Path(__file__).resolve().parent.parent
payload = ROOT / "runs/duo-cuda-payload"
payload.mkdir(exist_ok=False)
overlay = payload / "optimized/blokus_rl"
overlay.mkdir(parents=True)
modules = (
    "duo_deep",
    "duo_shared",
    "duo_mcts",
    "duo_indices",
    "duo_inference",
    "duo_replay",
    "duo_cuda_graphs",
)
for name in modules:
    shutil.copy2(ROOT / f"python/blokus_rl/{name}.py", overlay / (name + ".py"))
(overlay / "__init__.py").write_text(
    "import os, sysconfig\n"
    '__path__.append(sysconfig.get_paths()["purelib"] + "/blokus_rl")\n'
    'if os.environ.get("BLOKUS_NATIVE_PATH"):\n'
    '    __path__.insert(0, os.environ["BLOKUS_NATIVE_PATH"])\n'
)
for name in ("duo_cuda_benchmark.py", "duo_cuda_native_check.py", "build_cuda_native.py"):
    shutil.copy2(ROOT / "cloud" / name, payload / name)
native = payload / "native"
native.mkdir()
(native / "duo.rs").write_text(corner_source((ROOT / "src/duo.rs").read_text()))
(native / "duo_search.rs").write_text(priority_source((ROOT / "src/duo_search.rs").read_text()))
manifest = {
    str(p.relative_to(payload)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(payload.rglob("*"))
    if p.is_file()
}
with tarfile.open(ROOT / "runs/duo-cuda-payload.tar.gz", "w:gz") as tar:
    for name in manifest:
        tar.add(payload / name, arcname=name)
(ROOT / "runs/duo-cuda-payload-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {"files": len(manifest), "bytes": sum((payload / n).stat().st_size for n in manifest)}
    )
)
