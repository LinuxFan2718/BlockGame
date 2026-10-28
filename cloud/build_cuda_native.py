"""Build only the isolated native benchmark extension inside the cached image."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

origin, build, payload = Path("/opt/blokus"), Path("/opt/cg/native-source"), Path("/opt/cg/payload")
build.mkdir(parents=True)
baseline = {}
for name in ("src", "bindings", "python"):
    shutil.copytree(
        origin / name, build / name, ignore=shutil.ignore_patterns("target", "__pycache__")
    )
for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
    if (origin / name).exists():
        shutil.copy2(origin / name, build / name)
for name in ("duo.rs", "duo_search.rs"):
    baseline[name] = hashlib.sha256((build / "src" / name).read_bytes()).hexdigest()
    shutil.copy2(payload / "native" / name, build / "src" / name)
subprocess.run(
    [
        sys.executable,
        "-m",
        "maturin",
        "build",
        "--release",
        "--offline",
        "--manifest-path",
        str(build / "bindings/python/Cargo.toml"),
        "--out",
        "/opt/cg/wheels",
    ],
    cwd=build,
    check=True,
)
destination = Path("/opt/cg/native-extension")
destination.mkdir()
with zipfile.ZipFile(next(Path("/opt/cg/wheels").glob("*.whl"))) as wheel:
    member = next(n for n in wheel.namelist() if "/_native." in n and n.endswith(".so"))
    extension = destination / Path(member).name
    extension.write_bytes(wheel.read(member))
old = Path(importlib.util.find_spec("blokus_rl._native").origin)
record = {
    "original_source_sha256": baseline,
    "optimized_source_sha256": {
        name: hashlib.sha256((build / "src" / name).read_bytes()).hexdigest()
        for name in ("duo.rs", "duo_search.rs")
    },
    "baseline_extension_sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
    "optimized_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
}
Path("/opt/cg/native-build.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
