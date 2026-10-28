"""Build production Rust sources against the immutable image's locked toolchain."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

origin, build, payload = (
    Path("/opt/blokus"),
    Path("/opt/integrated/source"),
    Path("/opt/integrated/payload"),
)
build.mkdir(parents=True)
for name in ("src", "bindings", "python"):
    shutil.copytree(
        origin / name, build / name, ignore=shutil.ignore_patterns("target", "__pycache__")
    )
for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
    if (origin / name).exists():
        shutil.copy2(origin / name, build / name)
for p in (payload / "native").rglob("*"):
    if p.is_file():
        shutil.copy2(p, build / p.relative_to(payload / "native"))
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
        "/opt/integrated/wheels",
    ],
    cwd=build,
    check=True,
)
destination = Path("/opt/integrated/native-extension")
destination.mkdir()
with zipfile.ZipFile(next(Path("/opt/integrated/wheels").glob("*.whl"))) as wheel:
    name = next(n for n in wheel.namelist() if "/_native." in n and n.endswith(".so"))
    native = destination / Path(name).name
    native.write_bytes(wheel.read(name))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


baseline = Path(importlib.util.find_spec("blokus_rl._native").origin)
record = {
    "native_sha256": sha(native),
    "baseline_native_sha256": sha(baseline),
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "locked_manifests": {
        p: sha(build / p)
        for p in (
            "Cargo.toml",
            "Cargo.lock",
            "bindings/python/Cargo.toml",
            "bindings/python/Cargo.lock",
        )
    },
}
Path("/opt/integrated/native-build.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
