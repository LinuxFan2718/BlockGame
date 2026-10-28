"""Local compressed backup, with every archived member compared by SHA256 readback."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("simple archive name required")
    destination = ROOT / ".backups" / (args.name + ".tar.gz")
    destination.parent.mkdir(exist_ok=True)
    paths = {
        ROOT / "README.md",
        ROOT / "docs/control-replication.md",
        ROOT / "docs/control-replication-protocol.json",
        ROOT / "runs/integrated-validation-verified.json",
        ROOT / "python/blokus_rl/duo_control_replication_report.py",
        ROOT / "docs/control-replication-performance.md",
        ROOT / "python/blokus_rl/duo_cuda_graphs.py",
        ROOT / "python/tests/test_duo_graph_outputs.py",
        ROOT / "cloud/profile_result_copies.py",
        ROOT / "cloud/report_live_performance.py",
    }
    paths.update((ROOT / "cloud").glob("*control_replication*.py"))
    paths.update(p for p in (ROOT / "runs").glob("*replication*") if p.is_file())
    for prefix in (
        ROOT / "runs/integrated-payload-1070",
        *[p for p in (ROOT / "runs").glob("control-replication-profile-*") if p.is_dir()],
        *[p for p in (ROOT / "runs").glob("control-replication-monitor-*") if p.is_dir()],
        *args.run,
    ):
        if prefix.exists():
            paths.update(p for p in prefix.rglob("*") if p.is_file())
    paths.update((ROOT / "docs/assets").glob("control-replication-*.png"))
    manifest = {
        str(p.resolve().relative_to(ROOT)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in sorted(paths)
    }
    with tarfile.open(destination, "x:gz", compresslevel=3) as archive:
        for name in manifest:
            archive.add(ROOT / name, arcname=name, recursive=False)
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("archive-member-sha256.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(manifest) + 1 and all(m.isfile() for m in members)
        for m in members:
            if m.name == "archive-member-sha256.json":
                assert json.loads(archive.extractfile(m).read()) == manifest
            else:
                assert (
                    hashlib.file_digest(archive.extractfile(m), "sha256").hexdigest()
                    == manifest[m.name]
                ), m.name
    digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        digest + "  " + destination.name + "\n"
    )
    print(
        json.dumps(
            {
                "archive": str(destination),
                "members_verified": len(manifest) + 1,
                "sha256": digest,
                "bytes": destination.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
