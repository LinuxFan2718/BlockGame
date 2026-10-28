"""Back up this port and timing evidence, verifying every archive member."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("simple archive name required")
    paths = {ROOT / name for name in (
        "README.md", "docs/local-play.md", "docs/metal-training.md",
        "docs/assets/metal-training-results.png", "python/tests/test_duo_metal_training.py",
        "python/blokus_rl/duo_deep.py", "python/blokus_rl/duo_continuation.py",
        "python/blokus_rl/duo_training_device.py", "python/blokus_rl/duo_metal_train.py",
        "python/blokus_rl/duo_training_benchmark.py", "cloud/modal_metal_training.py",
        "cloud/report_metal_training.py", "cloud/backup_metal_training.py",
        "runs/control-replication-preserved-before.json",
    )}
    for prefix in (ROOT / "runs").glob("*metal*"):
        if prefix.is_file():
            paths.add(prefix)
        elif prefix.is_dir():
            paths.update(p for p in prefix.rglob("*") if p.is_file())
    paths = {p for p in paths if "__pycache__" not in p.parts}
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    archive_path = ROOT / ".backups" / (args.name + ".tar.gz")
    archive_path.parent.mkdir(exist_ok=True)
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    with tarfile.open(archive_path, "x:gz", compresslevel=3) as archive:
        for name in manifest:
            archive.add(ROOT / name, arcname=name, recursive=False)
        info = tarfile.TarInfo("archive-member-sha256.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(manifest) + 1 or not all(m.isfile() for m in members):
            raise ValueError("archive inventory mismatch")
        for member in members:
            with archive.extractfile(member) as handle:
                if member.name == "archive-member-sha256.json":
                    if handle.read() != data:
                        raise ValueError("archive manifest mismatch")
                elif hashlib.file_digest(handle, "sha256").hexdigest() != manifest[member.name]:
                    raise ValueError("archive content mismatch: " + member.name)
    result = {"archive": str(archive_path.relative_to(ROOT)), "sha256": digest(archive_path),
              "members_verified": len(manifest) + 1, "bytes": archive_path.stat().st_size}
    archive_path.with_suffix(".gz.sha256").write_text(result["sha256"] + "  " + archive_path.name + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
