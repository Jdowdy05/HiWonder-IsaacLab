# SPDX-License-Identifier: BSD-3-Clause
"""Create a credential-free, byte-verified transfer archive from committed files."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def create_bundle(repo, target, files, *, revision):
    root, target = Path(repo).resolve(), Path(target).resolve()
    rows = []
    forbidden = {".git", ".ssh", ".venv", "venv", "logs", "runs", "checkpoints", "node_modules", "__pycache__"}
    for name in sorted(set(files)):
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or forbidden.intersection(path.parts)
            or any(p.startswith(".env") for p in path.parts)
            or path.suffix in {".pt", ".pth", ".onnx", ".pem", ".key", ".pyc"}
        ):
            raise ValueError(f"Excluded or unsafe bundle path: {name}")
        source = root.joinpath(*path.parts)
        if source.is_symlink() or any(p.is_symlink() for p in source.parents if p.is_relative_to(root)):
            raise ValueError(f"Symlink is not allowed: {name}")
        if not source.is_file() or not source.resolve().is_relative_to(root) or source.resolve() == target:
            raise ValueError(f"Not a project file: {name}")
        rows.append((name, source, hashlib.sha256(source.read_bytes()).hexdigest()))
    manifest = dict(revision=revision, files={name: sha for name, _, sha in rows})
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "x:gz") as archive:
        for name, source, _ in rows:
            archive.add(source, arcname="HiWonder-IsaacLab/" + name, recursive=False)
        info = tarfile.TarInfo("HiWonder-IsaacLab/BUNDLE-MANIFEST.json")
        info.size, info.mode = len(encoded), 0o644
        archive.addfile(info, io.BytesIO(encoded))
    return dict(
        path=str(target),
        revision=revision,
        file_count=len(rows),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise SystemExit("Commit reviewed changes before producing a reproducible bundle")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().strip("\x00").split("\x00")
    print(json.dumps(create_bundle(root, args.output, files, revision=revision), indent=2))


if __name__ == "__main__":
    main()
