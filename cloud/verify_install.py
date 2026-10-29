# SPDX-License-Identifier: BSD-3-Clause
"""Refuse to reuse an installation whose source/lock differs from this bundle."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def verify_install(root, installed):
    root, installed = Path(root), Path(installed)
    manifest = json.loads((root / "integration/checksums.json").read_text())
    for relative, digest in manifest.items():
        target = installed / (
            Path(relative).name if relative in ("integration/pyproject.toml", "integration/uv.lock") else relative
        )
        if target.is_symlink() or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing install differs: {relative}. Choose a fresh workspace; no files were replaced.")
    pins = json.loads((root / "integration/pins.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=installed, text=True).strip()
    if head != pins["upstream_commit"]:
        raise ValueError("Existing upstream revision differs; choose a fresh workspace")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--"], cwd=installed, text=True).splitlines()
    unexpected = sorted(set(dirty) - {"pyproject.toml", "uv.lock"})
    if unexpected:
        raise ValueError("Existing upstream tracked files differ: " + ", ".join(unexpected))
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    args = parser.parse_args()
    verify_install(Path(__file__).resolve().parents[1], args.installed)
    print("Existing pinned source and lock match this bundle")


if __name__ == "__main__":
    main()
