#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Install this overlay into a new pinned Isaac Lab clone (Python 3.12+)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def verified_payload(repo: Path) -> list[tuple[Path, Path]]:
    """Check every installation file before cloning or copying anything."""
    checksums = json.loads((repo / "integration/checksums.json").read_text(encoding="utf-8"))
    required = {"integration/pyproject.toml", "integration/uv.lock"}
    if not required.issubset(checksums):
        raise ValueError("Missing integration manifests in checksum list")
    payload = []
    for name, expected in checksums.items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or not (
                name.startswith("source/isaaclab_hiwonder/")
                or name in {*required, "integration/pins.json"}
            )
        ):
            raise ValueError(f"Unsafe payload path: {name}")
        source = repo.joinpath(*relative.parts)
        if source.is_symlink() or any(parent.is_symlink() for parent in source.parents if parent != repo.parent):
            raise ValueError(f"Unsafe symlink in payload: {name}")
        if not source.is_file():
            raise ValueError(f"Missing payload file: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        destination = Path(relative.name) if name in required else Path(*relative.parts)
        payload.append((source, destination))
    return payload


def install(repo: Path, target: Path, skip_sync: bool = False, uv: str = "uv") -> None:
    """Create a fresh checkout, refusing any existing destination."""
    repo = repo.resolve()
    target = target.absolute()
    if target.exists() or target.is_symlink():
        raise ValueError(f"Target already exists; choose a new directory: {target}")
    if not target.parent.is_dir():
        raise ValueError(f"Target parent must already exist: {target.parent}")
    payload = verified_payload(repo)
    pins = json.loads((repo / "integration/pins.json").read_text(encoding="utf-8"))
    if not skip_sync:
        output = subprocess.check_output([uv, "--version"]).decode().strip()
        match = re.match(r"uv (\d+)\.(\d+)\.(\d+)(?:\s|$)", output)
        minimum = tuple(int(value) for value in pins["uv_minimum"].split("."))
        if not match or tuple(map(int, match.groups())) < minimum:
            raise ValueError(f"uv >= {pins['uv_minimum']} is required; found {output!r}")
    with tempfile.TemporaryDirectory(prefix=".hiwonder-setup-", dir=target.parent) as scratch:
        checkout = Path(scratch) / "checkout"
        subprocess.run(
            ["git", "-c", "core.autocrlf=false", "clone", "--config", "core.longpaths=true", "--depth", "1", "--branch", pins["upstream_tag"],
             pins["upstream_url"], str(checkout)],
            check=True,
        )
        revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).decode().strip()
        if revision != pins["upstream_commit"]:
            raise ValueError(f"Upstream revision mismatch: {revision}")
        manifest = subprocess.check_output(["git", "-C", str(checkout), "show", "HEAD:pyproject.toml"])
        if hashlib.sha256(manifest).hexdigest() != pins["upstream_pyproject_sha256"]:
            raise ValueError("Unexpected upstream manifest; refusing overlay")
        for source, relative in payload:
            destination = checkout / relative
            if relative.parts[:2] == ("source", "isaaclab_hiwonder") and destination.exists():
                raise ValueError(f"Upstream already contains a HiWonder file: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if target.exists() or target.is_symlink():
            raise ValueError(f"Target appeared during installation: {target}")
        checkout.rename(target)
    print(f"Pinned sources installed: {target}")
    if not skip_sync:
        subprocess.run(
            [uv, "sync", "--locked", "--extra", "isaacsim", "--extra", "rsl-rl", "--extra", "test", "--inexact"],
            cwd=target,
            check=True,
        )
    else:
        print("SDK sync skipped. Run the documented uv sync command inside this new checkout when ready.")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=repo.parent / "IsaacLab-HiWonder", help="new clone directory")
    parser.add_argument("--skip-sync", action="store_true", help="clone and overlay without downloading the SDK")
    parser.add_argument("--uv", default="uv", help="installed uv executable (>=0.13.1)")
    args = parser.parse_args()
    try:
        install(repo, args.target, args.skip_sync, args.uv)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Setup failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
