# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run provenance for source, asset/calibration evidence and consumed configuration."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch


def write_run_provenance(env, directory: str | Path, runner_config: dict, arguments: dict) -> dict:
    """Save owned source hashes, full profile, structural cohort and runtime version metadata."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    repository = package.parents[2]
    versions = {}
    for name in ("torch", "isaacsim", "isaaclab", "newton", "warp-lang", "rsl-rl-lib", "usd-exchange"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as distribution"
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    source_hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest() for path in package.rglob("*.py")
    }
    manifest = package / "assets" / "generated" / "manifest.json"
    metadata = dict(
        arguments=arguments,
        runner_config=runner_config,
        profile=asdict(env.profile),
        versions=versions,
        python=platform.python_version(),
        platform=platform.platform(),
        git_revision=revision.stdout.strip(),
        source_sha256=source_hashes,
        torch_cuda_runtime=torch.version.cuda,
        asset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
        physics_backend=type(env.cfg.sim.physics).__name__,
        camera_calibration_status=env.profile.camera.calibration_status,
        observation_shapes={key: list(value.shape) for key, value in env.obs_buf.items()},
        critic_layout=env.cfg.hiwonder_metadata.get("critic_layout"),
        task_support=env.cfg.hiwonder_metadata.get("task_support"),
        structural_snapshot_sha256=env.cfg.hiwonder_metadata.get("structural_snapshot_sha256"),
        physical_replay="Native solver warm-start state is not serialized; validate replay within backend tolerances",
        native_rod_inertia=getattr(getattr(env, "native_binding", None), "rod_inertia_receipt", None),
    )
    (destination / "profile.json").write_text(json.dumps(metadata["profile"], indent=2) + "\n", encoding="utf-8")
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    torch.save(env.randomization.snapshot(), destination / "structural_randomization.pt")
    if manifest.exists():
        (destination / "asset_manifest.json").write_bytes(manifest.read_bytes())
    return metadata
