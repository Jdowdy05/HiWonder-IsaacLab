# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict, partial JSON profiles for the physical camera and ten nuisance groups."""

import dataclasses
import json
from pathlib import Path

from isaaclab_hiwonder.config import HiWonderEnvConfig


def _overlay(current, values, path="profile"):
    if not isinstance(values, dict):
        raise ValueError(f"{path} must be an object")
    known = {field.name for field in dataclasses.fields(current)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {path} fields: {sorted(unknown)}")
    replacements = {}
    for name, value in values.items():
        old = getattr(current, name)
        if dataclasses.is_dataclass(old):
            value = _overlay(old, value, f"{path}.{name}")
        elif isinstance(old, tuple):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{path}.{name} must be an array")
            value = tuple(tuple(item) if isinstance(item, list) else item for item in value)
        replacements[name] = value
    return dataclasses.replace(current, **replacements)


def load_profile(path: str | Path) -> HiWonderEnvConfig:
    """Overlay a partial file on defaults; reject misspelled/unused settings."""
    return profile_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def profile_from_dict(values: dict) -> HiWonderEnvConfig:
    """Reconstruct JSON or embedded checkpoint profiles through the same strict schema."""
    return _overlay(HiWonderEnvConfig(), values)


def save_profile(profile: HiWonderEnvConfig, path: str | Path):
    Path(path).write_text(json.dumps(dataclasses.asdict(profile), indent=2) + "\n", encoding="utf-8")
