# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical bank identity independent of run output folders and USD reference spelling."""

import hashlib
import json
from dataclasses import asdict

import numpy as np


def task_definition_hash(definition) -> str:
    """Hash all task geometry, native materials, rest topology, joints, goals and initial states."""

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported physical task value {type(value).__name__}")

    payload = json.dumps(asdict(definition), sort_keys=True, separators=(",", ":"), default=encode, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def structural_snapshot_hash(snapshot: dict) -> str:
    """Pin the accepted baked cohort, excluding fresh actuator/base/sensor episode nuisance."""
    fixed = ("table", "room", "robot", "task", "distractor", "confetti")
    values = {
        group: {name: value.detach().cpu().tolist() for name, value in snapshot[group].items()} for group in fixed
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
