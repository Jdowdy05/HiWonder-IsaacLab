# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Pure model/control compatibility checks shared by launchers and native motors."""

import re

from ..assets.robot_models import get_robot_model


def validate_checkpoint_robot(saved: dict, robot_model="stock") -> None:
    """Reject cross-model acting weights as well as incompatible resume histories."""
    selected = get_robot_model(robot_model)
    checkpoint = saved.get("hiwonder", {})
    profile = checkpoint.get("metadata", {}).get("profile", {})
    recorded = profile.get("robot_model", checkpoint.get("environment", {}).get("robot_model", "stock"))
    if recorded != selected.name:
        raise ValueError(f"Checkpoint robot model {recorded!r} differs from selected {selected.name!r}")
    signature = checkpoint.get("metadata", {}).get("robot_joint_names")
    if signature is not None and tuple(signature) != selected.joint_names:
        raise ValueError("Checkpoint robot joint/action order differs from selected model")
    shapes = checkpoint.get("metadata", {}).get("observation_shapes", {})
    for name, width in (("proprio", 3 * selected.num_joints), ("critic", selected.critic_width)):
        if name in shapes and (len(shapes[name]) != 2 or shapes[name][-1] != width):
            raise ValueError(f"Checkpoint {name} observation shape differs from selected robot dimensions")
    actions = checkpoint.get("environment", {}).get("actions")
    if actions is not None and (actions.ndim != 2 or actions.shape[-1] != selected.num_joints):
        raise ValueError("Checkpoint action dimensions differ from selected robot model")


def robot_motor_indices(labels) -> list[int] | None:
    """Resolve complete homogeneous robot motor topology without backend DOF assumptions."""
    selected = {}
    for index, label in enumerate(labels):
        match = re.search(r"/env_(\d+)/Room/Robot/joints/([^/]+)$", label)
        if match:
            if match[2] == "fixed_base":
                continue
            key = (int(match[1]), match[2])
            if key in selected:
                raise ValueError("Native robot mapping contains a duplicate joint")
            selected[key] = index
    if not selected:
        return None
    environments = sorted({env for env, _ in selected})
    if environments != list(range(len(environments))):
        raise ValueError("Native robot mapping has missing environments")
    for model_name in ("stock", "eka_v6"):
        names = get_robot_model(model_name).joint_names
        expected = {(env, name) for env in environments for name in names}
        if set(selected) == expected:
            return [selected[env, name] for env in environments for name in names]
    raise ValueError("Native robot joint mapping does not match a complete registered topology")
