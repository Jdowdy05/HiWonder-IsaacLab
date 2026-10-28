# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CAD-registered forward kinematics for reset generation and frame inspection."""

from __future__ import annotations

import numpy as np

from .robot_models import get_robot_model


def forward_kinematics(joint_positions_rad, geometry_scale: float = 1.0, robot_model="stock") -> dict[str, np.ndarray]:
    """Return support-frame-from-link matrices, with the base at the support origin.

    Joint order comes from the explicit model descriptor. The all-zero pose is
    the corresponding supplied CAD assembly. This geometric model
    does not enforce contact, servo travel or reachability of intermediate motions.
    """
    model = get_robot_model(robot_model)
    q = np.asarray(joint_positions_rad, dtype=float)
    if (
        q.shape != (model.num_joints,)
        or not np.isfinite(q).all()
        or not np.isfinite(geometry_scale)
        or geometry_scale <= 0
    ):
        raise ValueError(f"Need {model.num_joints} finite joint positions and a positive geometry scale")
    transforms = {"base": np.eye(4)}
    for joint, angle in zip(model.joints, q, strict=True):
        parent_origin = np.asarray(model.links[joint["parent"]]["origin_mm"])
        child_origin = np.asarray(model.links[joint["child"]]["origin_mm"])
        axis = np.asarray(joint["axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
        relative = np.eye(4)
        relative[:3, :3] = rotation
        relative[:3, 3] = (child_origin - parent_origin) * 0.001 * geometry_scale
        transforms[joint["child"]] = transforms[joint["parent"]] @ relative
    camera = np.eye(4)
    # Camera OpenCV axes: +x right, +y down, +z forward; CAD estimate Rx(160deg).
    angle = np.deg2rad(160)
    camera[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    camera[:3, 3] = np.asarray(model.camera["optical_translation_m"]) * geometry_scale
    transforms["camera_opencv"] = transforms[model.camera["optical_parent_link"]] @ camera
    transforms["camera_opengl"] = transforms["camera_opencv"] @ np.diag([1, -1, -1, 1])
    if model.finger_pad_offsets_m:
        tool = transforms["gripper_body"].copy()
        tool[:3, 3] = np.mean(
            [
                transforms[name][:3, :3] @ (np.asarray(offset) * geometry_scale) + transforms[name][:3, 3]
                for name, offset in model.finger_pad_offsets_m.items()
            ],
            axis=0,
        )
        transforms["tool_center"] = tool
    else:
        tool = np.eye(4)
        tool[:3, 3] = np.asarray(model.tool_center_m) * geometry_scale
        transforms["tool_center"] = transforms["gripper_body"] @ tool
    return transforms
