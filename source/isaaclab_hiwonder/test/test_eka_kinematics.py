# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Independent finger motion, wrist-fixed camera and actual moving pad TCP."""

import numpy as np
import pytest

from isaaclab_hiwonder.assets.kinematics import forward_kinematics
from isaaclab_hiwonder.assets.robot_models import get_robot_model


def test_eka_preserves_arm_and_camera_when_only_fingers_move():
    q = np.array([0.1, -0.3, 0.2, 0.15, -0.2, 0, 0, 0, 0])
    initial = forward_kinematics(q, robot_model="eka_v6")
    moved = forward_kinematics(q + [0, 0, 0, 0, 0, 0.12, 0.2, 0, 0], robot_model="eka_v6")
    stock = forward_kinematics(np.r_[q[:5], 0])
    for name in ("base", "shoulder", "upper_arm", "forearm", "wrist", "gripper_body", "camera_opencv", "camera_opengl"):
        np.testing.assert_allclose(initial[name], stock[name], atol=1e-12, rtol=0)
        np.testing.assert_allclose(initial[name], moved[name], atol=1e-12, rtol=0)
    for name in ("eka_right_proximal", "eka_right_tip"):
        np.testing.assert_array_equal(initial[name], moved[name])
    assert not np.allclose(initial["eka_left_proximal"], moved["eka_left_proximal"])
    assert not np.allclose(initial["eka_left_tip"], moved["eka_left_tip"])


def test_tip_actuation_moves_only_its_tip_and_tcp_is_the_actual_pad_midpoint():
    model = get_robot_model("eka_v6")
    q = np.zeros(9)
    before = forward_kinematics(q, robot_model=model)
    q[6] = 0.3
    after = forward_kinematics(q, robot_model=model)
    for name in model.links:
        if name != "eka_left_tip":
            np.testing.assert_array_equal(before[name], after[name])
    pads = [
        after[name][:3, :3] @ np.asarray(local) + after[name][:3, 3]
        for name, local in model.finger_pad_offsets_m.items()
    ]
    np.testing.assert_allclose(after["tool_center"][:3, 3], np.mean(pads, axis=0), atol=1e-12)
    assert not np.allclose(before["tool_center"][:3, 3], after["tool_center"][:3, 3])
    scaled = forward_kinematics(q, geometry_scale=1.01, robot_model=model)
    for name, pose in after.items():
        np.testing.assert_allclose(scaled[name][:3, 3], pose[:3, 3] * 1.01, atol=1e-12)
        np.testing.assert_allclose(scaled[name][:3, :3], pose[:3, :3], atol=1e-12)


def test_model_dimension_mismatch_is_not_silently_truncated():
    with pytest.raises(ValueError, match="joint"):
        forward_kinematics(np.zeros(6), robot_model="eka_v6")
    with pytest.raises(ValueError, match="joint"):
        forward_kinematics(np.zeros(9), robot_model="stock")
