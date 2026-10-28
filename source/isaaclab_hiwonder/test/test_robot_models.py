# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""A different hand must own its control, contact, frame and asset contracts."""

import numpy as np
import pytest

from isaaclab_hiwonder.assets import registration
from isaaclab_hiwonder.assets.robot_models import get_robot_model


def test_stock_descriptor_preserves_existing_registration_and_grasp_contract():
    stock = get_robot_model("stock")
    assert stock.joint_names == registration.JOINT_NAMES
    assert stock.links == registration.LINKS
    assert stock.joints == tuple(registration.JOINTS)
    assert stock.nominal_joint_positions_rad == registration.NOMINAL_JOINT_POSITIONS_RAD
    assert stock.camera_path == registration.CAMERA_PATH
    assert stock.asset_path("nominal") == registration.USD_PATH
    assert stock.contact_bodies == {"left_finger": ("gripper_body",), "right_finger": ("moving_jaw",)}
    assert stock.critic_width == 256


def test_eka_has_four_real_finger_axes_and_complete_hand_contact_ownership():
    stock, eka = get_robot_model("stock"), get_robot_model("eka_v6")
    assert len(eka.joint_names) == 9 and len(eka.links) == 10
    assert eka.joint_names[:5] == stock.joint_names[:5]
    assert eka.joints[:5] == stock.joints[:5]
    assert "gripper" not in eka.joint_names and "moving_jaw" not in eka.links
    assert eka.gripper_joint_indices == (5, 6, 7, 8)
    assert eka.contact_bodies["left_finger"] == ("eka_left_tip",)
    assert eka.contact_bodies["right_finger"] == ("eka_right_tip",)
    assert set(eka.contact_bodies["gripper"]) == set(eka.links) - set(
        stock.links.keys() - {"gripper_body", "moving_jaw"}
    )
    assert eka.critic_width >= 267
    for key, scale in eka.actuator_scales.items():
        assert len(scale) == 9
        np.testing.assert_array_equal(scale[:5], np.ones(5))
        assert np.isfinite(scale).all(), key
    assert eka.asset_path("nominal") != stock.asset_path("nominal")
    assert eka.collision_profile_path != stock.collision_profile_path


def test_eka_mount_matches_the_supplied_adapter_hole_pattern_to_the_original_wrist_horn():
    eka = get_robot_model("eka_v6")
    source = np.asarray(eka.mount["adapter_holes_mm"])
    rotation = np.asarray(eka.mount["rotation"])
    translation = np.asarray(eka.mount["translation_support_mm"])
    transformed = source @ rotation.T + translation
    expected = np.asarray(eka.mount["horn_holes_support_mm"])
    np.testing.assert_allclose(transformed, expected, atol=0.003, rtol=0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
    assert np.linalg.det(rotation) > 0.999999


def test_model_and_geometry_names_fail_explicitly():
    with pytest.raises(ValueError, match="robot model"):
        get_robot_model("missing")
    with pytest.raises(ValueError, match="geometry"):
        get_robot_model("eka_v6").asset_path("missing")
