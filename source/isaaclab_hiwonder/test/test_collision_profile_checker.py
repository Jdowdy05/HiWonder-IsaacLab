# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Checker model selection and real protected-hand certificates without collision queries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from isaaclab_hiwonder.sim.robot_collisions import load_collision_profile

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("collision_checker", ROOT / "tools/check_hiwonder_collision_profile.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_sample_poses_sweep_each_selected_joint_without_changing_stock_protocol():
    stock = checker.sample_joint_poses("stock", 3)
    eka = np.asarray(checker.sample_joint_poses("eka_v6", 3))
    assert np.asarray(stock).shape == (130, 6)
    assert eka.shape == (193, 9)
    np.testing.assert_array_equal(eka[0], np.zeros(9))
    # Left root and left distal are independent sweep dimensions after five arm joints.
    assert eka[1 + 5 * 21, 5] == pytest.approx(np.deg2rad(-25))
    assert eka[1 + 6 * 21 + 20, 6] == pytest.approx(np.deg2rad(60))
    assert eka[1 + 8 * 21 + 20, 8] == pytest.approx(np.deg2rad(60))
    assert np.count_nonzero(eka[1 + 6 * 21 + 20]) == 1
    np.testing.assert_array_equal(eka[-3:], np.asarray(checker.sample_joint_poses("eka_v6", 3))[-3:])


@pytest.fixture(scope="module")
def eka_profile():
    return load_collision_profile(model="eka_v6")


def test_actual_nominal_eka_certificate_protects_all_five_hand_links(eka_profile):
    original, proof = checker.reference_and_certificate(eka_profile, "nominal", model="eka_v6")
    protected = {entry["link"]: entry for entry in proof if entry["protected"]}
    assert set(protected) == {
        "gripper_body",
        "eka_left_proximal",
        "eka_left_tip",
        "eka_right_proximal",
        "eka_right_tip",
    }
    assert all(row["subset_certified"] and row["maximum_body_loss_m"] == 0 for row in protected.values())
    original_shapes = original["variants"]["nominal"]["shapes"]
    assert all(any(shape["link"] == link for shape in original_shapes) for link in protected)


def test_legacy_stock_certificate_retains_both_original_jaws():
    data = load_collision_profile()
    _, proof = checker.reference_and_certificate(data, "nominal")
    protected = [entry for entry in proof if entry["protected"]]
    assert {entry["link"] for entry in protected} == {"gripper_body", "moving_jaw"}
    assert all(entry["subset_certified"] and entry["maximum_body_loss_m"] == 0 for entry in protected)


@pytest.mark.parametrize("damage", ("missing", "unprotected"))
def test_missing_or_unprotected_finger_group_cannot_pass_a_partial_certificate(eka_profile, damage):
    data = dict(eka_profile)
    variants = dict(data["variants"])
    row = dict(variants["nominal"])
    groups = [dict(group) for group in row["groups"]]
    if damage == "missing":
        groups = [group for group in groups if group["link"] != "eka_left_tip"]
    else:
        next(group for group in groups if group["link"] == "eka_left_tip")["protected"] = False
    row["groups"] = groups
    variants["nominal"] = row
    data["variants"] = variants
    with pytest.raises(ValueError, match="[Pp]rotected|hand|gripper"):
        checker.reference_and_certificate(data, "nominal", model="eka_v6")
