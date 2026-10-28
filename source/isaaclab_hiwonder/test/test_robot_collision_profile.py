# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""USD boundary: simplified bodies must retain grasp surfaces and mechanics."""

import numpy as np
import pytest
from isaaclab_hiwonder.assets.registration import ASSET_DIR, JOINT_NAMES, LINKS
from isaaclab_hiwonder.config import HiWonderEnvConfig
from isaaclab_hiwonder.sim.robot_collisions import apply_robot_collision_profile, load_collision_profile

from pxr import Usd, UsdGeom


def stage_for(variant="nominal"):
    stage = Usd.Stage.CreateInMemory()
    suffix = "" if variant == "nominal" else "_" + variant
    source = ASSET_DIR / "generated" / f"hiwonder_soarm101{suffix}.usdc"
    robot = stage.DefinePrim("/Robot", "Xform")
    robot.GetReferences().AddReference(str(source))
    robot.SetCustomDataByKey("hiwonder:sourceAsset", str(source))
    return stage


def physical_signature(stage):
    paths = ["/Robot/" + link for link in LINKS]
    paths += ["/Robot/joints/" + joint for joint in JOINT_NAMES]
    paths += ["/Robot/gripper_body/camera_mount", "/Robot/gripper_body/camera_mount/Camera"]
    return {
        path: {str(a.GetName()): repr(a.Get()) for a in stage.GetPrimAtPath(path).GetAttributes()} for path in paths
    }


def grip_signature(stage):
    return {
        str(p.GetPath()): {str(a.GetName()): repr(a.Get()) for a in p.GetAttributes()}
        for p in Usd.PrimRange(stage.GetPrimAtPath("/Robot"))
        if "/collisions/" in str(p.GetPath()) and str(p.GetPath()).split("/")[2] in ("gripper_body", "moving_jaw")
    }


def test_profile_default_and_invalid_settings_have_an_explicit_contract():
    profile = HiWonderEnvConfig()
    assert profile.robot_collision_profile == "manipulation"
    assert profile.robot_self_collisions is None
    assert HiWonderEnvConfig(robot_collision_profile="cad", robot_self_collisions=True).robot_self_collisions
    with pytest.raises(ValueError, match="collision"):
        HiWonderEnvConfig(robot_collision_profile="missing")
    with pytest.raises(ValueError, match="collision"):
        HiWonderEnvConfig(robot_self_collisions="yes")


@pytest.mark.parametrize("variant", ("nominal", "scale099", "scale101"))
def test_physx_profile_preserves_original_gripper_and_all_registered_mechanics(variant):
    stage = stage_for(variant)
    before, before_grip = physical_signature(stage), grip_signature(stage)
    receipt = apply_robot_collision_profile(stage, "/Robot", native=False)
    assert receipt["profile"] == "cad" and receipt["requested_profile"] == "manipulation"
    assert receipt["reason"] == "physx_cad_already_lower_complexity"
    assert physical_signature(stage) == before
    assert grip_signature(stage) == before_grip
    assert not any("manipulation_collisions" in str(p.GetPath()) for p in stage.Traverse())


def test_cad_selection_is_a_reversible_noop_and_self_collision_is_not_blanket_filtered():
    stage = stage_for()
    before = physical_signature(stage)
    receipt = apply_robot_collision_profile(stage, "/Robot", mode="cad")
    assert receipt["profile"] == "cad"
    assert physical_signature(stage) == before
    assert not any("manipulation_collisions" in str(p.GetPath()) for p in stage.Traverse())
    apply_robot_collision_profile(stage, "/Robot", mode="cad", self_collisions=True)
    fixed = stage.GetPrimAtPath("/Robot/joints/fixed_base")
    assert fixed.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert fixed.GetAttribute("newton:selfCollisionEnabled").Get() is True
    for name in JOINT_NAMES:
        joint = stage.GetPrimAtPath("/Robot/joints/" + name)
        assert joint.GetAttribute("physics:collisionEnabled").Get() is False
    assert not any(p.GetRelationship("physics:filteredPairs").HasAuthoredTargets() for p in stage.Traverse())


def test_native_gripper_recipe_is_unmodified_reference_hull_geometry():
    stage = stage_for()
    profile = load_collision_profile()
    before = physical_signature(stage)
    receipt = apply_robot_collision_profile(stage, "/Robot", native=True, profile=profile)
    assert physical_signature(stage) == before
    entries = profile["variants"]["nominal"]["shapes"]
    protected = [e for e in entries if e["link"] in ("gripper_body", "moving_jaw")]
    assert protected and all(e["kind"] == "reference" for e in protected)
    for entry in protected:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Robot/" + entry["path"]))
        np.testing.assert_array_equal(
            np.asarray(mesh.GetPointsAttr().Get()), np.asarray(entry["vertices"], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), np.asarray(entry["triangles"]).ravel()
        )
    assert receipt["protected_shapes"] == len(protected)


def test_missing_and_changed_source_assets_fail_before_authoring(tmp_path):
    profile = load_collision_profile()
    stage = stage_for()
    stage.GetPrimAtPath("/Robot").SetCustomDataByKey("hiwonder:sourceAsset", str(tmp_path / "missing.usdc"))
    with pytest.raises(ValueError, match="source|asset"):
        apply_robot_collision_profile(stage, "/Robot", native=True, profile=profile)
