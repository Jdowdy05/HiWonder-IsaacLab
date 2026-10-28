# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Saved model contracts: unchanged arm, four independent hinges, retained optics."""

import hashlib
import json

import numpy as np
import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_hiwonder.assets.robot_models import get_robot_model


@pytest.mark.parametrize("variant", ("nominal", "scale099", "scale101"))
def test_saved_eka_retains_arm_and_has_the_declared_kinematic_tree(variant):
    model, stock = get_robot_model("eka_v6"), get_robot_model("stock")
    assert model.asset_path(variant).is_file(), "Eka USD must be generated and saved"
    stage, original = Usd.Stage.Open(str(model.asset_path(variant))), Usd.Stage.Open(str(stock.asset_path(variant)))
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 10
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    assert set(joints) == set(model.joint_names)
    assert not stage.GetPrimAtPath("/HiWonder/moving_jaw")
    assert not stage.GetPrimAtPath("/HiWonder/joints/gripper")
    for body in ("base", "shoulder", "upper_arm", "forearm", "wrist"):
        a, b = stage.GetPrimAtPath("/HiWonder/" + body), original.GetPrimAtPath("/HiWonder/" + body)
        assert {str(x.GetName()): repr(x.Get()) for x in a.GetAttributes()} == {
            str(x.GetName()): repr(x.Get()) for x in b.GetAttributes()
        }
        for old in Usd.PrimRange(b):
            new = stage.GetPrimAtPath(old.GetPath())
            assert {str(x.GetName()): repr(x.Get()) for x in old.GetAttributes()} == {
                str(x.GetName()): repr(x.Get()) for x in new.GetAttributes()
            }
    for name in model.joint_names[:5]:
        a, b = joints[name], original.GetPrimAtPath("/HiWonder/joints/" + name)
        assert {str(x.GetName()): repr(x.Get()) for x in a.GetAttributes()} == {
            str(x.GetName()): repr(x.Get()) for x in b.GetAttributes()
        }
    for name in model.joint_names[5:]:
        joint = UsdPhysics.RevoluteJoint(joints[name])
        assert joint.GetCollisionEnabledAttr().Get() is False
        assert joint.GetBody0Rel().GetTargets() != joint.GetBody1Rel().GetTargets()
    for relative in ("gripper_body/camera_mount", model.camera_path):
        a, b = stage.GetPrimAtPath("/HiWonder/" + relative), original.GetPrimAtPath("/HiWonder/" + relative)
        assert {str(x.GetName()): repr(x.Get()) for x in a.GetAttributes()} == {
            str(x.GetName()): repr(x.Get()) for x in b.GetAttributes()
        }
    points = [np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert points and all(np.isfinite(p).all() for p in points)


def test_eka_source_provenance_and_all_scaled_artifact_hashes_are_real():
    model = get_robot_model("eka_v6")
    data = json.loads(model.manifest_path.read_text())
    assert data["source"]["sha256"] == hashlib.sha256((model.asset_directory / "source.step").read_bytes()).hexdigest()
    assert data["source"]["printable_solids"] == 15
    assert data["source"]["raw_invalid_brep_parts"], "Input CAD defects must remain disclosed"
    assert len(data["artifacts"]) == 3
    for artifact in data["artifacts"]:
        path = model.asset_directory / artifact["file"]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
