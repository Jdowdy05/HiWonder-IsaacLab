# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Mechanism anchors must compose through room translation, rotation and table height."""

import numpy as np
from isaaclab_hiwonder.config import HiWonderEnvConfig
from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.sim.scene_config import build_scene
from isaaclab_hiwonder.tasks.catalog import build_task

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def test_task_world_joint_frames_follow_translated_rotated_room_instances(tmp_path):
    profile = HiWonderEnvConfig(num_envs=2)
    _, metadata = build_scene(
        profile, [build_task("knob") for _ in range(2)], RandomizationManager(2).snapshot(), tmp_path
    )
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    for i, (translation, yaw) in enumerate([((2, 3, 0.2), 35), ((-3, 1, 0.4), -60)]):
        room = UsdGeom.Xform.Define(stage, f"/World/env_{i}/Room")
        room.GetPrim().GetReferences().AddReference(metadata["room_files"][i])
        room.AddTranslateOp().Set(Gf.Vec3d(*translation))
        room.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
    parsed = UsdPhysics.UsdPhysicsLoadStageFromPrimRange(stage, [Sdf.Path("/World")])
    bodies = {str(body.primPath): body for body in parsed[UsdPhysics.ObjectType.RigidBody][1]}
    joints = [joint for joint in parsed[UsdPhysics.ObjectType.FixedJoint][1] if "/Task/" in str(joint.primPath)]
    assert len(joints) == 2
    for joint in joints:
        assert joint.isValid and not joint.body0
        body = bodies[str(joint.body1)]
        np.testing.assert_allclose(joint.localPose0Position, body.position, atol=3e-7)
        first = joint.localPose0Orientation
        second = body.rotation
        q0 = np.array([*first.GetImaginary(), first.GetReal()])
        q1 = np.array([*second.GetImaginary(), second.GetReal()])
        assert abs(np.dot(q0, q1)) > 1 - 1e-6
