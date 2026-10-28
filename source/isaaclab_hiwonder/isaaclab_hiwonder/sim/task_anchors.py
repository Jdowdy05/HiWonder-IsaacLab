# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""World-fixed mechanism anchors that compose through room and support transforms."""


def bind_task_world_anchors(stage, task_path: str) -> list[str]:
    """Use the non-rigid task Xform as each world joint's static reference frame.

    An empty body0 relationship interprets localPos0/localRot0 in world space;
    those values cannot be authored once in a portable task-local USD. Targeting
    a non-rigid Xform keeps body0 static/world in UsdPhysics while the parser
    composes its frame through table height, room translation and rotation.
    No kinematic body, new constraint, or change to the mechanism is introduced.
    """
    from pxr import Usd, UsdPhysics

    task = stage.GetPrimAtPath(task_path)
    if not task or task.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Task anchor frame must be a valid non-rigid transform")
    bound = []
    for prim in Usd.PrimRange(task):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        targets = joint.GetBody0Rel().GetTargets()
        if targets and targets != [task.GetPath()]:
            continue
        child = joint.GetBody1Rel().GetTargets()
        if len(child) != 1 or not str(child[0]).startswith(task_path + "/"):
            raise ValueError(f"World-fixed task joint has an unexpected body1: {prim.GetPath()}")
        joint.CreateBody0Rel().SetTargets([task.GetPath()])
        bound.append(str(prim.GetPath()))
    return bound
