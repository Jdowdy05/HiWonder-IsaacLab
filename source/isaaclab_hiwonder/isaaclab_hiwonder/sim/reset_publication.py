# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-only publication of measured PhysX rigid poses to Fabric, without integration."""

import numpy as np
import warp as wp

from ..tasks.api import rotation_matrix


def publish_reset_body_poses(env, env_ids):
    """Publish actual body/link poses; fixed visual and camera children keep their local frames."""
    import usdrt
    from pxr import UsdGeom, UsdUtils
    from usdrt import hierarchy

    if not env.sim.cfg.use_fabric:
        raise RuntimeError("PhysX reset publication requires Fabric")
    ids = env_ids.detach().cpu().tolist()
    stage = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().GetId(env.sim.stage).ToLongInt())
    graph = hierarchy.IFabricHierarchy().get_fabric_hierarchy(stage.GetFabricId(), stage.GetStageIdAsStageId())
    records = []
    robot = env.robot.data.body_link_pose_w.torch
    for i in ids:
        prefix = f"/World/envs/env_{i}/Room"
        for j, name in enumerate(env.robot.body_names):
            records.append((f"{prefix}/Robot/{name}", robot[i, j]))
        for name, key in env.cfg.hiwonder_metadata["task_body_assets"].items():
            records.append((f"{prefix}/Task/{name}", env.scene[key].data.root_link_pose_w.torch[i]))
        for _, (asset, bodies, _) in env.runtime.articulation_maps.items():
            for name, j in bodies.items():
                records.append((f"{prefix}/Task/{name}", asset.data.body_link_pose_w.torch[i, j]))
        for category, group in (("distractor", "Distractors/Object"), ("confetti", "PhysicalConfetti/Piece")):
            for index, key in env.cfg.hiwonder_metadata.get(f"{category}_assets", {}).items():
                records.append((f"{prefix}/{group}{index}", env.scene[key].data.root_link_pose_w.torch[i]))
        records.append((f"{prefix}/Tabletop", env.scene["table"].data.root_link_pose_w.torch[i]))
    cache = UsdGeom.XformCache()
    graph.update_world_xforms()
    tracking = graph.tracking_world_xform_changes
    graph.track_world_xform_changes(True)
    try:
        for path, pose in records:
            value = pose.detach().cpu().numpy()
            authored = np.asarray(cache.GetLocalToWorldTransform(env.sim.stage.GetPrimAtPath(path)))
            scale = np.linalg.norm(authored[:3, :3], axis=1)
            matrix = np.eye(4)
            matrix[:3, :3] = scale[:, None] * rotation_matrix(value[3:]).T
            matrix[3, :3] = value[:3]
            # Only Fabric is written. Hierarchy tracking derives matching local
            # matrices and propagates the unchanged fixed camera/visual frames.
            xform = usdrt.Rt.Xformable(stage.GetPrimAtPath(path))
            xform.CreateFabricHierarchyWorldMatrixAttr().Set(usdrt.Gf.Matrix4d(*matrix.reshape(-1).tolist()))
        graph.update_world_xforms()
        if not graph.update_world_xforms_gpu(False):
            raise RuntimeError("PhysX reset Fabric hierarchy publication failed")
        wp.synchronize_device(env.device)
    finally:
        graph.track_world_xform_changes(tracking)
