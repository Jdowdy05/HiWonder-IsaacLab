# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Validate authored USD topology and registration without claiming physics execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def validate_asset(path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    artifact = next(a for a in manifest["artifacts"] if a["file"] == path.name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("USD hash does not match conversion manifest")
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetDefaultPrim() or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("USD must have a valid default prim and meter units")
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(bodies) != 7 or len(joints) != 6:
        raise ValueError(f"Expected seven bodies and six actuated joints, got {len(bodies)}, {len(joints)}")
    cache = UsdGeom.XformCache()
    frame_errors = []
    edges = {}
    for joint in joints:
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        if child.GetPath() in edges:
            raise ValueError("A body has more than one parent")
        edges[child.GetPath()] = parent.GetPath()
        positions, axes = [], []
        for body, pos, rot in [
            (parent, joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            (child, joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
        ]:
            world = cache.GetLocalToWorldTransform(body)
            positions.append(np.array(world.Transform(Gf.Vec3d(pos))))
            axis = Gf.Rotation(Gf.Quatd(rot)).TransformDir(Gf.Vec3d(1, 0, 0))
            axes.append(np.array(world.TransformDir(axis)))
        frame_errors.append(float(np.linalg.norm(positions[0] - positions[1])))
        if frame_errors[-1] > 1e-7 or np.linalg.norm(axes[0] - axes[1]) > 1e-6:
            raise ValueError(f"Disconnected joint frame: {joint.GetPath()}")
        if joint.GetLowerLimitAttr().Get() >= joint.GetUpperLimitAttr().Get():
            raise ValueError("Invalid travel limits")
    for body in bodies:
        mass = UsdPhysics.MassAPI(body)
        if mass.GetMassAttr().Get() <= 0 or min(mass.GetDiagonalInertiaAttr().Get()) <= 0:
            raise ValueError("Nonphysical nonpositive mass or inertia")
        seen, current = set(), body.GetPath()
        while current in edges:
            if current in seen:
                raise ValueError("Cyclic articulation")
            seen.add(current)
            current = edges[current]
        if str(current) != "/HiWonder/base":
            raise ValueError("Articulation does not connect to the base")
    scale = artifact["scale"]
    camera = stage.GetPrimAtPath("/HiWonder/gripper_body/camera_mount/Camera")
    world = cache.GetLocalToWorldTransform(camera)
    lens_world = np.array(world.ExtractTranslation())
    expected_lens = np.array([-0.00200012, -0.16390936, 0.32012591 - 0.00039]) * scale
    lens_error = float(np.linalg.norm(lens_world - expected_lens))
    direction = np.array(world.TransformDir(Gf.Vec3d(0, 0, -1)))
    if lens_error > 1e-7 or np.linalg.norm(direction - [0, -np.sin(np.pi / 9), -np.cos(np.pi / 9)]) > 1e-6:
        raise ValueError("Camera position/direction disagrees with mechanical CAD lens registration")
    camera_schema = UsdGeom.Camera(camera)
    fx = (
        manifest["camera"]["width"]
        * camera_schema.GetFocalLengthAttr().Get()
        / camera_schema.GetHorizontalApertureAttr().Get()
    )
    if abs(fx - manifest["camera"]["intrinsics"][0]) > 1e-3:
        raise ValueError("USD projection disagrees with preview intrinsics")
    object_map = {o["index"]: o for o in manifest["objects"]}
    bounds_errors = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
            continue
        index = int(prim.GetName().split("_")[1])
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        transform = cache.GetLocalToWorldTransform(prim)
        world_points = np.array([transform.Transform(Gf.Vec3d(*map(float, p))) for p in points])
        reconstructed = world_points / scale * 1000 + [0, 0, 0.39]
        original = np.asarray(object_map[index]["source_bounds_mm"])
        bounds_errors.append(float(np.max(np.abs(np.array([reconstructed.min(0), reconstructed.max(0)]) - original))))
    # Bounds are a coarse decimation guard, not a surface-distance proof.
    if max(bounds_errors) > 2.0:
        raise ValueError("Decimation or registration displaced a component bound by >2mm")
    return dict(
        status="PASS_USD_REGISTRATION",
        physics_executed=False,
        rigid_bodies=len(bodies),
        revolute_joints=len(joints),
        maximum_joint_frame_error_m=max(frame_errors),
        camera_lens_error_m=lens_error,
        camera_forward_world=direction.tolist(),
        preview_fx_px=fx,
        maximum_component_bound_change_mm=max(bounds_errors),
        collision_meshes=sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()),
        scale=scale,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_asset(args.asset, args.manifest or args.asset.parent / "manifest.json")
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
