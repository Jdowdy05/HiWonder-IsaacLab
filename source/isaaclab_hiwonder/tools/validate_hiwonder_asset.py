# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Validate authored USD topology and registration without claiming physics execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_hiwonder.assets.robot_models import get_robot_model
from isaaclab_hiwonder.tasks.api import rotation_matrix


def _component_bounds_errors(stage, cache, artifact, manifest, model, scale):
    """Compare every visual against its retained-CAD or supplied Eka mesh reference."""
    base_origin = np.asarray(model.links["base"]["origin_mm"])
    with ExitStack() as stack:
        source_manifest = (
            manifest if model.name == "stock" else json.loads(get_robot_model("stock").manifest_path.read_text())
        )
        object_map = {o["index"]: o for o in source_manifest["objects"]}
        bounds_errors = []
        source_archive = precision = None
        if model.name == "eka_v6":
            source_archive = stack.enter_context(
                np.load(model.asset_directory / "source_geometry.npz", allow_pickle=False)
            )
            precision = stack.enter_context(np.load(model.asset_directory / "precision_caps.npz", allow_pickle=False))
            part_map = {part["visual"]: part for part in artifact["parts"]}
            checked_parts = set()
            variant = next(v for v in model.geometry_variants if model.asset_path(v).name == artifact["file"])
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
                continue
            index = int(prim.GetName().split("_")[1])
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            if not np.isfinite(points).all():
                raise ValueError("Visual mesh contains nonfinite geometry")
            transform = cache.GetLocalToWorldTransform(prim)
            world_points = np.array([transform.Transform(Gf.Vec3d(*map(float, p))) for p in points])
            if prim.GetName().startswith("eka_"):
                if model.name != "eka_v6" or str(prim.GetPath()) not in part_map:
                    raise ValueError("Unregistered Eka visual component")
                part = part_map[str(prim.GetPath())]
                checked_parts.add(str(prim.GetPath()))
                key = f"{variant}_v{index}"
                if key in precision:
                    expected = np.asarray(precision[key], float)
                    expected += (np.asarray(model.links[part["body"]]["origin_mm"]) - base_origin) * (0.001 * scale)
                else:
                    expected = (
                        np.asarray(source_archive[f"v{index}"], float) @ np.asarray(model.mount["rotation"]).T
                        + model.mount["translation_cad_mm"]
                        - base_origin
                    ) * (0.001 * scale)
                original = np.array([expected.min(0), expected.max(0)])
                bounds_errors.append(
                    float(np.max(np.abs(np.array([world_points.min(0), world_points.max(0)]) - original)))
                    * 1000
                    / scale
                )
            else:
                reconstructed = world_points / scale * 1000 + base_origin
                original = np.asarray(object_map[index]["source_bounds_mm"])
                bounds_errors.append(
                    float(np.max(np.abs(np.array([reconstructed.min(0), reconstructed.max(0)]) - original)))
                )
        if source_archive is not None:
            if checked_parts != part_map.keys():
                raise ValueError("A declared Eka visual component is missing")
        return bounds_errors


def validate_asset(path: Path, manifest_path: Path | None = None, robot_model="stock") -> dict:
    model = get_robot_model(robot_model)
    path = Path(path)
    manifest_path = Path(manifest_path or model.manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("robot_model", "stock") != model.name:
        raise ValueError("Conversion manifest belongs to a different robot model; select it explicitly")
    if "joint_order" in manifest and tuple(manifest["joint_order"]) != model.joint_names:
        raise ValueError("Conversion manifest joint order differs from selected robot model")
    artifacts = [a for a in manifest["artifacts"] if a["file"] == path.name]
    if len(artifacts) != 1 or path.name not in {model.asset_path(v).name for v in model.geometry_variants}:
        raise ValueError("Asset is not a unique selected-model artifact in the conversion manifest")
    artifact = artifacts[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("USD hash does not match conversion manifest")
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetDefaultPrim() or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("USD must have a valid default prim and meter units")
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if {body.GetName() for body in bodies} != set(model.links) or len(bodies) != model.num_links:
        raise ValueError(f"Expected {model.num_links} registered bodies for {model.name}")
    if {joint.GetPrim().GetName() for joint in joints} != set(model.joint_names) or len(joints) != model.num_joints:
        raise ValueError(f"Expected {model.num_joints} registered actuator joints for {model.name}")
    cache = UsdGeom.XformCache()
    frame_errors = []
    edges = {}
    joint_specs = {joint["name"]: joint for joint in model.joints}
    for joint in joints:
        spec = joint_specs[joint.GetPrim().GetName()]
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        if parent.GetName() != spec["parent"] or child.GetName() != spec["child"]:
            raise ValueError("Joint parent/child differs from selected robot topology")
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
        expected_axis = np.asarray(spec["axis"], dtype=float)
        expected_axis /= np.linalg.norm(expected_axis)
        if (
            frame_errors[-1] > 1e-7
            or np.linalg.norm(axes[0] - axes[1]) > 1e-6
            or np.linalg.norm(axes[0] - expected_axis) > 1e-6
        ):
            raise ValueError(f"Disconnected joint frame: {joint.GetPath()}")
        limits = np.array([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
        if (
            not np.isfinite(limits).all()
            or limits[0] >= limits[1]
            or not np.allclose(limits, spec["limits_deg"], atol=1e-5, rtol=0)
        ):
            raise ValueError("Invalid travel limits")
    for body in bodies:
        mass = UsdPhysics.MassAPI(body)
        physical = np.r_[mass.GetMassAttr().Get(), mass.GetDiagonalInertiaAttr().Get()]
        if not np.isfinite(physical).all() or np.any(physical <= 0):
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
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Artifact scale must be finite and positive")
    camera = stage.GetPrimAtPath("/HiWonder/" + model.camera_path)
    world = cache.GetLocalToWorldTransform(camera)
    lens_world = np.array(world.ExtractTranslation())
    camera_profile = model.camera
    parent_origin = np.asarray(model.links[camera_profile["optical_parent_link"]]["origin_mm"])
    base_origin = np.asarray(model.links["base"]["origin_mm"])
    expected_lens = (
        (parent_origin - base_origin) * 0.001 + np.asarray(camera_profile["optical_translation_m"])
    ) * scale
    lens_error = float(np.linalg.norm(lens_world - expected_lens))
    direction = np.array(world.TransformDir(Gf.Vec3d(0, 0, -1)))
    expected_direction = rotation_matrix(camera_profile["optical_quaternion_xyzw"])[:, 2]
    if lens_error > 1e-7 or np.linalg.norm(direction - expected_direction) > 1e-6:
        raise ValueError("Camera position/direction disagrees with mechanical CAD lens registration")
    camera_schema = UsdGeom.Camera(camera)
    fx = (
        camera_profile["width"]
        * camera_schema.GetFocalLengthAttr().Get()
        / camera_schema.GetHorizontalApertureAttr().Get()
    )
    if abs(fx - camera_profile["intrinsics"][0]) > 1e-3:
        raise ValueError("USD projection disagrees with preview intrinsics")
    bounds_errors = _component_bounds_errors(stage, cache, artifact, manifest, model, scale)
    # Bounds are a coarse decimation guard, not a surface-distance proof.
    if max(bounds_errors) > 2.0:
        raise ValueError("Decimation or registration displaced a component bound by >2mm")
    return dict(
        status="PASS_USD_REGISTRATION",
        physics_executed=False,
        robot_model=model.name,
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
    parser.add_argument("--robot-model", choices=("stock", "eka_v6"), default="stock")
    args = parser.parse_args()
    result = validate_asset(
        args.asset, args.manifest or args.asset.parent / "manifest.json", robot_model=args.robot_model
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
