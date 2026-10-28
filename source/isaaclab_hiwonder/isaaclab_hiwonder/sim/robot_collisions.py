# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch-time manipulation colliders; visual CAD and articulation stay intact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from ..assets.registration import ASSET_DIR
from ..assets.robot_models import get_robot_model

PROTECTED_LINKS = ("gripper_body", "moving_jaw")
PROFILE_PATH = ASSET_DIR / "generated" / "manipulation_collisions.json"


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_collision_profile(path=None, model=None):
    """Read and checksum one immutable recipe before creating a room cohort."""
    selected = get_robot_model(model or "stock")
    path = Path(path or selected.collision_profile_path)
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    digest = _digest(path)
    if manifest.get("schema_version") != 1 or manifest.get("file") != path.name or manifest.get("sha256") != digest:
        raise ValueError("Collision profile manifest or content checksum differs")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("name") != "manipulation":
        raise ValueError("Unsupported collision profile schema")
    if data.get("robot_model", "stock") != selected.name:
        raise ValueError("Collision profile belongs to a different robot model")
    if data.get("protected_links") != list(selected.protected_links):
        raise ValueError("Collision profile must protect every selected gripper link")
    loss = data.get("maximum_body_surface_loss_m")
    if type(loss) not in (int, float) or not np.isfinite(loss) or not 0 < loss <= 0.002:
        raise ValueError("Collision profile exceeds the 2mm body loss contract")
    for variant in data["variants"].values():
        source = path.parent / variant["asset_file"]
        reference = path.parent / variant["reference_file"]
        if (
            Path(variant["asset_file"]).name != variant["asset_file"]
            or Path(variant["reference_file"]).name != variant["reference_file"]
        ):
            raise ValueError("Collision source paths must remain inside generated assets")
        if _digest(source) != variant["asset_sha256"] or _digest(reference) != variant["reference_sha256"]:
            raise ValueError("Collision profile source asset/reference checksum differs")
        if not variant["shapes"]:
            raise ValueError("Collision profile has no structural shapes")
        for shape in variant["shapes"]:
            link, relative = shape["link"], shape["path"]
            if (
                link not in selected.links
                or not relative.startswith(link + "/manipulation_collisions/")
                or ".." in relative.split("/")
            ):
                raise ValueError("Invalid collision shape owner/path")
            points = np.asarray(shape["vertices"], dtype=np.float32)
            faces = np.asarray(shape["triangles"])
            transform, scale = np.asarray(shape["transform"]), np.asarray(shape["scale"])
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("Collision vertices must be finite 3D geometry")
            if (
                faces.ndim != 2
                or faces.shape[1] != 3
                or faces.dtype.kind not in "iu"
                or faces.min() < 0
                or faces.max() >= len(points)
            ):
                raise ValueError("Collision triangle indices are invalid")
            if (
                transform.shape != (7,)
                or scale.shape != (3,)
                or not np.isfinite(transform).all()
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError("Invalid collision frame/scale")
            if not np.isclose(np.linalg.norm(transform[3:]), 1, atol=1e-6, rtol=0):
                raise ValueError("Collision orientation must be normalized")
            if shape["kind"] not in ("reference", "inscribed") or (
                link in selected.protected_links and shape["kind"] != "reference"
            ):
                raise ValueError("Gripper geometry may not be simplified")
    return dict(data, sha256=digest)


def _self_collisions(stage, robot_path, enabled):
    if enabled is None:
        return
    if type(enabled) is not bool:
        raise ValueError("Robot self-collision setting must be bool or None")
    roots = [p for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path)) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise ValueError("Robot must have exactly one authored articulation root")
    for prim in roots:
        prim.AddAppliedSchema("PhysxArticulationAPI")
        prim.AddAppliedSchema("NewtonArticulationRootAPI")
        prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(enabled)
        prim.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(enabled)


def apply_robot_collision_profile(
    stage, robot_path, *, mode="manipulation", self_collisions=None, native=False, profile=None, model=None
):
    """Author only collision geometry; profile changes require a fresh scene."""
    selected = get_robot_model(model or "stock")
    root = stage.GetPrimAtPath(robot_path)
    if not root or root.GetCustomDataByKey("hiwonder:collisionProfile"):
        raise ValueError("Choose a collision profile on a fresh robot scene")
    if mode not in ("cad", "manipulation"):
        raise ValueError("Unknown robot collision profile")
    if (self_collisions is not None and type(self_collisions) is not bool) or type(native) is not bool:
        raise ValueError("Collision settings must be explicit booleans")
    if mode == "cad":
        _self_collisions(stage, robot_path, self_collisions)
        return dict(profile="cad", sha256=None, self_collisions=self_collisions)
    if not native:
        # Preserve the backend's own CAD cooker. Native convex references are
        # not a qualified replacement for PhysX cooking or its hull budget.
        _self_collisions(stage, robot_path, self_collisions)
        return dict(
            profile="cad",
            requested_profile=mode,
            sha256=None,
            self_collisions=self_collisions,
            reason="physx_cad_already_lower_complexity" if selected.name == "stock" else "physx_cad_backend_preserved",
        )
    data = profile or load_collision_profile(model=selected)
    if data.get("robot_model", "stock") != selected.name:
        raise ValueError("Collision profile belongs to a different robot model")
    source_value = root.GetCustomDataByKey("hiwonder:sourceAsset")
    source = Path(source_value) if source_value else None
    if source is None or not source.is_file():
        raise ValueError("Robot lacks a readable source asset")
    source_hash = _digest(source)
    matches = [(name, row) for name, row in data["variants"].items() if row["asset_sha256"] == source_hash]
    if len(matches) != 1:
        raise ValueError("No collision profile matches the exact source asset")
    name, variant = matches[0]
    active = variant["shapes"]
    source_paths = variant["structural_paths"]
    for relative in source_paths:
        prim = stage.GetPrimAtPath(robot_path + relative)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError("Collision source prim is absent")
    for entry in active:
        body = stage.GetPrimAtPath(robot_path + "/" + entry["link"])
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Collision target lacks a registered rigid body")
    for relative in source_paths:
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(robot_path + relative)).CreateCollisionEnabledAttr(False)
    for entry in active:
        mesh = UsdGeom.Mesh.Define(stage, robot_path + "/" + entry["path"])
        mesh.CreatePointsAttr(np.asarray(entry["vertices"], dtype=np.float32).tolist())
        faces = np.asarray(entry["triangles"], dtype=np.int32)
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateVisibilityAttr("invisible")
        mesh.CreateDoubleSidedAttr(True)
        transform = entry["transform"]
        mesh.AddTranslateOp().Set(Gf.Vec3d(*transform[:3]))
        mesh.AddOrientOp().Set(Gf.Quatf(transform[6], Gf.Vec3f(*transform[3:6])))
        mesh.AddScaleOp().Set(Gf.Vec3f(*entry["scale"]))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        mesh.GetPrim().AddAppliedSchema("NewtonMeshCollisionAPI")
        mesh.GetPrim().CreateAttribute("newton:maxHullVertices", Sdf.ValueTypeNames.Int).Set(
            max(64, len(entry["vertices"]))
        )
    _self_collisions(stage, robot_path, self_collisions)
    root.SetCustomDataByKey("hiwonder:collisionProfile", "manipulation")
    root.SetCustomDataByKey("hiwonder:collisionProfileSha256", data["sha256"])
    root.SetCustomDataByKey("hiwonder:nativeCollisionProfile", bool(native))
    return dict(
        profile="manipulation",
        sha256=data["sha256"],
        variant=name,
        self_collisions=self_collisions,
        reference_structural_shapes=variant["reference_hulls"],
        new_structural_shapes=len(active),
        protected_shapes=sum(s["link"] in selected.protected_links for s in active),
        body_surface_loss_m=data["maximum_body_surface_loss_m"],
    )


def restore_profile_hulls(builder, stage, robot_path):
    """Restore authored hull topology after convex reimport, before SDF cooking.

    Some importers re-hull/reorder coplanar faces. Preserving the authored
    gripper vertices, triangles and rigid frame requires restoring those exact
    mesh sources, just as the original full-CAD cache path does.
    """
    import newton

    from ..tasks.api import rotation_matrix

    root = stage.GetPrimAtPath(robot_path)
    if root.GetCustomDataByKey("hiwonder:collisionProfile") != "manipulation" or not root.GetCustomDataByKey(
        "hiwonder:nativeCollisionProfile"
    ):
        raise ValueError("Native restoration requires an authored native manipulation profile")
    inertials = {name: np.asarray(getattr(builder, name)).copy() for name in ("body_mass", "body_com", "body_inertia")}
    labels = {}
    for index, label in enumerate(builder.shape_label):
        labels.setdefault(label, []).append(index)
    xforms = UsdGeom.XformCache()
    restored, maximum_error = 0, 0.0
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if "/manipulation_collisions/" not in path or not prim.IsA(UsdGeom.Mesh):
            continue
        active = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if not active:
            continue
        matches = labels.get(path, ())
        if len(matches) != 1:
            raise ValueError("Authored collision hull missing or duplicated after import: " + path)
        index = matches[0]
        mesh = builder.shape_source[index]
        if builder.shape_type[index] != newton.GeoType.CONVEX_MESH or mesh is None or mesh.sdf is not None:
            raise ValueError("Profile restoration requires a convex mesh before SDF creation")
        if int(builder.shape_flags[index]) & 6 != 6:
            raise ValueError("Profile hull lost rigid/particle collision flags")
        usd_mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
        counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get())
        triangles = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if not (counts == 3).all() or len(triangles) != 3 * len(counts):
            raise ValueError("Profile collision topology must be triangular")
        body_path = robot_path + "/" + path[len(robot_path) + 1 :].split("/")[0]
        if builder.body_label[builder.shape_body[index]] != body_path:
            raise ValueError("Profile hull imported under the wrong rigid body")
        frame = xforms.ComputeRelativeTransform(prim, stage.GetPrimAtPath(body_path))[0]
        expected = np.array([frame.Transform(Gf.Vec3d(*map(float, vertex))) for vertex in vertices])
        transform, scale = np.asarray(builder.shape_transform[index]), np.asarray(builder.shape_scale[index])
        measured = (vertices * scale) @ rotation_matrix(transform[3:]).T + transform[:3]
        error = float(np.abs(expected - measured).max())
        maximum_error = max(maximum_error, error)
        if error > 1e-6:
            raise ValueError("Profile hull frame changed during import")
        replacement = mesh.copy(vertices=vertices, indices=triangles, recompute_inertia=False)
        replacement.maxhullvert = max(64, len(vertices))
        builder.shape_source[index] = replacement
        restored += 1
    if not restored:
        raise ValueError("Native manipulation profile contains no active collision hulls")
    if any(not np.array_equal(value, np.asarray(getattr(builder, name))) for name, value in inertials.items()):
        raise ValueError("Profile restoration changed physical inertials")
    return dict(
        status="PASS_COLLISION_PROFILE_REIMPORT",
        mesh_sources_restored_before_sdf=restored,
        maximum_frame_error_m=maximum_error,
        mass_inertia_unchanged=True,
        profile_sha256=root.GetCustomDataByKey("hiwonder:collisionProfileSha256"),
    )
