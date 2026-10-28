# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Optional persistence of exact Newton-generated robot convex hulls.

Capture runs through MODEL_INIT after the normal USD importer decomposes CAD.
Reuse authors those hull vertices, body-local transforms and scales separately.
The callback validates a reused cache against the newly imported builder before
physics allocation. No Newton or CoACD implementation is patched.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import numpy as np


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _serialize(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def default_cache_directory() -> Path:
    """Stable local cache outside scene-output directories; override through the environment."""
    override = os.environ.get("HIWONDER_COLLISION_CACHE_DIR")
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return Path(override) if override else root / "IsaacLab" / "HiWonder" / "newton_hulls"


def _identity(robot_usd: Path, builder=None) -> dict:
    import newton

    settings = (
        newton.ModelBuilder.MeshApproximationConfig() if builder is None else builder.default_mesh_approximation_cfg
    )
    return dict(
        schema=1,
        source_robot_sha256=_digest(robot_usd.read_bytes()),
        newton_version=importlib.metadata.version("newton"),
        coacd_version=importlib.metadata.version("coacd"),
        builder_implementation_sha256=_digest(Path(inspect.getfile(newton.ModelBuilder)).read_bytes()),
        importer_implementation_sha256=_digest(
            (Path(inspect.getfile(newton.ModelBuilder)).parents[1] / "utils" / "import_usd.py").read_bytes()
        ),
        settings=dict(
            threshold=settings.coacd_threshold,
            mcts_nodes=20,
            mcts_iterations=5,
            mcts_max_depth=1,
            merge=False,
            max_convex_hull=newton.Mesh.MAX_HULL_VERTICES,
        ),
    )


def _source(stage, robot_path: str) -> Path:
    source = stage.GetPrimAtPath(robot_path).GetCustomDataByKey("hiwonder:sourceAsset")
    if not source or not Path(source).is_file():
        raise ValueError("Robot lacks a readable hiwonder:sourceAsset provenance path")
    return Path(source)


def _cache_path(identity, directory) -> Path:
    return Path(directory or default_cache_directory()) / (_digest(_serialize(identity)) + ".json.gz")


def _read(path: Path, identity: dict) -> dict:
    with gzip.open(path, "rb") as stream:
        envelope = json.loads(stream.read())
    payload = envelope["payload"]
    if payload["identity"] != identity or _digest(_serialize(payload)) != envelope["payload_sha256"]:
        raise ValueError("Hull cache identity or payload checksum mismatch")
    return payload


def _body_inertials(stage, path):
    from pxr import Usd, UsdPhysics

    result = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.MassAPI(prim)
            result[str(prim.GetPath())[len(path) :]] = dict(
                mass=float(api.GetMassAttr().Get()),
                diagonal_inertia=list(api.GetDiagonalInertiaAttr().Get()),
                center_of_mass=list(api.GetCenterOfMassAttr().Get()),
            )
    return result


def _transformed_points(vertices, transform, scale):
    from ..tasks.api import rotation_matrix

    transform = np.asarray(transform)
    return (np.asarray(vertices) * scale) @ rotation_matrix(transform[3:]).T + transform[:3]


def _candidate_colliders(stage, robot_path):
    from pxr import Usd, UsdPhysics

    return [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
        if "/collisions/" in str(prim.GetPath())
        and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexDecomposition"
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    ]


def capture_robot_hulls(builder, stage, robot_path: str, cache_directory: Path | None = None) -> dict:
    """Persist normal-import hulls, or verify a cache hit against the actual builder.

    Call before SDF provisioning/finalize from a public MODEL_INIT callback.
    Failure raises before cache creation; callers may report an optional-cache
    capture failure, but a reused-cache verification failure must stop simulation.
    """
    if stage.GetPrimAtPath(robot_path).GetCustomDataByKey("hiwonder:collisionProfile") == "manipulation":
        from .robot_collisions import restore_profile_hulls

        return restore_profile_hulls(builder, stage, robot_path)

    import newton

    source = _source(stage, robot_path)
    identity = _identity(source, builder)
    path = _cache_path(identity, cache_directory)
    if stage.GetPrimAtPath(robot_path).GetCustomDataByKey("hiwonder:collisionCacheApplied"):
        return restore_cached_builder_hulls(builder, stage, robot_path, cache_directory)
    original = _candidate_colliders(stage, robot_path)
    if not original:
        return {"status": "NO_DECOMPOSITION_COLLIDERS"}
    labels = list(builder.shape_label)
    body_labels = list(builder.body_label)
    groups = []
    selected = set()
    for prim in original:
        label = str(prim.GetPath())
        indices = [i for i, value in enumerate(labels) if value == label or value.startswith(label + "_convex_")]
        if not indices or sum(labels[i] == label for i in indices) != 1:
            raise ValueError(f"Cannot uniquely group decomposed hulls for {label}")
        parts = []
        for index in indices:
            mesh = builder.shape_source[index]
            if builder.shape_type[index] != newton.GeoType.CONVEX_MESH or mesh is None:
                raise ValueError(f"Expected actual convex output for {labels[index]}")
            if mesh.maxhullvert != newton.Mesh.MAX_HULL_VERTICES:
                raise ValueError("Custom per-mesh hull vertex limits require a separate cache implementation")
            body = body_labels[builder.shape_body[index]]
            if not label.startswith(body + "/") or not body.startswith(robot_path + "/"):
                raise ValueError(f"Unexpected body association for {labels[index]}: {body}")
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            triangles = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
            transform = np.asarray(builder.shape_transform[index], dtype=float).reshape(7)
            scale = np.asarray(builder.shape_scale[index], dtype=float).reshape(3)
            if not np.isfinite(vertices).all() or np.any(scale <= 0):
                raise ValueError("Nonfinite or reflected hull geometry is not cacheable")
            points = _transformed_points(vertices, transform, scale)
            parts.append(
                dict(
                    vertices=vertices.tolist(),
                    triangles=triangles.tolist(),
                    transform=transform.tolist(),
                    scale=scale.tolist(),
                    flags=int(builder.shape_flags[index]),
                    body=body[len(robot_path) :],
                    bounds_body_local=[points.min(0).tolist(), points.max(0).tolist()],
                )
            )
            selected.add(index)
        groups.append(dict(original=label[len(robot_path) :], parts=parts))
    authored = _body_inertials(stage, robot_path)
    physical = {}
    for relative, values in authored.items():
        index = body_labels.index(robot_path + relative)
        physical[relative] = dict(
            mass=float(builder.body_mass[index]),
            inertia=np.asarray(builder.body_inertia[index]).reshape(3, 3).tolist(),
            center_of_mass=np.asarray(builder.body_com[index]).tolist(),
        )
    payload = dict(identity=identity, groups=groups, authored_inertials=authored, builder_inertials=physical)
    envelope = dict(payload=payload, payload_sha256=_digest(_serialize(payload)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with gzip.open(temporary, "wb") as stream:
        stream.write(_serialize(envelope))
    os.replace(temporary, path)
    return {
        "status": "CAPTURED_EXACT_NEWTON_HULLS",
        "file": str(path),
        "original_colliders": len(groups),
        "hulls": len(selected),
    }


def apply_cached_robot_collisions(stage, robot_path: str, cache_directory: Path | None = None) -> dict:
    """Author cached convex pieces before import, preserving mesh-local frames and inertia."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    if os.environ.get("HIWONDER_COLLISION_CACHE_DISABLE") == "1":
        return {"status": "CACHE_DISABLED"}
    identity = _identity(_source(stage, robot_path))
    path = _cache_path(identity, cache_directory)
    if not path.exists():
        return {"status": "CACHE_MISS", "file": str(path)}
    payload = _read(path, identity)
    originals = {str(prim.GetPath())[len(robot_path) :] for prim in _candidate_colliders(stage, robot_path)}
    if originals != {group["original"] for group in payload["groups"]}:
        raise ValueError("Cached original collider set disagrees with current robot")
    before = _body_inertials(stage, robot_path)
    flat = stage.Flatten()
    max_error = 0.0
    for group in payload["groups"]:
        original_path = robot_path + group["original"]
        for index, part in enumerate(group["parts"]):
            body_path = robot_path + part["body"]
            destination = f"{body_path}/cached_collisions/{Path(group['original']).name}/hull_{index}"
            UsdGeom.Xform.Define(stage, destination.rsplit("/", 1)[0])
            Sdf.CreatePrimInLayer(stage.GetRootLayer(), destination)
            Sdf.CopySpec(flat, original_path, stage.GetRootLayer(), destination)
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(destination))
            mesh.CreatePointsAttr(part["vertices"])
            mesh.CreateFaceVertexCountsAttr([3] * len(part["triangles"]))
            mesh.CreateFaceVertexIndicesAttr(np.asarray(part["triangles"]).reshape(-1).tolist())
            mesh.ClearXformOpOrder()
            t, q, scale = part["transform"][:3], part["transform"][3:], part["scale"]
            mesh.AddTranslateOp(opSuffix="cached").Set(Gf.Vec3d(*t))
            mesh.AddOrientOp(opSuffix="cached").Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
            mesh.AddScaleOp(opSuffix="cached").Set(Gf.Vec3f(*scale))
            mesh.CreateExtentAttr(
                [Gf.Vec3f(*np.min(part["vertices"], axis=0)), Gf.Vec3f(*np.max(part["vertices"], axis=0))]
            )
            UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            # CoACD's max_convex_hull caps the number of parts, not each part's
            # vertices. Prevent the ordinary convexHull importer from reducing
            # a captured >64-vertex piece during this exact-geometry reuse path.
            mesh.GetPrim().AddAppliedSchema("NewtonMeshCollisionAPI")
            mesh.GetPrim().CreateAttribute("newton:maxHullVertices", Sdf.ValueTypeNames.Int).Set(
                max(identity["settings"]["max_convex_hull"], len(part["vertices"]))
            )
            UsdPhysics.CollisionAPI(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            matrix = UsdGeom.XformCache().ComputeRelativeTransform(mesh.GetPrim(), stage.GetPrimAtPath(body_path))[0]
            actual = np.array([matrix.Transform(Gf.Vec3d(*p)) for p in part["vertices"]])
            expected = _transformed_points(part["vertices"], part["transform"], part["scale"])
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            if not np.allclose([actual.min(0), actual.max(0)], part["bounds_body_local"], atol=1e-6, rtol=0):
                raise ValueError("Cached hull body-local bounds changed while authoring")
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(original_path)).CreateCollisionEnabledAttr(False)
    if max_error > 1e-6 or before != _body_inertials(stage, robot_path):
        raise ValueError("Cached hull transform or authored mass/inertia changed")
    stage.GetPrimAtPath(robot_path).SetCustomDataByKey("hiwonder:collisionCacheApplied", str(path))
    return {
        "status": "CACHE_APPLIED_PENDING_BUILDER_VERIFICATION",
        "file": str(path),
        "maximum_transformed_vertex_error_m": max_error,
        "authored_mass_inertia_unchanged": True,
        "original_colliders": len(payload["groups"]),
        "hulls": sum(len(g["parts"]) for g in payload["groups"]),
    }


def restore_cached_builder_hulls(builder, stage, robot_path: str, cache_directory: Path | None = None) -> dict:
    """Restore exact captured mesh topology after USD's automatic convex re-hull.

    MODEL_INIT exposes the mutable public ModelBuilder before physics allocation.
    USD import can retriangulate or simplify coplanar hulls even when its vertex
    limit is raised. Replace only cached mesh source objects through Mesh.copy;
    do not recompute inertia or alter body/shape transforms, flags or materials.
    This must run before SDF generation so SDFs see the original mesh frames.
    """
    import newton

    identity = _identity(_source(stage, robot_path), builder)
    payload = _read(_cache_path(identity, cache_directory), identity)
    labels, bodies = list(builder.shape_label), list(builder.body_label)
    restored = 0
    for group in payload["groups"]:
        for index, part in enumerate(group["parts"]):
            label = f"{robot_path}{part['body']}/cached_collisions/{Path(group['original']).name}/hull_{index}"
            if labels.count(label) != 1:
                raise ValueError(f"Cached hull missing or duplicated before restoration: {label}")
            item = labels.index(label)
            if (
                bodies[builder.shape_body[item]] != robot_path + part["body"]
                or int(builder.shape_flags[item]) != part["flags"]
            ):
                raise ValueError("Cached shape body/flags changed before exact mesh restoration")
            mesh = builder.shape_source[item]
            if builder.shape_type[item] != newton.GeoType.CONVEX_MESH:
                raise ValueError("Cached shape did not import as a convex mesh")
            if mesh is None or mesh.sdf is not None:
                raise ValueError("Exact mesh restoration requires a source mesh before SDF provisioning")
            replacement = mesh.copy(vertices=part["vertices"], indices=part["triangles"], recompute_inertia=False)
            replacement.maxhullvert = identity["settings"]["max_convex_hull"]
            builder.shape_source[item] = replacement
            restored += 1
    report = validate_cached_builder(builder, stage, robot_path, cache_directory)
    report["exact_mesh_sources_restored_before_sdf"] = restored
    return report


def validate_cached_builder(builder, stage, robot_path: str, cache_directory: Path | None = None) -> dict:
    """Check reimported hull point sets, flags, body frames and physical inertias."""
    identity = _identity(_source(stage, robot_path), builder)
    payload = _read(_cache_path(identity, cache_directory), identity)
    labels, bodies = list(builder.shape_label), list(builder.body_label)
    max_vertex_error = 0.0
    for group in payload["groups"]:
        original_label = robot_path + group["original"]
        for item, label in enumerate(labels):
            if label == original_label and int(builder.shape_flags[item]) & 6:
                raise ValueError("Original decomposed collider remained collision-enabled after cache replacement")
        for index, part in enumerate(group["parts"]):
            label = f"{robot_path}{part['body']}/cached_collisions/{Path(group['original']).name}/hull_{index}"
            if labels.count(label) != 1:
                raise ValueError(f"Cached hull missing or duplicated after import: {label}")
            item = labels.index(label)
            if (
                bodies[builder.shape_body[item]] != robot_path + part["body"]
                or int(builder.shape_flags[item]) != part["flags"]
            ):
                raise ValueError("Cached hull body or contact flags changed on import")
            mesh = builder.shape_source[item]
            if mesh is None:
                raise ValueError("Cached convex shape lost its mesh")
            if not np.array_equal(np.asarray(mesh.indices).reshape(-1), np.asarray(part["triangles"]).reshape(-1)):
                raise ValueError("Cached hull triangle topology changed after exact restoration")
            vertices = np.asarray(mesh.vertices)
            reference = np.asarray(part["vertices"])
            # The public convex-hull importer may reorder equivalent vertices.
            distances = np.linalg.norm(vertices[:, None] - reference[None], axis=-1)
            max_vertex_error = max(max_vertex_error, float(max(distances.min(0).max(), distances.min(1).max())))
            if not np.allclose(builder.shape_transform[item], part["transform"], atol=1e-6, rtol=0) or not np.allclose(
                builder.shape_scale[item], part["scale"], atol=1e-7, rtol=0
            ):
                raise ValueError("Cached hull mesh frame or scale changed on import")
    authored = _body_inertials(stage, robot_path)
    for relative, current in authored.items():
        baseline = payload["authored_inertials"][relative]
        ratio = current["mass"] / baseline["mass"]
        if not np.allclose(
            np.array(current["diagonal_inertia"]), np.array(baseline["diagonal_inertia"]) * ratio, rtol=1e-5, atol=1e-10
        ):
            raise ValueError("Cache verification supports coherent per-link mass/inertia scaling only")
        reference = payload["builder_inertials"][relative]
        item = bodies.index(robot_path + relative)
        if not np.isclose(builder.body_mass[item], reference["mass"] * ratio, rtol=1e-5, atol=1e-7) or not np.allclose(
            np.asarray(builder.body_inertia[item]).reshape(3, 3),
            np.array(reference["inertia"]) * ratio,
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError("Cached hull import changed physical body mass/inertia")
        if not np.allclose(builder.body_com[item], reference["center_of_mass"], atol=1e-6, rtol=0):
            raise ValueError("Cached hull import changed physical center of mass")
    if max_vertex_error > 1e-6:
        raise ValueError(f"Cached convex hull geometry changed on import: {max_vertex_error}m")
    return {
        "status": "PASS_EXACT_HULL_CACHE_REIMPORT",
        "maximum_mesh_vertex_set_error_m": max_vertex_error,
        "body_assignments_flags_frames_unchanged": True,
        "physical_mass_inertia_equivalent": True,
    }
