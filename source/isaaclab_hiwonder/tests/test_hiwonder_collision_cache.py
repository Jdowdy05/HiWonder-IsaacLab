# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise real Newton USD reimport of a cached concave fixture on CPU.

This owns the cache frame/inertia contract. A translated, rotated, nonuniformly
scaled concave shape catches double transforms and replacing its parts by one
hull; disabling CoACD on reimport proves that the acceleration actually applies.
"""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_hiwonder.sim.collision_cache import (
    apply_cached_robot_collisions,
    capture_robot_hulls,
    validate_cached_builder,
)

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def test_cached_hulls_preserve_real_import_frames_and_inertia(tmp_path, monkeypatch):
    wp.set_device("cpu")
    source = tmp_path / "concave.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("hiwonder:sourceAsset", str(source))
    body = UsdGeom.Xform.Define(stage, "/Robot/body")
    body.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.3, 0.4))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(0.2)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.001, 0.002, 0.003))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.01, 0.02, 0.03))
    mesh = UsdGeom.Mesh.Define(stage, "/Robot/body/collisions/part_0")
    polygon = [(0, 0), (0.1, 0), (0.1, 0.02), (0.02, 0.02), (0.02, 0.1), (0, 0.1)]
    vertices = [(x, y, z) for z in (-0.01, 0.01) for x, y in polygon]
    top = [(6, 7, 8), (6, 8, 9), (6, 9, 10), (6, 10, 11)]
    faces = [(c - 6, b - 6, a - 6) for a, b, c in top] + top
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([(i, j, j + 6), (i, j + 6, i + 6)])
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.01, -0.02, 0.03))
    mesh.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 15).GetQuat()))
    mesh.AddScaleOp().Set(Gf.Vec3f(1.2, 0.8, 1.1))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
    stage.GetRootLayer().Save()
    original = newton.ModelBuilder()
    original.add_usd(stage)
    report = capture_robot_hulls(original, stage, "/Robot", tmp_path / "cache")
    assert report["hulls"] > 1
    applied = apply_cached_robot_collisions(stage, "/Robot", tmp_path / "cache")
    assert applied["maximum_transformed_vertex_error_m"] < 1e-6
    assert applied["authored_mass_inertia_unchanged"]
    import coacd

    def no_repeat_decomposition(*args, **kwargs):
        raise AssertionError("Cached import unexpectedly invoked CoACD")

    monkeypatch.setattr(coacd, "run_coacd", no_repeat_decomposition)
    restored = newton.ModelBuilder()
    restored.add_usd(stage)
    checked = capture_robot_hulls(restored, stage, "/Robot", tmp_path / "cache")
    assert checked["status"] == "PASS_EXACT_HULL_CACHE_REIMPORT"
    np.testing.assert_allclose(restored.body_mass, original.body_mass, atol=1e-7)
    np.testing.assert_allclose(restored.body_inertia, original.body_inertia, atol=1e-8)
    item = next(i for i, label in enumerate(restored.shape_label) if "/cached_collisions/" in label)
    transform = restored.shape_transform[item]
    restored.shape_transform[item] = wp.transform((transform.p[0] + 0.004, transform.p[1], transform.p[2]), transform.q)
    with pytest.raises(ValueError, match="mesh frame or scale changed"):
        validate_cached_builder(restored, stage, "/Robot", tmp_path / "cache")
