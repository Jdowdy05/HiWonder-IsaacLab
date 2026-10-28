# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Real Newton importer: exact gripper frames, inertia and self-collision masks."""

import gzip
import json

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_hiwonder.assets.registration import ASSET_DIR
from isaaclab_hiwonder.sim.robot_collisions import (
    apply_robot_collision_profile,
    load_collision_profile,
    restore_profile_hulls,
)

from pxr import Usd, UsdGeom, UsdPhysics


def build(variant, enabled):
    wp.set_device("cpu")
    data = load_collision_profile()
    row = data["variants"][variant]
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = stage.DefinePrim("/Robot", "Xform")
    root.GetReferences().AddReference(str(ASSET_DIR / "generated" / row["asset_file"]))
    root.SetCustomDataByKey("hiwonder:sourceAsset", str(ASSET_DIR / "generated" / row["asset_file"]))
    receipt = apply_robot_collision_profile(stage, "/Robot", native=True, self_collisions=enabled, profile=data)
    builder = newton.ModelBuilder()
    builder.add_usd(stage)
    restored = restore_profile_hulls(builder, stage, "/Robot")
    return builder, stage, data, receipt, restored


@pytest.mark.parametrize("variant", ("nominal", "scale099", "scale101"))
def test_real_import_retains_registered_inertia_and_exact_gripper_hulls(variant):
    builder, stage, data, receipt, restored = build(variant, True)
    row = data["variants"][variant]
    assert restored["status"] == "PASS_COLLISION_PROFILE_REIMPORT"
    assert restored["mesh_sources_restored_before_sdf"] == row["profile_hulls"]
    with gzip.open(ASSET_DIR / "generated" / row["reference_file"], "rb") as stream:
        reference = json.load(stream)["payload"]
    # Existing cached hulls were captured in randomized scenes. Their cached
    # mass/inertia values describe that first scene, not this fresh CAD actor.
    # Geometry is immutable; physical properties must match current authored USD.
    for relative in reference["builder_inertials"]:
        index = builder.body_label.index("/Robot" + relative)
        authored = UsdPhysics.MassAPI(stage.GetPrimAtPath("/Robot" + relative))
        np.testing.assert_allclose(builder.body_mass[index], authored.GetMassAttr().Get(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(builder.body_inertia[index]).reshape(3, 3),
            np.diag(np.asarray(authored.GetDiagonalInertiaAttr().Get())),
            rtol=1e-5,
            atol=1e-9,
        )
        np.testing.assert_allclose(builder.body_com[index], authored.GetCenterOfMassAttr().Get(), atol=1e-7, rtol=0)
    for entry in row["shapes"]:
        if entry["link"] not in ("gripper_body", "moving_jaw"):
            continue
        index = builder.shape_label.index("/Robot/" + entry["path"])
        mesh = builder.shape_source[index]
        group = row["groups"][entry["group_index"]]
        source_group, source_index = group["references"][entry["reference_index"]]
        original = reference["groups"][source_group]["parts"][source_index]
        np.testing.assert_array_equal(mesh.vertices, np.asarray(original["vertices"], dtype=np.float32))
        np.testing.assert_array_equal(mesh.indices, np.asarray(original["triangles"], dtype=np.int32).ravel())
        np.testing.assert_allclose(builder.shape_transform[index], original["transform"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(builder.shape_scale[index], original["scale"], atol=1e-7, rtol=0)


def test_self_collision_enables_nonadjacent_pairs_and_keeps_only_joint_interface_exclusions():
    builder, stage, *_ = build("nominal", True)

    def shape(owner, link):
        return next(
            i
            for i, label in enumerate(owner.shape_label)
            if label.startswith("/Robot/" + link + "/manipulation_collisions/")
        )

    base, shoulder, forearm = [shape(builder, link) for link in ("base", "shoulder", "forearm")]
    pairs = builder.shape_collision_filter_pairs
    assert tuple(sorted((base, forearm))) not in pairs
    assert tuple(sorted((base, shoulder))) in pairs
    disabled, *_ = build("nominal", False)
    assert tuple(sorted((shape(disabled, "base"), shape(disabled, "forearm")))) in disabled.shape_collision_filter_pairs
