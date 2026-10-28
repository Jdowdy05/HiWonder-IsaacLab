# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Patched caps stay closed at actual SDK precision and retain contact triangles."""

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from pxr import Usd, UsdGeom

from isaaclab_hiwonder.assets.robot_models import get_robot_model


@pytest.mark.parametrize("variant,scale", (("nominal", 1), ("scale099", 0.99), ("scale101", 1.01)))
def test_repaired_cap_meshes_are_closed_and_preserve_source_finger_contact_surfaces(variant, scale):
    model = get_robot_model("eka_v6")
    stage = Usd.Stage.Open(str(model.asset_path(variant)))
    rotation = np.asarray(model.mount["rotation"])
    translation = np.asarray(model.mount["translation_cad_mm"])
    with np.load(model.asset_directory / "original_cap_meshes.npz", allow_pickle=False) as source:
        for name, index, sign in (("eka_right_tip", 6, -1), ("eka_left_tip", 13, 1)):
            prim = UsdGeom.Mesh(stage.GetPrimAtPath(f"/HiWonder/{name}/collisions/eka_{index}"))
            points = np.asarray(prim.GetPointsAttr().Get())
            faces = np.asarray(prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            actual = trimesh.Trimesh(points, faces, process=True)
            assert actual.is_watertight and actual.is_winding_consistent and actual.volume > 0
            assert len(actual.split(only_watertight=False)) == 1
            vertices, old_faces = source[f"v{index}"], source[f"f{index}"]
            raw = trimesh.Trimesh(vertices, old_faces, process=False)
            center = raw.triangles_center
            contact = (
                (raw.face_normals[:, 0] * sign > 0.5) & (center[:, 1] < -58) & (center[:, 2] > 0) & (center[:, 2] < 21)
            )
            assert contact.any()
            expected = (
                (vertices @ rotation.T + translation - np.asarray(model.links[name]["origin_mm"])) * (0.001 * scale)
            ).astype(np.float32)
            distance, closest = cKDTree(actual.vertices).query(expected)
            ids = np.unique(old_faces[contact])
            assert distance[ids].max() <= 1e-8
            actual_faces = {tuple(sorted(map(int, face))) for face in actual.faces}
            for face in old_faces[contact]:
                mapped = tuple(sorted(map(int, closest[face])))
                assert mapped in actual_faces, "A source contact triangle was removed or bridged"
