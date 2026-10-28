# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Independent analytic solids guard joint cavities and bounded body erosion."""

import copy

import numpy as np
import pytest
from isaaclab_hiwonder.assets.collision_geometry import (
    certify_hull_cover,
    convex_distances,
    select_collision_hulls,
    simplify_convex,
)


def box(low, high):
    return np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float
    )


def bracket():
    return [
        box((0, 0, 0), (0.002, 0.008, 0.003)),
        box((0.006, 0, 0), (0.008, 0.008, 0.003)),
        box((0, 0, 0), (0.008, 0.002, 0.003)),
    ]


def test_convex_surface_distances_do_not_confuse_far_vertices_with_near_faces():
    target = box((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02))
    points = np.array([[0, 0, 0], [0.0203, 0, 0], [0.0203, 0.0204, 0]])
    np.testing.assert_allclose(convex_distances(points, target), [0, 0.0003, 0.0005], atol=1e-12)


def test_cover_prunes_redundant_pieces_without_creating_a_new_joint_cavity():
    solids = bracket()
    solids += [box((0.0002, y, 0.0002), (0.0018, y + 0.0004, 0.0028)) for y in np.linspace(0.0003, 0.006, 16)]
    cover = select_collision_hulls(solids, max_surface_loss=0.001)
    checked = certify_hull_cover(solids, cover, max_surface_loss=0.001)
    assert len(cover["parts"]) == 3
    assert checked["maximum_body_loss_m"] <= 0.001
    gap = np.array([[0.004, 0.005, 0.0015]])
    assert all(convex_distances(gap, p["vertices"])[0] > 0.001 for p in cover["parts"])


def test_certificate_rejects_hull_that_fills_open_fork_and_invalid_coverage_witness():
    solids = bracket()
    cover = select_collision_hulls(solids, max_surface_loss=0.001)
    damaged = copy.deepcopy(cover)
    damaged["parts"][0]["vertices"] = box((0, 0, 0), (0.008, 0.008, 0.003))
    with pytest.raises(ValueError, match="subset|outside"):
        certify_hull_cover(solids, damaged, max_surface_loss=0.001)
    damaged = copy.deepcopy(cover)
    damaged["witnesses"][:] = 0
    with pytest.raises(ValueError, match="coverage|loss"):
        certify_hull_cover(solids, damaged, max_surface_loss=0.001)


def test_vertex_reduction_is_inscribed_after_float32_authoring():
    z = np.linspace(-0.008, 0.008, 9)
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    vertices = np.array([[0.017 * np.cos(a), 0.017 * np.sin(a), h] for h in z for a in angles])
    result = simplify_convex(vertices, max_vertices=48, max_surface_loss=0.0005)
    assert len(result["vertices"]) <= 48 and not result["unchanged"]
    from scipy.spatial import ConvexHull

    planes = ConvexHull(vertices).equations
    authored = result["vertices"].astype(np.float32).astype(float)
    assert (authored @ planes[:, :3].T + planes[:, 3] <= 1e-10).all()
    assert convex_distances(vertices, authored).max() <= 0.0005


def test_thin_link_and_input_rejections_keep_geometry_explicit():
    thin = box((0, 0, 0), (0.004, 0.004, 0.00001))
    cover = select_collision_hulls([thin], max_surface_loss=0.001)
    assert len(cover["parts"]) == 1
    assert certify_hull_cover([thin], cover, max_surface_loss=0.001)["maximum_body_loss_m"] < 1e-12
    with pytest.raises(ValueError, match="finite"):
        select_collision_hulls([np.full((8, 3), np.nan)])
    with pytest.raises(ValueError, match="loss"):
        select_collision_hulls([thin], max_surface_loss=-0.001)


def test_planar_reference_triangle_remains_explicit_when_no_solid_can_cover_it():
    triangle = np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.0, 0.003, 0.0]])
    cover = select_collision_hulls([triangle], max_surface_loss=0.001)
    assert len(cover["parts"]) == 1 and cover["parts"][0]["unchanged"]
    assert certify_hull_cover([triangle], cover, max_surface_loss=0.001)["maximum_body_loss_m"] == 0
