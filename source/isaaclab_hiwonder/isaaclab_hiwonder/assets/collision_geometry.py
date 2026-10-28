# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Offline convex subset covers with whole-solid geometric certificates.

Every new convex piece is inside one old convex piece. Dropped pieces have a
bounded distance to one retained convex piece. Convexity extends vertex checks
to the complete solids; this does not depend on sampling joint configurations.
SciPy/trimesh are loaded only by offline generation and validation.
"""

from __future__ import annotations

import numpy as np

INSET_M = 0.000005


def _points(value):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Need finite three-dimensional convex vertex sets")
    return points


def _solid(points):
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    faces = hull.simplices.copy()
    triangles = points[faces]
    inward = (
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) * hull.equations[:, :3]
    ).sum(1) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return dict(planes=hull.equations, triangles=points[faces], faces=faces, volume=hull.volume)


def _distances(points, solid):
    from trimesh.triangles import closest_point

    if solid is None:
        return np.full(len(points), np.inf)
    result = np.zeros(len(points))
    planes, triangles = solid["planes"], solid["triangles"]
    outside = np.flatnonzero((points @ planes[:, :3].T + planes[:, 3] > 1e-12).any(1))
    block = max(1, min(128, 16384 // len(triangles)))
    for start in range(0, len(outside), block):
        indices = outside[start : start + block]
        queries = np.repeat(points[indices], len(triangles), axis=0)
        candidates = np.tile(triangles, (len(indices), 1, 1))
        nearest = closest_point(candidates, queries)
        result[indices] = np.linalg.norm(queries - nearest, axis=1).reshape(len(indices), -1).min(1)
    return result


def convex_distances(points, target_vertices):
    """Euclidean distances to a closed convex solid, with bounded work buffers."""
    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("Distance queries must be finite 3D points")
    return _distances(source, _solid(_points(target_vertices)))


def simplify_convex(vertices, max_vertices=64, max_surface_loss=0.0005):
    """Choose an inscribed vertex subset, or retain a shape that cannot qualify."""
    source = _points(vertices)
    if type(max_vertices) is not int or max_vertices < 4:
        raise ValueError("max_vertices must be at least four")
    if not np.isfinite(max_surface_loss) or max_surface_loss <= 2 * INSET_M:
        raise ValueError("surface loss must exceed the float-authoring inset")
    unchanged = dict(vertices=source.copy(), unchanged=True)
    if len(source) <= max_vertices:
        return unchanged
    selected = sorted(set(np.argmax(source @ np.concatenate((np.eye(3), -np.eye(3))).T, axis=0).tolist()))
    while len(selected) <= max_vertices:
        points = source[selected]
        solid = _solid(points)
        if solid is None:
            return unchanged
        distances = _distances(source, solid)
        if distances.max() <= max_surface_loss - 2 * INSET_M:
            center = points.mean(0)
            radius = -(solid["planes"][:, :3] @ center + solid["planes"][:, 3]).max()
            if radius <= 10 * INSET_M:
                return unchanged
            authored = (points + (center - points) * (INSET_M / radius)).astype(np.float32).astype(float)
            if convex_distances(source, authored).max() <= max_surface_loss:
                return dict(vertices=authored, unchanged=False)
            return unchanged
        if len(selected) == max_vertices:
            break
        index = int(np.argmax(distances))
        if index in selected:
            break
        selected.append(index)
    return unchanged


def select_collision_hulls(reference_hulls, max_surface_loss=0.002, max_vertices=64):
    """Remove nearby redundant hulls; never merge across an empty joint cavity."""
    if not np.isfinite(max_surface_loss) or max_surface_loss <= 0:
        raise ValueError("Body surface loss must be finite and positive")
    reference = [_points(p) for p in reference_hulls]
    if not reference:
        raise ValueError("At least one reference collision hull is required")
    solids = [_solid(p) for p in reference]
    order = sorted(range(len(reference)), key=lambda i: (-(solids[i]["volume"] if solids[i] else 0), i))
    lower = np.array([p.min(0) for p in reference])
    upper = np.array([p.max(0) for p in reference])
    witnesses = np.full(len(reference), -1, dtype=int)
    distances = np.full(len(reference), np.inf)
    parts = []
    for original in order:
        if witnesses[original] >= 0:
            continue
        candidate = simplify_convex(
            reference[original], max_vertices=max_vertices, max_surface_loss=min(0.0005, max_surface_loss / 2)
        )
        points = candidate["vertices"]
        solid = _solid(points)
        part_id = len(parts)
        parts.append(dict(reference_index=original, **candidate, triangles=solid["faces"] if solid else None))
        possible = np.flatnonzero(
            (witnesses < 0)
            & (lower >= points.min(0) - max_surface_loss).all(1)
            & (upper <= points.max(0) + max_surface_loss).all(1)
        )
        for index in possible:
            distance = (
                0.0
                if index == original and candidate["unchanged"]
                else float(_distances(reference[index], solid).max())
            )
            if distance <= max_surface_loss:
                witnesses[index], distances[index] = part_id, distance
        if witnesses[original] < 0:
            raise ValueError("Retained candidate failed its own coverage bound")
    return dict(parts=parts, witnesses=witnesses, bounds=distances, maximum_surface_loss_m=float(max_surface_loss))


def certify_hull_cover(reference_hulls, cover, max_surface_loss=0.002):
    """Independently verify subset membership and all whole-hull loss witnesses."""
    reference = [_points(p) for p in reference_hulls]
    parts = cover["parts"]
    if not parts or len(cover["witnesses"]) != len(reference):
        raise ValueError("Coverage witnesses are incomplete")
    compiled = []
    for part in parts:
        index = part["reference_index"]
        if type(index) is not int or not 0 <= index < len(reference):
            raise ValueError("Subset names an absent source hull")
        points = _points(part["vertices"])
        original = reference[index]
        if part["unchanged"]:
            if not np.array_equal(points, original):
                raise ValueError("Unchanged subset no longer equals its reference")
        else:
            planes = _solid(original)["planes"]
            authored = points.astype(np.float32).astype(float)
            if (authored @ planes[:, :3].T + planes[:, 3] > 1e-10).any():
                raise ValueError("New convex subset lies outside its original hull")
        compiled.append(_solid(points))
    maximum = 0.0
    for index, points in enumerate(reference):
        witness = int(cover["witnesses"][index])
        if not 0 <= witness < len(parts):
            raise ValueError("Coverage witness names an absent retained piece")
        part = parts[witness]
        distance = (
            0.0
            if part["unchanged"] and part["reference_index"] == index
            else float(_distances(points, compiled[witness]).max())
        )
        if not np.isfinite(distance) or distance > max_surface_loss + 1e-12:
            raise ValueError("Body coverage loss exceeds the declared bound")
        maximum = max(maximum, distance)
    return dict(
        reference_hulls=len(reference), retained_hulls=len(parts), maximum_body_loss_m=maximum, subset_certified=True
    )
