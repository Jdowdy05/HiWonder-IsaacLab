# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-ready compound meshes; apertures are never convex-filled."""

import numpy as np

from .api import ShapeSpec


def box(size, position=(0.0, 0.0, 0.0)):
    return ShapeSpec("box", tuple(size), tuple(position))


def cylinder(radius, height, position=(0.0, 0.0, 0.0)):
    return ShapeSpec("cylinder", (radius, height), tuple(position))


def prism(polygon, height, position=(0.0, 0.0, 0.0)):
    """Extrude a convex CCW polygon. Each returned mesh is one convex collider."""
    polygon = np.asarray(polygon, dtype=float)
    n = len(polygon)
    vertices = np.concatenate(
        [
            np.column_stack([polygon, np.full(n, -height / 2)]),
            np.column_stack([polygon, np.full(n, height / 2)]),
        ]
    )
    faces = []
    for i in range(1, n - 1):
        faces.extend([(0, i + 1, i), (n, n + i, n + i + 1)])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([(i, j, j + n), (i, j + n, i + n)])
    return ShapeSpec("mesh", position=tuple(position), vertices=vertices, triangles=np.asarray(faces))


def annulus(inner, outer, height, segments=32):
    """Separate convex wedges preserve the physical through-hole in every backend."""
    if not 0 < inner < outer or segments < 3:
        raise ValueError("Annulus requires 0 < inner < outer and at least three segments")
    shapes = []
    for i in range(segments):
        a, b = np.array([i, i + 1]) * (2 * np.pi / segments)
        p = [
            (inner * np.cos(a), inner * np.sin(a)),
            (outer * np.cos(a), outer * np.sin(a)),
            (outer * np.cos(b), outer * np.sin(b)),
            (inner * np.cos(b), inner * np.sin(b)),
        ]
        shapes.append(prism(p, height))
    return shapes


def polygon_socket(polygon, clearance, height, wall=0.006, floor=0.003):
    """Wall strips around convex hole with a closed floor below insertion depth."""
    polygon = np.asarray(polygon, dtype=float)
    shapes = []
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        edge = b - a
        outward = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
        p = [
            a + outward * clearance,
            a + outward * (clearance + wall),
            b + outward * (clearance + wall),
            b + outward * clearance,
        ]
        shapes.append(prism(p, height))
    width = np.max(np.abs(polygon), axis=0) * 2 + 2 * (wall + clearance)
    shapes.append(box((*width, floor), (0, 0, -(height + floor) / 2)))
    return shapes


def tray(width=0.06, length=0.05, depth=0.018, thickness=0.003):
    return [
        box((width, length, thickness), (0, 0, thickness / 2)),
        box((width, thickness, depth), (0, (length - thickness) / 2, depth / 2)),
        box((width, thickness, depth), (0, -(length - thickness) / 2, depth / 2)),
        box((thickness, length, depth), ((width - thickness) / 2, 0, depth / 2)),
        box((thickness, length, depth), (-(width - thickness) / 2, 0, depth / 2)),
    ]


def gear(module=0.0015, teeth=16, height=0.007, bore=0.003):
    """Involute tooth flanks, individually convex-fanned, with a physical shaft bore."""
    pitch = module * teeth / 2
    base = pitch * np.cos(np.deg2rad(20))
    root, tip = pitch - 1.25 * module, pitch + module
    shapes = annulus(bore, root, height)

    def inv(radius):
        t = np.sqrt(max((radius / base) ** 2 - 1, 0))
        return t - np.arctan(t)

    half = np.pi / (2 * teeth)
    for index in range(teeth):
        center = 2 * np.pi * index / teeth
        radii = np.linspace(max(base, root), tip, 5)
        left = [
            (r * np.cos(center - half - inv(pitch) + inv(r)), r * np.sin(center - half - inv(pitch) + inv(r)))
            for r in radii
        ]
        right = [
            (r * np.cos(center + half + inv(pitch) - inv(r)), r * np.sin(center + half + inv(pitch) - inv(r)))
            for r in radii[::-1]
        ]
        outline = (
            [(root * np.cos(center - half), root * np.sin(center - half))]
            + left
            + right
            + [(root * np.cos(center + half), root * np.sin(center + half))]
        )
        anchor = np.array([root * np.cos(center), root * np.sin(center)])
        for a, b in zip(outline[:-1], outline[1:]):
            shapes.append(prism([anchor, a, b], height))
    return shapes


def garment_mesh(kind, size=0.18, spacing=0.009):
    """Structured triangulation of distinct garment silhouettes, including openings.

    Shirts have sleeves/neck, shorts have a crotch gap, underwear two leg cuts,
    socks have an L-shaped foot. These are miniature one-arm tabletop garments.
    """
    proportions = {
        "towel": (1, 0.7),
        "handtowel": (0.7, 0.5),
        "rag": (0.7, 0.65),
        "shirt": (1, 1),
        "shorts": (0.8, 0.8),
        "underwear": (0.8, 0.55),
        "sock": (0.5, 1),
    }
    if kind not in proportions:
        raise ValueError(f"Unknown garment {kind}")
    w, h = np.array(proportions[kind]) * size
    # Odd dimensions place a material row on the commanded center crease.
    # Keep vertex identity/count stable across size variants in a batched asset.
    nominal_w, nominal_h = np.asarray(proportions[kind]) * 0.18
    xs = np.linspace(-w / 2, w / 2, max(5, (int(nominal_w / spacing) // 2) * 2 + 1))
    ys = np.linspace(-h / 2, h / 2, max(5, (int(nominal_h / spacing) // 2) * 2 + 1))

    def inside(x, y):
        u, v = x / w, y / h
        if kind == "shirt":
            return (abs(u) <= 0.27 or v > 0.12) and not (abs(u) < 0.12 and v > 0.34)
        if kind == "shorts":
            return not (abs(u) < 0.09 and v < 0)
        if kind == "underwear":
            return abs(u) < 0.15 + 0.7 * (v + 0.5)
        if kind == "sock":
            return u < 0.03 or v < -0.25
        return True

    vertices, lookup, faces = [], {}, []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if inside(x, y):
                lookup[i, j] = len(vertices)
                vertices.append([x, y, 0])
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            for cell in [((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i, j + 1))]:
                if all(v in lookup for v in cell):
                    faces.append([lookup[v] for v in cell])
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=int)
    if kind in ("shirt", "shorts", "underwear", "sock"):
        # Front/back fabric panels with sewn side seams and open neck/waist/cuffs.
        # This preserves garment topology (a sock tube, shirt sleeves, leg openings).
        n = len(vertices)
        boundary = {}
        boundary_direction = {}
        for face in faces:
            for a, b in zip(face, np.roll(face, -1)):
                key = tuple(sorted((int(a), int(b))))
                boundary[key] = boundary.get(key, 0) + 1
                boundary_direction[key] = (int(a), int(b))
        seams = []
        for key, count in boundary.items():
            if count != 1:
                continue
            a, b = boundary_direction[key]
            midpoint = (vertices[a] + vertices[b]) / 2
            u, v = midpoint[0] / w, midpoint[1] / h
            if kind == "shirt":
                opening = v < -0.49 or (abs(u) > 0.49 and v > 0.12) or (abs(u) < 0.14 and v > 0.30)
            elif kind == "shorts":
                opening = v > 0.49 or v < -0.49
            elif kind == "underwear":
                opening = v > 0.49 or (-0.49 < v < 0.15 and abs(u) > 0.1)
            else:
                opening = v > 0.49
            if not opening:
                seams.extend([(b, a, a + n), (b, a + n, b + n)])
        back = vertices.copy()
        back[:, 2] += 0.0014
        vertices = np.concatenate([vertices, back])
        faces = np.concatenate([faces, faces[:, ::-1] + n, np.asarray(seams, int)])
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    return vertices, edges, faces


def paper_mesh(scale=1.0):
    """Rectangle triangulated along both 45-degree nose creases and center fold.

    A four-triangle cell has both diagonals; its center is a real material vertex.
    Without crease-conforming triangles, a target fold would shorten edges across
    each crease and demand impossible in-plane compression.
    """
    xs = np.linspace(-0.09, 0.09, 31) * scale
    ys = np.linspace(-0.06, 0.06, 21) * scale
    vertices = [[x, y, 0] for y in ys for x in xs]
    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            corners = [j * len(xs) + i, j * len(xs) + i + 1, (j + 1) * len(xs) + i + 1, (j + 1) * len(xs) + i]
            center = len(vertices)
            vertices.append([(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, 0])
            faces.extend([(corners[k], corners[(k + 1) % 4], center) for k in range(4)])
    faces = np.asarray(faces, int)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    return np.asarray(vertices), edges, faces
