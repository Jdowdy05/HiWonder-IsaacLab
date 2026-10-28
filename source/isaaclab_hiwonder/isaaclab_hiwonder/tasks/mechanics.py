# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constitutive models and topology measurements used by task backend adapters.

Force-returning functions must be applied on every physics substep, with equal
opposite reactions, not as reset-time pose edits. Parameters are estimates awaiting
material/contact qualification. No function advances a rigid body by teleporting it.
"""

from dataclasses import dataclass

import numpy as np

from .api import ActorState, rotation_matrix


def cable_attachment_wrench(nodes, velocities, plug: ActorState, local_anchors, stiffness, damping):
    """Three noncollinear endpoint anchors transfer translation AND orientation.

    Returns node forces and the equal-opposite torque on the rigid plug. The rigid
    plug also receives -sum(forces). A centerline-only pin cannot satisfy this model;
    the cable endpoint must have the triangular material frame specified by anchors.
    """
    offsets = np.asarray(local_anchors) @ rotation_matrix(plug.orientation).T
    targets = np.asarray(plug.position) + offsets
    target_velocity = np.asarray(plug.linear_velocity) + np.cross(plug.angular_velocity, offsets)
    forces = stiffness * (targets - nodes) + damping * (target_velocity - velocities)
    return forces, -np.cross(offsets, forces).sum(axis=0)


def screw_reaction(angle, displacement, angular_velocity, linear_velocity, pitch, stiffness=1500, damping=2):
    """Reciprocal force/torque from U=k*(z-pitch*theta/(2*pi))^2/2."""
    ratio = pitch / (2 * np.pi)
    force = -stiffness * (displacement - ratio * angle) - damping * (linear_velocity - ratio * angular_velocity)
    return force, -ratio * force


def detent_torque(angle, velocity, states, stiffness=0.03, damping=0.002):
    """Conservative periodic multiwell potential with bounded detent torque."""
    lo, hi = states
    span = hi - lo
    return -stiffness * np.sin(2 * np.pi * (angle - lo) / span) - damping * velocity


@dataclass
class PlasticHinge:
    """Elastoplastic return mapping with isotropic hardening; radians and N*m."""

    stiffness: float
    yield_angle: float
    hardening: float = 0.02
    rest_angle: float = 0.0
    accumulated_plastic: float = 0.0

    def advance(self, angle):
        elastic = float(np.arctan2(np.sin(angle - self.rest_angle), np.cos(angle - self.rest_angle)))
        threshold = self.yield_angle + self.hardening * self.accumulated_plastic
        excess = abs(elastic) - threshold
        if excess > 0:
            increment = excess / (1 + self.hardening)
            self.rest_angle += np.sign(elastic) * increment
            self.rest_angle = float(np.arctan2(np.sin(self.rest_angle), np.cos(self.rest_angle)))
            self.accumulated_plastic += increment
        residual = np.arctan2(np.sin(angle - self.rest_angle), np.cos(angle - self.rest_angle))
        return -self.stiffness * residual


def dihedral(points):
    a, b, c, d = np.asarray(points)
    axis = b - a
    axis /= max(np.linalg.norm(axis), 1e-12)
    first, second = np.cross(b - a, c - a), np.cross(d - a, b - a)
    first /= max(np.linalg.norm(first), 1e-12)
    second /= max(np.linalg.norm(second), 1e-12)
    return np.arctan2(np.dot(np.cross(first, second), axis), np.dot(first, second))


class PaperCreaseModel:
    """Persistent rest-angle shell bending applied through vertex forces.

    Native surface elasticity supplies stretch/shear and contact. This replaces
    bending along the indexed crease hinges; do not double-count native bending
    there. Finite-difference energy gradients are a CPU reference implementation,
    intended for correctness validation before a batched Warp implementation.
    """

    def __init__(self, hinges, stiffness=0.015, yield_angle=0.15):
        self.hinges = np.asarray(hinges, dtype=int)
        self.models = [PlasticHinge(stiffness, yield_angle) for _ in self.hinges]

    @property
    def rest_angles(self):
        return np.array([h.rest_angle for h in self.models])

    def forces(self, positions):
        forces = np.zeros_like(positions, dtype=float)
        epsilon = 1e-6
        for indices, hinge in zip(self.hinges, self.models):
            points = np.array(positions[indices], dtype=float)
            torque = hinge.advance(dihedral(points))
            gradient = np.zeros((4, 3))
            for vertex in range(4):
                for dim in range(3):
                    plus, minus = points.copy(), points.copy()
                    plus[vertex, dim] += epsilon
                    minus[vertex, dim] -= epsilon
                    delta = dihedral(plus) - dihedral(minus)
                    gradient[vertex, dim] = np.arctan2(np.sin(delta), np.cos(delta)) / (2 * epsilon)
            np.add.at(forces, indices, torque * gradient)
        return forces


def surface_hinges(triangles):
    adjacency = {}
    for triangle in triangles:
        for i in range(3):
            a, b, c = triangle[i], triangle[(i + 1) % 3], triangle[(i + 2) % 3]
            edge = tuple(sorted((int(a), int(b))))
            if edge in adjacency:
                x, y, opposite = adjacency.pop(edge)
                yield x, y, opposite, int(c)
            else:
                adjacency[edge] = (int(a), int(b), int(c))


def ring_crossings(nodes, center, normal, radius):
    """Signed segment intersections with the aperture disk, excluding tangencies."""
    points = np.asarray(nodes) - center
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    distances = points @ normal
    count = 0
    for i in range(len(points) - 1):
        a, b = distances[i : i + 2]
        if (a <= 0 < b) or (b <= 0 < a):
            intersection = points[i] + (-a / (b - a)) * (points[i + 1] - points[i])
            if np.linalg.norm(intersection) < radius:
                count += 1 if b > a else -1
    return count


def knot_determinant(nodes):
    """Alexander determinant at -1 of an explicitly closed polygonal curve.

    Three generic projections must agree. Zero means degenerate/ambiguous, never
    a success. Determinants 3/5 identify candidate overhand/figure-eight topology
    within the generated reset family, not a universal complete knot classifier.
    """
    points = np.asarray(nodes, dtype=float)
    if np.linalg.norm(points[0] - points[-1]) > 1e-8:
        raise ValueError("Knot measurement requires an explicit documented closure")
    answers = []
    for direction in [(1, 0.31, 0.17), (0.23, 1, 0.41), (0.37, 0.19, 1)]:
        normal = np.asarray(direction)
        normal /= np.linalg.norm(normal)
        x = np.cross(normal, [0, 0, 1])
        x /= np.linalg.norm(x)
        y = np.cross(normal, x)
        xy, depths = points @ np.stack([x, y]).T, points @ normal
        crossings = []
        for i in range(len(points) - 1):
            for j in range(i + 2, len(points) - 1):
                if i == 0 and j == len(points) - 2:
                    continue
                matrix = np.column_stack([xy[i + 1] - xy[i], -(xy[j + 1] - xy[j])])
                if abs(np.linalg.det(matrix)) < 1e-12:
                    continue
                t, u = np.linalg.solve(matrix, xy[j] - xy[i])
                if 1e-7 < t < 1 - 1e-7 and 1e-7 < u < 1 - 1e-7:
                    zi = depths[i] + t * (depths[i + 1] - depths[i])
                    zj = depths[j] + u * (depths[j + 1] - depths[j])
                    if abs(zi - zj) < 1e-7:
                        return 0
                    crossings.append((i + t, j + u) if zi > zj else (j + u, i + t))
        if not crossings:
            answers.append(1)
            continue
        under = sorted(c[1] for c in crossings)
        n = len(under)
        matrix = np.zeros((n, n), dtype=float)
        for row, (over, below) in enumerate(crossings):
            entering = under.index(below)
            exiting = (entering + 1) % n
            over_arc = np.searchsorted(under, over, side="right") % n
            matrix[row, over_arc] += 2
            matrix[row, entering] -= 1
            matrix[row, exiting] -= 1
        answers.append(int(round(abs(np.linalg.det(matrix[:-1, :-1])))))
    return answers[0] if len(set(answers)) == 1 else 0


def close_rope(nodes, apex=(0.25, 0, 0.6)):
    """Fixed external closure, valid only while both endpoints stay in goal gates."""
    return np.concatenate([nodes, [apex], [nodes[0]]])
