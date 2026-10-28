# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded five-arm-DOF CAD inverse kinematics for reset rejection.

Position reachability is necessary but does not establish a collision-free path,
grasp, or feasible orientation for precision assembly.
"""

import json
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from isaaclab_hiwonder.assets.kinematics import forward_kinematics
from isaaclab_hiwonder.assets.registration import JOINTS
from isaaclab_hiwonder.tasks.api import rotation_matrix


@lru_cache(maxsize=1)
def _base_bounds():
    """Conservative complete CAD base envelope, including non-collision electronics."""
    manifest = Path(__file__).parents[1] / "assets" / "generated" / "manifest.json"
    artifacts = json.loads(manifest.read_text(encoding="utf-8"))["artifacts"]
    nominal = next(item for item in artifacts if item["scale"] == 1.0)
    return np.asarray(nominal["links"]["base"]["bounds_local_m"])


def _initial_base_overlap(definition, base, yaw, scale):
    """Conservative initial-geometry rejection, not a robot path collision checker.

    The oriented CAD base envelope is fixed below 76 mm. Elevated grasp seeds
    therefore remain eligible. Each compound part, cable segment and surface
    triangle is bounded separately; apertures and cable paths are not filled by
    a single task-wide box. Two millimeters cover collision contact envelopes.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    inverse_rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    bounds = _base_bounds() * scale

    def intersects(vertices, padding=0.0):
        local = (np.asarray(vertices) - base) @ inverse_rotation.T
        lower, upper = local.min(axis=-2), local.max(axis=-2)
        margin = np.asarray(padding) + 0.002
        return bool(np.any(np.all(upper + margin >= bounds[0], axis=-1) & np.all(lower - margin <= bounds[1], axis=-1)))

    signs = np.asarray(list(product((-1, 1), repeat=3)))
    for body in definition.bodies:
        state = definition.initial[body.name]
        body_rotation = rotation_matrix(state.orientation)
        # Each loose-object XY jitter component is bounded by 8 mm. The
        # Euclidean bound remains conservative at every sampled base yaw.
        jitter = np.array([np.sqrt(2) * 0.008, np.sqrt(2) * 0.008, 0]) if not body.static else 0.0
        for shape in body.shapes:
            if not shape.collision:
                continue
            if shape.kind == "mesh":
                vertices = np.asarray(shape.vertices)
            else:
                if shape.kind == "box":
                    half = np.asarray(shape.dimensions) / 2
                elif shape.kind == "cylinder":
                    radius, height = shape.dimensions
                    half = np.array([radius, radius, height / 2])
                elif shape.kind == "sphere":
                    half = np.full(3, shape.dimensions[0])
                else:
                    raise ValueError(f"Unsupported initial collider {shape.kind}")
                vertices = signs * half
            vertices = vertices @ rotation_matrix(shape.orientation).T + shape.position
            vertices = vertices @ body_rotation.T + state.position
            if intersects(vertices, jitter):
                return True
    for spec in definition.deformables:
        cells = spec.edges if spec.kind == "cable" else spec.triangles
        if len(cells) and intersects(np.asarray(spec.vertices)[cells], spec.radius):
            return True
    return False


def solve_position(target, base_position=(0, 0, 0), base_yaw=np.pi / 2, scale=1.0, tolerance=0.004):
    """Return a bounded six-joint witness or None; jaw stays open at -20 degrees."""
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Need a finite Cartesian target")
    cosine, sine = np.cos(base_yaw), np.sin(base_yaw)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    local = rotation.T @ (target - base_position)
    if np.linalg.norm(local) > 0.55:
        return None
    limits = np.deg2rad(np.array([joint["limits_deg"] for joint in JOINTS[:5]]))

    def residual(arm):
        return forward_kinematics(np.r_[arm, -np.pi / 9], scale)["tool_center"][:3, 3] - local

    seeds = [np.zeros(5), [0, -0.6, 0.8, 0.4, 0], [0, 0.6, -0.8, -0.4, 0], [0, -0.9, -0.7, 0.8, 0]]
    for seed in seeds:
        result = least_squares(
            residual, seed, bounds=(limits[:, 0], limits[:, 1]), max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7
        )
        if np.linalg.norm(residual(result.x)) <= tolerance:
            return np.r_[result.x, -np.pi / 9]
    return None


def task_landmarks(definition):
    """Finite contact/goal workspace candidates independent of hidden actor observations."""
    points = [
        np.asarray(definition.initial[body.name].position)
        for body in definition.bodies
        if body.shapes and (not body.static or body.name in ("socket", "ring", "container"))
    ]
    parameters = definition.parameters
    for key in ("seat", "base", "center"):
        value = np.asarray(parameters.get(key, []))
        if value.shape == (3,):
            points.append(value)
    for value in parameters.get("targets", {}).values():
        if np.asarray(value).shape == (3,):
            points.append(np.asarray(value))
    for value in parameters.get("goal_landmarks", []):
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Task goal landmarks must be finite 3D positions")
        points.append(point)
    for spec in definition.deformables:
        points.extend(spec.vertices[np.linspace(0, len(spec.vertices) - 1, 4).astype(int)])
    return np.asarray(points)


def reject_unreachable_bases(manager, definitions, env_ids=None, max_attempts=32):
    """Resample base draws until initial geometry clears the base and CAD IK succeeds.

    Structural geometry and bank IDs remain untouched. The small outward margin
    also covers the immutable loose-object bank perturbation. A fixed/disabled
    unreachable pose fails clearly rather than granting extra robot DOFs.
    """
    ids = list(range(manager.num_envs)) if env_ids is None else np.asarray(env_ids).astype(int).reshape(-1).tolist()
    witnesses = {}
    for env_id in ids:
        points = task_landmarks(definitions[env_id])
        for attempt in range(max_attempts):
            sample = manager.snapshot([env_id])
            base = sample["base"]["position"][0].numpy().copy()
            base[2] = 0  # All candidate coordinates are relative to the same support.
            quaternion = sample["base"]["quaternion_xyzw"][0].numpy()
            yaw = 2 * np.arctan2(quaternion[2], quaternion[3]) + np.pi / 2
            scale = float(sample["robot"]["geometry_scale"][0])
            poses = []
            overlap = _initial_base_overlap(definitions[env_id], base, yaw, scale)
            for point in [] if overlap else points:
                target = point.copy()
                delta = target[:2] - base[:2]
                target[:2] += 0.009 * delta / max(np.linalg.norm(delta), 1e-8)
                witness = solve_position(target, base, yaw, scale)
                if witness is None:
                    break
                poses.append(witness)
            if not overlap and len(poses) == len(points):
                witnesses[env_id] = np.asarray(poses)
                break
            control = manager.config.controls.base
            if not control.enabled or control.frozen or control.cadence == "fixed":
                if overlap:
                    raise ValueError(f"Initial task/base overlap at fixed base in environment {env_id}")
                raise ValueError(f"Unreachable fixed base/task layout in environment {env_id}")
            manager.reset([env_id], groups=["base"], max_attempts=max_attempts)
        else:
            raise ValueError(
                f"No reachable base/task layout clear of initial overlap after {max_attempts} attempts"
                f" in environment {env_id}"
            )
    return witnesses
