# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction support and conservative task/physical-nuisance separation."""

import itertools

import numpy as np

from isaaclab_hiwonder.randomization import table_outline
from isaaclab_hiwonder.tasks.api import rotation_matrix


def _vertices(shape):
    if shape.kind == "mesh":
        vertices = np.asarray(shape.vertices)
    elif shape.kind == "box":
        vertices = np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * shape.dimensions
    elif shape.kind == "sphere":
        vertices = np.array(list(itertools.product((-1, 1), repeat=3))) * shape.dimensions[0]
    elif shape.kind == "cylinder":
        radius, height = shape.dimensions
        angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        disk = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros(32)))
        vertices = np.concatenate((disk + (0, 0, height / 2), disk - (0, 0, height / 2)))
    else:
        raise ValueError(f"No support geometry adapter for {shape.kind}")
    return vertices @ rotation_matrix(shape.orientation).T + shape.position


def task_supported(definition, table, shape_name, margin=0.003):
    """Require support for loose reset envelopes, fixtures, soft geometry and stack goals.

    Jointed children are supported by their mechanism root and may overhang the
    table. Their geometry still participates in physical nuisance separation.
    """
    outline = table_outline(
        shape_name, float(table["width"]), float(table["depth"]), float(table["edge_radius"])
    ).numpy()
    lower, upper = _task_reset_bounds(definition, True, for_support=True)
    if not len(lower):
        return True
    points = np.concatenate(
        (
            lower[:, :2],
            upper[:, :2],
            np.column_stack((lower[:, 0], upper[:, 1])),
            np.column_stack((upper[:, 0], lower[:, 1])),
        )
    )
    edges = np.roll(outline, -1, axis=0) - outline
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths > 1e-12
    edges, outline, lengths = edges[valid], outline[valid], lengths[valid]
    relative = points[:, None] - outline[None]
    distance = (edges[None, :, 0] * relative[..., 1] - edges[None, :, 1] * relative[..., 0]) / lengths
    return bool(np.all(distance >= margin - 1e-7))


def _body_reset_transforms(definition, include_reset_envelopes):
    initial = {}
    for name, state in definition.initial.items():
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix(state.orientation)
        pose[:3, 3] = state.position
        initial[name] = pose
    positions = dict(definition.parameters.get("initial_joints", {}))
    coordinates = [positions]
    if include_reset_envelopes and definition.name in ("knob", "latch"):
        near = positions.copy()
        name = definition.parameters["joint"]
        start = near.get(name, 0.0)
        near[name] = start + 0.65 * (definition.parameters["target"] - start)
        coordinates.append(near)
    for values in coordinates:
        transforms = dict(initial)
        deltas = {}
        children = {joint.child for joint in definition.joints}
        pending = list(definition.joints)
        while pending:
            ready = [joint for joint in pending if joint.parent not in children or joint.parent in deltas]
            if not ready:
                raise ValueError("Task joints must have an acyclic rooted topology")
            for joint in ready:
                motion = np.eye(4)
                axis = np.asarray(joint.axis, dtype=float)
                axis /= np.linalg.norm(axis)
                q = values.get(joint.name, 0.0)
                if joint.kind == "prismatic":
                    motion[:3, 3] = axis * q
                else:
                    rotation = rotation_matrix(np.r_[axis * np.sin(q / 2), np.cos(q / 2)])
                    pivot = np.asarray(joint.position)
                    motion[:3, :3] = rotation
                    motion[:3, 3] = pivot - rotation @ pivot
                delta = deltas.get(joint.parent, np.eye(4)) @ motion
                deltas[joint.child] = delta
                transforms[joint.child] = delta @ initial[joint.child]
                pending.remove(joint)
        yield transforms
    if include_reset_envelopes and definition.name in ("peg_insert", "connector_insert"):
        near = dict(initial)
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(definition.parameters["seat"]) + (0.022, -0.012, 0.035)
        near[definition.parameters["moving"]] = pose
        yield near


def _task_reset_bounds(definition, include_reset_envelopes, for_support=False):
    bounds = []
    connected = {name for joint in definition.joints for name in (joint.parent, joint.child)}
    children = {joint.child for joint in definition.joints}
    if definition.name == "bottle_cap" and definition.parameters["operation"] == "open":
        connected.add("cap")
    for transforms in _body_reset_transforms(definition, include_reset_envelopes):
        for body in definition.bodies:
            if for_support and body.name in children:
                continue
            pose = transforms[body.name]
            jitter = np.zeros(3)
            if (
                include_reset_envelopes
                and not body.static
                and body.name not in connected
                and not definition.deformables
            ):
                jitter[:2] = np.sqrt(2) * 0.008
            for shape in body.shapes:
                if not shape.collision:
                    continue
                if shape.kind == "cylinder":
                    radius, height = shape.dimensions
                    # A sampled polygon would underbound the continuous cylinder.
                    points = np.asarray(list(itertools.product((-1, 1), repeat=3))) * (radius, radius, height / 2)
                    points = points @ rotation_matrix(shape.orientation).T + shape.position
                else:
                    points = _vertices(shape)
                points = points @ pose[:3, :3].T + pose[:3, 3]
                bounds.append((points.min(0) - jitter, points.max(0) + jitter))
    if for_support and definition.name == "cube_stack":
        bodies = {body.name: body for body in definition.bodies}
        for index, name in enumerate(definition.parameters["order"]):
            center = np.asarray(definition.parameters["base"]) + (0, 0, index * definition.parameters["size"])
            for shape in bodies[name].shapes:
                if shape.collision:
                    points = _vertices(shape) + center
                    bounds.append((points.min(0), points.max(0)))
    for spec in definition.deformables:
        cells = spec.edges if spec.kind == "cable" else spec.triangles
        for vertices in np.asarray(spec.vertices)[cells]:
            bounds.append((vertices.min(0) - spec.radius, vertices.max(0) + spec.radius))
    if not bounds:
        return np.empty((0, 3)), np.empty((0, 3))
    lower, upper = map(np.asarray, zip(*bounds))
    # Cover the native/PhysX collision envelopes, not only strict mesh interior.
    margin = 0.0 if for_support else 0.002
    return lower - margin, upper + margin


def _bounds_overlap_obb(lower, upper, center, rotation, half_extent):
    """Exact15-axis SAT for conservative task AABBs against the nuisance OBB."""
    if not len(lower):
        return False
    axes = np.concatenate((np.eye(3), rotation.T, np.cross(np.eye(3)[:, None], rotation.T[None]).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    midpoint, half = (lower + upper) / 2, (upper - lower) / 2
    projected_radius = half @ np.abs(axes).T + half_extent @ np.abs(rotation.T @ axes.T)
    separation = np.abs((midpoint - center) @ axes.T) > projected_radius + 1e-10
    return bool(np.any(~np.any(separation, axis=1)))


def physical_nuisance_overlaps(definition, snapshot_one_env, randomization_config, include_reset_envelopes=True):
    """Return offending physical group names: ``distractor`` and/or ``confetti``.

    The snapshot has the manager's one-environment leading dimension. Task poses
    and derived nuisance heights use the scene author's common support frame,
    including low-floor support overrides. Bounds are conservative per collider,
    rod segment and surface triangle; apertures are not task-wide filled boxes.
    Construction includes loose-object jitter and built-in near-goal placements.
    An actual immutable reset record must pass False to avoid reserving those twice.
    External stable-grasp placements require that actual-record check as well.
    """
    active = {
        group: snapshot_one_env[group]["active"] & snapshot_one_env[group]["physical"]
        for group in ("distractor", "confetti")
    }
    if not any(mask.any() for mask in active.values()):
        return set()
    fields = {
        "distractor": ("active", "physical", "position", "size", "quaternion_xyzw"),
        "confetti": ("active", "physical", "position", "size", "aspect_ratio", "yaw"),
    }
    sample = {
        group: {key: snapshot_one_env[group][key][0].detach().cpu().numpy() for key in names}
        for group, names in fields.items()
    }
    lower, upper = _task_reset_bounds(definition, include_reset_envelopes)
    overlaps = set()
    distractors = sample["distractor"]
    for i in np.flatnonzero(distractors["active"] & distractors["physical"]):
        size = distractors["size"][i]
        center = distractors["position"][i].astype(float).copy()
        center[2] = size[2] / 2 + 0.001
        rotation = rotation_matrix(distractors["quaternion_xyzw"][i])
        # Ellipsoid/cylinder distractors fit inside the same oriented box.
        if _bounds_overlap_obb(lower, upper, center, rotation, size / 2):
            overlaps.add("distractor")
            break
    cards = sample["confetti"]
    for i in np.flatnonzero(cards["physical"] & cards["active"]):
        size, aspect = cards["size"][i], cards["aspect_ratio"][i]
        center = cards["position"][i].astype(float).copy()
        center[2] = randomization_config.confetti.elevation_m
        yaw = cards["yaw"][i]
        rotation = rotation_matrix((0, 0, np.sin(yaw / 2), np.cos(yaw / 2)))
        if _bounds_overlap_obb(lower, upper, center, rotation, np.array((size / 2, size * aspect / 2, 0.00005))):
            overlaps.add("confetti")
            break
    return overlaps


def bootstrap_task_cohorts(manager, definitions, max_attempts=32):
    """Qualify structural task/table layouts before baking physics and constructing the SGS bank.

    Enabled, unfrozen support groups can redraw regardless of their runtime cadence. Frozen
    robot/appearance fields and per-reset actuator/sensor streams retain their original draws.
    """
    if len(definitions) != manager.num_envs or max_attempts < 1:
        raise ValueError("Need one definition per environment and a positive bootstrap budget")
    checks = []
    for i, definition in enumerate(definitions):
        for attempt in range(max_attempts):
            sample = manager.snapshot([i])
            table = {name: value[0] for name, value in sample["table"].items()}
            shape = manager.config.table.shapes[int(table["shape_id"])]
            supported = task_supported(definition, table, shape)
            overlaps = physical_nuisance_overlaps(definition, sample, manager.config)
            if supported and not overlaps:
                checks.append(attempt + 1)
                break
            control = manager.config.controls.table
            if not supported and (not control.enabled or control.frozen):
                raise ValueError(f"Task {definition.name} has unsupported frozen/disabled table cohort {i}")
            for group in sorted(overlaps):
                control = getattr(manager.config.controls, group)
                if not control.enabled or control.frozen:
                    raise ValueError(f"Frozen/disabled {group} intersects task {definition.name} in cohort {i}")
            manager.redraw_cohorts([i], max_attempts=max_attempts)
        else:
            raise ValueError(
                f"No supported task/physical-nuisance separation for {definition.name}"
                f" after {max_attempts} attempts in cohort {i}"
            )
    return dict(
        accepted=[True] * manager.num_envs,
        task_support_checks=checks,
        initial_draw_attempts=manager.bootstrap_attempts.tolist(),
    )
