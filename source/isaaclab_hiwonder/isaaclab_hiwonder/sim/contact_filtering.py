# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Validate PhysX filter names and aggregate compound actor contact evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..assets.robot_models import get_robot_model


def body_contact_aliases(robot_model, body: str) -> tuple[str, ...]:
    """Return all semantic contacts: a distal tip can pinch and also block release."""
    return tuple(alias for alias, bodies in get_robot_model(robot_model).contact_bodies.items() if body in bodies)


def robot_contact_actors(robot_model, robot_path: str) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Give each sensing body one owner, retaining every semantic alias separately."""
    model = get_robot_model(robot_model)
    if model.name == "stock":
        paths = {alias: robot_path + "/" + bodies[0] for alias, bodies in model.contact_bodies.items()}
        return paths, {alias: (alias,) for alias in paths}
    bodies = dict.fromkeys(body for group in model.contact_bodies.values() for body in group)
    paths = {"robot_contact_" + body: robot_path + "/" + body for body in bodies}
    aliases = {"robot_contact_" + body: body_contact_aliases(model, body) for body in bodies}
    return paths, aliases


def contact_filter_layout(
    actor_paths: Mapping[str, str], collider_paths: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """Build body-level filters after rejecting ambiguous collider leaf names.

    PhysX 6.2 wildcard body filters named ``shape_0`` can also select generic
    collider children named ``shape_0`` below other actors. This affects sensing
    patterns too, so expanding only target paths cannot repair it. Task USD
    authoring uses collider names distinct from all actor names. Body-level
    filtering then includes every wall/floor of a compound socket unchanged.
    """
    all_colliders = [path for paths in collider_paths.values() for path in paths]
    layout = {}
    for actor, body_path in actor_paths.items():
        leaf = body_path.rsplit("/", 1)[-1]
        alias = any(path != body_path and path.rsplit("/", 1)[-1] == leaf for path in all_colliders)
        if alias:
            raise ValueError(f"PhysX contact actor {actor} aliases a collider leaf name; use distinct collider names")
        layout[actor] = [body_path]
    return layout


def actor_contact_presence(
    force_matrix: np.ndarray, filter_labels: Sequence[str], threshold_n: float = 1e-5
) -> dict[str, np.ndarray]:
    """Map measured force columns to actor contacts, preserving environment order.

    Matrix axes are environment, sensing body, filter column, world XYZ force.
    Multiple columns may name one compound actor. Test each measured column
    before OR aggregation: opposing forces on two walls must not cancel away
    evidence of contact with their shared actor.
    """
    matrix = np.asarray(force_matrix)
    if matrix.ndim != 4 or matrix.shape[2] != len(filter_labels) or matrix.shape[3] != 3:
        raise ValueError(f"Contact matrix/actor mapping mismatch: {matrix.shape}, {len(filter_labels)} columns")
    if not np.isfinite(matrix).all() or not math.isfinite(threshold_n) or threshold_n < 0:
        raise ValueError("Contact forces and nonnegative threshold must be finite")
    column_contact = np.any(np.linalg.norm(matrix, axis=-1) > threshold_n, axis=1)
    result = {}
    for column, actor in enumerate(filter_labels):
        if actor not in result:
            result[actor] = column_contact[:, column].copy()
        else:
            result[actor] |= column_contact[:, column]
    return result
