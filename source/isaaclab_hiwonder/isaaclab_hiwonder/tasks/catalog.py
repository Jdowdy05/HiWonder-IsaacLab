# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural tabletop tasks for the five-arm-DOF SO-ARM101.

All coordinates are relative to a support plane at z=0. The scene adds its table
height exactly once. Geometry dimensions are deliberately miniature and are
estimated design choices, not claimed measurements of commercial task hardware.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from .api import ActorState, BodySpec, CableMaterialSpec, DeformableSpec, JointSpec, ShapeSpec, TaskDefinition
from .geometry import annulus, box, cylinder, garment_mesh, gear, paper_mesh, polygon_socket, prism, tray
from .mechanics import surface_hinges

TASK_NAMES = (
    "peg_insert",
    "cube_stack",
    "nut_pick_place",
    "washer_ring",
    "connector_insert",
    "shape_sort",
    "knob",
    "switch",
    "latch",
    "bottle_cap",
    "key_lock",
    "barrel_jack",
    "rope_thread",
    "cable_untangle",
    "cloth_fold",
    "paper_airplane",
    "gear_assembly",
    "circuit_restore",
    "catch_ball",
    "roll_ball",
)
GARMENTS = ("towel", "handtowel", "shirt", "rag", "shorts", "underwear", "sock")
KNOT_VARIANTS = ("thread", "overhand", "figure_eight")


def build_task(  # noqa: C901
    name: str, variant: str | None = None, seed: int = 0, *, task_parameters: Mapping[str, object] | None = None
) -> TaskDefinition:
    """Sample a reproducible structural configuration, suitable for an SGS bank.

    ``goal`` is public commanded information only. Randomized object/fixture poses
    remain in ``initial``/``parameters`` for the critic and scene, not the actor.
    """
    if name not in TASK_NAMES:
        raise ValueError(f"Unknown task {name!r}; available: {TASK_NAMES}")
    if task_parameters is not None and not isinstance(task_parameters, Mapping):
        raise ValueError("task_parameters must be a mapping")
    options = dict(task_parameters or {})
    allowed = {
        "cube_stack": {"count", "colors", "order"},
        "knob": {"target_angle_rad"},
        "switch": {"target_state"},
        "latch": {"target_state"},
    }.get(name, set())
    if any(not isinstance(key, str) for key in options) or options.keys() - allowed:
        raise ValueError(f"Unsupported task parameters for {name}; allowed keys: {sorted(allowed)}")
    rng = np.random.default_rng(seed)
    bodies, joints, deformables, initial, parameters = [], [], [], {}, {}
    family, horizon = "rigid", 30.0
    variant = variant or (
        "towel"
        if name == "cloth_fold"
        else "thread"
        if name == "rope_thread"
        else "floor"
        if name == "roll_ball"
        else "open"
        if name == "bottle_cap"
        else "default"
    )
    goal = np.array([TASK_NAMES.index(name)], dtype=np.float32)

    def body(label, shapes, pos, static=False, mass=0.03, color=(0.4, 0.6, 0.8)):
        bodies.append(BodySpec(label, shapes, mass=mass, static=static, color=color))
        initial[label] = ActorState(tuple(pos))

    def loose(label, shapes, z, x=0.18, y=-0.07, **kwargs):
        body(label, shapes, (x + rng.uniform(-0.012, 0.012), y + rng.uniform(-0.012, 0.012), z), **kwargs)

    def socket(label, radius, height=0.025, x=0.27, y=0.04):
        shapes = annulus(radius, radius + 0.007, height)
        shapes.append(cylinder(radius + 0.007, 0.003, (0, 0, -(height + 0.003) / 2)))
        body(label, shapes, (x, y, height / 2 + 0.003), True)

    if name in ("peg_insert", "connector_insert", "barrel_jack"):
        is_keyed = name == "connector_insert"
        plug_name = "peg" if name == "peg_insert" else "plug"
        height, radius = 0.024, 0.006
        if is_keyed:
            poly = [(-0.008, -0.005), (0.008, -0.005), (0.008, 0.002), (0.004, 0.005), (-0.008, 0.005)]
            body("socket", polygon_socket(poly, 0.0007, height), (0.27, 0.04, height / 2 + 0.003), True)
            shapes = [prism(poly, height), box((0.025, 0.018, 0.009), (0, 0, 0.0165))]
        else:
            socket("socket", radius + 0.0007, height)
            shapes = [cylinder(radius, height), cylinder(0.01, 0.009, (0, 0, 0.0165))]
            if name == "barrel_jack":
                shapes = annulus(0.0022, radius, height) + [cylinder(0.01, 0.009, (0, 0, 0.0165))]
                bodies[-1].shapes.append(cylinder(0.0018, 0.02, (0, 0, -0.002)))
        loose(plug_name, shapes, height / 2 + 0.001)
        parameters.update(
            moving=plug_name,
            fixture="socket",
            seat=(0.27, 0.04, height / 2 + 0.003),
            position_tolerance=0.0008,
            angle_tolerance=0.08,
            keyed=is_keyed,
        )
        if name == "barrel_jack":
            family = "cable"
            start = np.array(initial[plug_name].position) + (0, 0, 0.022)
            nodes = np.array([start + (0.11 * t, 0.015 * np.sin(np.pi * t), 0.008 * t) for t in np.linspace(0, 1, 41)])
            edges = np.array([(i, i + 1) for i in range(40)])
            tangent = nodes[1] - nodes[0]
            tangent /= np.linalg.norm(tangent)
            orientation = np.r_[np.cross([0, 0, 1], tangent), 1 + tangent[2]]
            orientation /= np.linalg.norm(orientation)
            parameters["attachment_orientation"] = orientation
            deformables.append(
                DeformableSpec(
                    "cable",
                    "cable",
                    nodes,
                    edges,
                    np.empty((0, 3), int),
                    0.0015,
                    1100,
                    1e8,
                    1e6,
                    0.2,
                    attachments=[
                        {
                            "kind": "rigid_frame",
                            "body": "plug",
                            "segment": 0,
                            "bonded_segments": (0, 1, 2),
                            "parent_frame": tuple(np.r_[[0, 0, 0.022], orientation]),
                        }
                    ],
                    cable_material=CableMaterialSpec(),
                )
            )

    elif name == "cube_stack":
        colors = None
        if "colors" in options:
            colors = np.asarray(options["colors"], dtype=float)
            if colors.ndim != 2 or colors.shape[1] != 3 or not np.isfinite(colors).all():
                raise ValueError("Stack colors must be finite RGB triples")
        count = options.get("count", len(colors) if colors is not None else 3)
        if isinstance(count, bool) or not isinstance(count, Integral) or not 1 <= count <= 6:
            raise ValueError("Stack count must be an integer from one to six")
        if colors is None:
            colors = np.asarray(
                [
                    (0.9, 0.15, 0.1),
                    (0.1, 0.8, 0.2),
                    (0.1, 0.2, 0.9),
                    (0.9, 0.8, 0.1),
                    (0.8, 0.1, 0.8),
                    (0.1, 0.8, 0.8),
                ]
            )[:count]
        if len(colors) != count or np.any(colors < 0) or np.any(colors > 1) or len(np.unique(colors, axis=0)) != count:
            raise ValueError("Supply one distinct RGB color in [0,1] per cube")
        try:
            order = tuple(options.get("order", range(count)))
        except TypeError as error:
            raise ValueError("Stack order must be a permutation of cube indices") from error
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in order) or sorted(order) != list(
            range(count)
        ):
            raise ValueError("Stack order must be a permutation of cube indices")
        for i, color in enumerate(colors):
            loose(f"cube_{i}", [box((0.025,) * 3)], 0.0125, x=0.16 + 0.045 * i, color=tuple(color))
        parameters.update(
            order=[f"cube_{index}" for index in order],
            base=(0.25, 0, 0.0125),
            size=0.025,
            goal_landmarks=[(0.25, 0, 0.0125 + index * 0.025) for index in range(count)],
        )
        goal = np.r_[count, colors[list(order)].ravel()].astype(np.float32)

    elif name == "nut_pick_place":
        for i in range(4):
            loose(f"nut_{i}", annulus(0.003, 0.006, 0.004, 6), 0.002, x=0.15 + i * 0.03)
        body("container", tray(), (0.26, 0.065, 0), True)
        parameters.update(
            objects=[f"nut_{i}" for i in range(4)],
            center=(0.26, 0.065),
            half_extent=(0.024, 0.019),
            floor=0.003,
            rim=0.018,
        )
        goal = np.array([4], np.float32)

    elif name == "washer_ring":
        # Open C ring in the XZ plane: the top quarter is absent for threading.
        shapes = []
        radius = 0.025
        for a in np.linspace(np.pi / 4, 7 * np.pi / 4, 40):
            tangent = np.array([-np.sin(a), 0, np.cos(a)])
            q = np.cross([0, 0, 1], tangent)
            quaternion = np.r_[q, 1 + np.dot([0, 0, 1], tangent)]
            quaternion /= np.linalg.norm(quaternion)
            shape = cylinder(0.0025, radius * (1.5 * np.pi / 39) * 1.08, (radius * np.cos(a), 0, radius * np.sin(a)))
            shape.orientation = tuple(quaternion)
            shapes.append(shape)
        supports = [box((0.025, 0.028, 0.006), (-0.035, 0, -0.047)), box((0.006, 0.012, 0.044), (-0.027, 0, -0.022))]
        body("ring", shapes + supports, (0.27, 0.04, 0.05), True)
        loose("washer", annulus(0.0045, 0.011, 0.002), 0.001)
        parameters.update(seat=(0.27, 0.04, 0.023), ring_tangent=(1, 0, 0), aperture=0.0045)

    elif name == "shape_sort":
        polys = [
            np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.007,
            np.array([[0, 1], [-0.866, -0.5], [0.866, -0.5]]) * 0.010,
            np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 7)[:-1]]) * 0.008,
        ]
        targets = {}
        for i, poly in enumerate(polys):
            label = f"shape_{i}"
            loose(label, [prism(poly, 0.018)], 0.009, x=0.15 + 0.045 * i)
            pos = (0.15 + 0.065 * i, 0.065, 0.012)
            body(f"slot_{i}", polygon_socket(poly, 0.0008, 0.018), pos, True)
            targets[label] = pos
        parameters.update(targets=targets, symmetry_orders=[4, 3, 6])
        goal = np.arange(3, dtype=np.float32)

    elif name in ("knob", "switch", "latch", "bottle_cap", "key_lock"):
        family = "articulated"
        body("housing", [box((0.07, 0.06, 0.012))], (0.25, 0.03, 0.006), True)
        if name == "knob":
            target = options.get("target_angle_rad", 1.2)
            if (
                isinstance(target, bool)
                or not isinstance(target, Real)
                or not np.isfinite(target)
                or not -2.6 <= target <= 2.6
            ):
                raise ValueError("Knob target_angle_rad must be finite and within [-2.6,2.6]")
            body("knob", [cylinder(0.02, 0.013), box((0.032, 0.007, 0.008), (0, 0, 0.009))], (0.25, 0.03, 0.025))
            joints.append(
                JointSpec("knob_angle", "revolute", "housing", "knob", (0, 0, 1), -2.6, 2.6, 0.08, (0.25, 0.03, 0.025))
            )
            parameters.update(
                joint="knob_angle",
                target=float(target),
                tolerance=0.06,
                initial_joints={"knob_angle": 0.0 if abs(target) > 0.12 else -1.2},
            )
        elif name == "switch":
            target_state = options.get("target_state", "on")
            if target_state not in ("on", "off"):
                raise ValueError("Switch target_state must be on or off")
            target = 0.35 if target_state == "on" else -0.35
            body("switch", [box((0.028, 0.016, 0.006)), box((0.006, 0.006, 0.015), (0, 0, 0.008))], (0.25, 0.03, 0.022))
            joints.append(
                JointSpec(
                    "switch_angle",
                    "revolute",
                    "housing",
                    "switch",
                    (0, 1, 0),
                    -0.35,
                    0.35,
                    0.05,
                    (0.25, 0.03, 0.022),
                    parameters={"detent_states": (-0.35, 0.35)},
                )
            )
            parameters.update(
                joint="switch_angle", target=target, tolerance=0.05, initial_joints={"switch_angle": -target}
            )
        elif name == "latch":
            target_state = options.get("target_state", "engaged")
            if target_state not in ("engaged", "released"):
                raise ValueError("Latch target_state must be engaged or released")
            target, start = (0.022, 0.0) if target_state == "engaged" else (0.0, 0.022)
            body("bolt", [box((0.045, 0.009, 0.009)), cylinder(0.007, 0.012, (0, 0, 0.008))], (0.25, 0.03, 0.02))
            body(
                "keeper",
                [
                    box((0.006, 0.022, 0.007), (0, 0, 0.008)),
                    box((0.006, 0.006, 0.02), (0, 0.009, 0)),
                    box((0.006, 0.006, 0.02), (0, -0.009, 0)),
                ],
                (0.282, 0.03, 0.02),
                True,
            )
            joints.append(
                JointSpec("bolt_travel", "prismatic", "housing", "bolt", (1, 0, 0), 0, 0.026, 3, (0.25, 0.03, 0.02))
            )
            parameters.update(
                joint="bolt_travel",
                target=target,
                tolerance=0.002,
                keeper_x=0.282,
                target_state=target_state,
                initial_joints={"bolt_travel": start},
                goal_landmarks=[(0.25 + start, 0.03, 0.02), (0.25 + target, 0.03, 0.02)],
            )
        elif name == "bottle_cap":
            if variant not in ("open", "close"):
                raise ValueError("bottle_cap variants are open and close")
            body(
                "bottle",
                annulus(0.014, 0.018, 0.045) + [cylinder(0.018, 0.004, (0, 0, -0.0245))],
                (0.25, 0.03, 0.03),
                True,
            )
            body(
                "cap",
                annulus(0.0188, 0.023, 0.009) + [cylinder(0.023, 0.003, (0, 0, 0.006))],
                (0.25, 0.03, 0.048) if variant == "open" else (0.18, -0.07, 0.0045),
            )
            parameters.update(
                pitch=0.003,
                open_height=0.010,
                thread_turns=3,
                operation=variant,
                seat_offset=(0, 0, 0.018),
                minimum_grasp_intervals=2,
            )
        else:
            keypoly = [
                (-0.002, -0.007),
                (0.002, -0.007),
                (0.002, 0.002),
                (0.005, 0.002),
                (0.005, 0.006),
                (-0.002, 0.006),
            ]
            body("cylinder", polygon_socket(keypoly, 0.0006, 0.018), (0.25, 0.03, 0.026))
            # The concave L profile must remain a union of convex pieces; a
            # single convexHull collider would fill its notch into the socket.
            key_parts = [
                prism([(-0.002, -0.007), (0.002, -0.007), (0.002, 0.006), (-0.002, 0.006)], 0.018),
                prism([(0.002, 0.002), (0.005, 0.002), (0.005, 0.006), (0.002, 0.006)], 0.018),
                box((0.026, 0.014, 0.005), (0, 0, 0.012)),
            ]
            loose("key", key_parts, 0.009)
            body("bolt", [box((0.025, 0.008, 0.008))], (0.285, 0.03, 0.018))
            joints.extend(
                [
                    JointSpec(
                        "lock_angle", "revolute", "housing", "cylinder", (0, 0, 1), 0, 1.6, 0.1, (0.25, 0.03, 0.026)
                    ),
                    JointSpec(
                        "bolt_travel", "prismatic", "housing", "bolt", (1, 0, 0), -0.018, 0, 4, (0.285, 0.03, 0.018)
                    ),
                ]
            )
            body(
                "strike",
                [
                    box((0.006, 0.022, 0.005), (0, 0, 0.007)),
                    box((0.006, 0.005, 0.016), (0, 0.009, 0)),
                    box((0.006, 0.005, 0.016), (0, -0.009, 0)),
                ],
                (0.295, 0.03, 0.018),
                True,
            )
            parameters.update(
                seat=(0.25, 0.03, 0.026),
                unlock_angle=1.2,
                cam_radius=0.015,
                joint="lock_angle",
                target=1.4,
                tolerance=0.1,
            )
        goal = np.array([parameters.get("target", 1)], dtype=np.float32)
        if name == "bottle_cap":
            goal = np.array([0 if variant == "open" else 1], np.float32)

    elif name in ("rope_thread", "cable_untangle"):
        family, horizon = "cable", 60.0
        if name == "rope_thread" and variant not in KNOT_VARIANTS:
            raise ValueError(f"Rope variants: {KNOT_VARIANTS}")
        t = np.linspace(0, 1, 81)
        if name == "cable_untangle":
            nodes = np.column_stack(
                [
                    0.22 + 0.055 * np.sin(2 * np.pi * t),
                    0.045 * np.sin(4 * np.pi * t),
                    0.008 + 0.008 * (1 + np.cos(2 * np.pi * t)),
                ]
            )
            nodes[-1] += (0.015, 0, 0.005)
            length = np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()
            nodes = (nodes - nodes.mean(axis=0)) * (0.22 / length) + [0.23, 0, 0.012]
        else:
            nodes = np.column_stack([0.12 + 0.21 * t, np.full_like(t, -0.07), np.full_like(t, 0.003)])
            if variant != "thread":
                nodes[:, 1] = 0.045 * np.sin(4 * np.pi * t)
        # Sub-diameter rods give ill-conditioned contact dynamics at physical
        # inertia. Retain the authored centerline and total material length,
        # using forty approximately equal-arclength Cosserat segments.
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))))
        samples = np.linspace(0, arc[-1], 41)
        nodes = np.column_stack([np.interp(samples, arc, nodes[:, axis]) for axis in range(3)])
        midpoint = (nodes[0] + nodes[-1]) / 2
        nodes = midpoint + (nodes - midpoint) * (arc[-1] / np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())
        deformables.append(
            DeformableSpec(
                "rope",
                "cable",
                nodes,
                np.column_stack([np.arange(40), np.arange(1, 41)]),
                np.empty((0, 3), int),
                0.0015,
                950,
                5e7,
                2e4,
                0.05,
                cable_material=CableMaterialSpec(5e7, 2e4, 5e7, 2e4),
            )
        )
        if name == "rope_thread" and variant == "thread":
            rings = []
            for i, x in enumerate([0.22, 0.28]):
                body(f"ring_{i}", annulus(0.015, 0.019, 0.005), (x, 0.04, 0.04), True)
                initial[f"ring_{i}"].orientation = (0, 2**-0.5, 0, 2**-0.5)
                rings.append({"center": (x, 0.04, 0.04), "normal": (1, 0, 0), "radius": 0.0135})
            parameters.update(rings=rings)
        elif name == "rope_thread":
            # Released tails rest on the support plane: their geometric center
            # is one rope radius above it. The 3D endpoint tolerance is unchanged.
            tail_height = deformables[-1].radius
            parameters.update(
                knot_determinant=3 if variant == "overhand" else 5,
                endpoint_goals=((0.13, 0, tail_height), (0.33, 0, tail_height)),
                closure_apex=(0.23, 0, 0.6),
            )
        else:
            parameters.update(endpoint_goals=((0.12, 0, 0.004), (0.33, 0, 0.004)), straightness=0.93)

    elif name in ("cloth_fold", "paper_airplane"):
        family, horizon = ("cloth", 60.0) if name == "cloth_fold" else ("plastic_shell", 90.0)
        if name == "cloth_fold" and variant not in GARMENTS:
            raise ValueError(f"Garment variants: {GARMENTS}")
        vertices, edges, faces = (
            garment_mesh(variant, size=0.17 + rng.uniform(-0.008, 0.008))
            if name == "cloth_fold"
            else paper_mesh(scale=rng.uniform(0.94, 1.03))
        )
        vertices += (0.23, 0, 0.003)
        spec = DeformableSpec(
            "sheet",
            "surface",
            vertices,
            edges,
            faces,
            0.0007 if name == "cloth_fold" else 0.00006,
            250 if name == "cloth_fold" else 800,
            150 if name == "cloth_fold" else 5000,
            0.003 if name == "cloth_fold" else 0,
            0.03,
        )
        targets = vertices.copy()
        midpoint = (vertices[:, 0].min() + vertices[:, 0].max()) / 2
        moving = np.flatnonzero(vertices[:, 0] > midpoint)
        targets[moving, 0] = 2 * midpoint - vertices[moving, 0]
        targets[moving, 2] += 0.002
        if name == "paper_airplane":
            # Three measurable folds: two nose diagonals, then the longitudinal center crease.
            midpoint_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2
            stages = []
            for sign in (-1, 1):
                stage = vertices.copy() if not stages else stages[-1].copy()
                # Reflect only the corner beyond the diagonal crease; points on
                # that line are fixed, so the commanded map is continuous.
                signed = stage[:, 0] + sign * (stage[:, 1] - midpoint_y) - vertices[:, 0].max()
                selected = signed > 0
                stage[selected, 0] -= signed[selected]
                stage[selected, 1] -= sign * signed[selected]
                stage[selected, 2] += 0.0004
                stages.append(stage)
            final = stages[-1].copy()
            mask = final[:, 1] > midpoint_y
            final[mask, 1] = 2 * midpoint_y - final[mask, 1]
            final[mask, 2] += 0.0004
            stages.append(final)
            # Form both wings; the first wing operation is held above the table.
            # Final wings lie on the table with the center keel standing upright.
            half_width = (vertices[:, 1].max() - vertices[:, 1].min()) / 2
            hinge = 0.3 * half_width
            across = np.abs(final[:, 1] - midpoint_y)
            side = np.where(vertices[:, 1] >= midpoint_y, 1.0, -1.0)
            upright = final.copy()
            upright[:, 1] = midpoint_y
            upright[:, 2] = hinge - across + 0.003
            wings = across > hinge + 1e-8
            first = upright.copy()
            selected = wings & (side > 0)
            first[selected, 1] = midpoint_y + across[selected] - hinge
            first[selected, 2] = 0.003
            first[:, 2] += 2 * half_width
            stages.append(first)
            airplane = upright.copy()
            airplane[wings, 1] = midpoint_y + side[wings] * (across[wings] - hinge)
            airplane[wings, 2] = 0.003
            stages.append(airplane)
            hinges = np.asarray(list(surface_hinges(faces)), int)
            crease_masks = []
            for folded_vertices in stages:
                points = folded_vertices[hinges]
                first = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
                second = np.cross(points[:, 3] - points[:, 0], points[:, 1] - points[:, 0])
                cosine = np.sum(first * second, axis=1) / np.maximum(
                    np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1), 1e-15
                )
                crease_masks.append(np.arccos(np.clip(cosine, -1, 1)) > 0.3)
            spec.kind = "plastic_shell"
            spec.parameters.update(hinges=hinges, hinge_stiffness=2e-5, yield_angle=0.15)
            parameters.update(fold_stages=stages, hinges=hinges, crease_masks=crease_masks, crease_fraction=0.8)
            targets = airplane
        else:
            parameters.update(moving_landmarks=moving, garment=variant)
        parameters.update(target_vertices=targets, fold_tolerance=0.012, rest_vertices=vertices.copy())
        deformables.append(spec)
        goal = np.array([GARMENTS.index(variant) if name == "cloth_fold" else 7, 1], np.float32)

    elif name == "gear_assembly":
        family = "articulated"
        module, teeth = 0.0015, 16
        for i, x in enumerate([0.24, 0.264]):
            body(
                f"shaft_{i}",
                [cylinder(0.0025, 0.018), cylinder(0.006, 0.0025, (0, 0, -0.00775))],
                (x, 0.04, 0.009),
                True,
            )
        body("drive_gear", gear(module, teeth), (0.24, 0.04, 0.006))
        joints.append(
            JointSpec(
                "drive_angle", "revolute", "shaft_0", "drive_gear", (0, 0, 1), -1000, 1000, 0.03, (0.24, 0.04, 0.006)
            )
        )
        loose("gear", gear(module, teeth), 0.0035)
        parameters.update(seat=(0.264, 0.04, 0.006), teeth=teeth, module=module, phase_tolerance=0.07)

    elif name == "circuit_restore":
        family = "articulated"
        poly = [(-0.006, -0.004), (0.006, -0.004), (0.006, 0.001), (0.003, 0.004), (-0.006, 0.004)]
        for i, x in enumerate([0.20, 0.28]):
            body(f"terminal_{i}", polygon_socket(poly, 0.0006, 0.014), (x, 0.05, 0.01), True)
            loose(
                f"connector_{i}",
                [prism(poly, 0.014), box((0.019, 0.014, 0.006), (0, 0, 0.010))],
                0.007,
                x=0.17 + i * 0.07,
            )
        body("switch_housing", [box((0.03, 0.025, 0.008))], (0.25, -0.025, 0.004), True)
        body("switch", [box((0.022, 0.013, 0.005))], (0.25, -0.025, 0.012))
        joints.append(
            JointSpec(
                "switch_angle",
                "revolute",
                "switch_housing",
                "switch",
                (0, 1, 0),
                -0.35,
                0.35,
                0.04,
                (0.25, -0.025, 0.012),
                parameters={"detent_states": (-0.35, 0.35)},
            )
        )
        parameters.update(
            terminals=[(0.20, 0.05, 0.01), (0.28, 0.05, 0.01)],
            graph=[("source", "a", "connector_0"), ("a", "b", "switch"), ("b", "load", "connector_1")],
            initial_joints={"switch_angle": -0.35},
        )

    elif name in ("catch_ball", "roll_ball"):
        radius = 0.012 if name == "catch_ball" else 0.05
        body(
            "ball",
            [ShapeSpec("sphere", (radius,))],
            (0.14, -0.07, radius),
            mass=0.012 if name == "catch_ball" else 0.12,
        )
        if name == "catch_ball":
            speed = rng.uniform(0.07, 0.14)
            initial["ball"].linear_velocity = (speed, 0.04, 0)
            parameters.update(capture_center=(0.27, 0, 0.018), capture_radius=0.035, ball_radius=radius)
        else:
            # Real shallow pocket with an entrance ramp and three containment walls.
            shapes = tray(0.13, 0.13, 0.025)
            shapes.pop(4)  # remove the -X wall; replace with a traversable ramp/lip
            ramp = prism([(-0.04, -0.005), (0.04, -0.005), (0.04, 0.005)], 0.13)
            ramp.orientation = (2**-0.5, 0, 0, 2**-0.5)
            ramp.position = (-0.085, 0, 0.005)
            shapes.append(ramp)
            body("pocket", shapes, (0.29, 0.04, 0), True)
            parameters.update(
                center=(0.29, 0.04, 0.053),
                pocket_half_extent=0.06,
                ball_radius=radius,
                max_gripper_width=0.065,
                max_roll_center_height=radius + 0.014,
                support_clearance=0.003,
            )
            if variant == "floor":
                parameters["support_height_m"] = 0.04
            elif variant != "table":
                raise ValueError("roll_ball variants are floor and table")
    validation = ["contact stability", "robot reachability", "backend execution"]
    if family in ("cable", "cloth", "plastic_shell"):
        validation += ["self collision", "two-way robot contact"]
    if name == "barrel_jack":
        validation += ["orientation-transferring endpoint attachment"]
    if name == "paper_airplane":
        validation += ["persistent crease forces", "complete fold sequence"]
    return TaskDefinition(
        name,
        variant,
        family,
        bodies,
        joints,
        deformables,
        initial,
        goal,
        parameters,
        horizon_seconds=horizon,
        required_validation=tuple(validation),
    )
