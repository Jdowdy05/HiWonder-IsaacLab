# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical outcome boundaries; these tests do not qualify simulator contact."""

import copy
from collections import Counter

import numpy as np
import pytest
from isaaclab_hiwonder.tasks import ActorState, SceneState, TaskEpisode, build_task
from isaaclab_hiwonder.tasks.api import rotation_matrix
from isaaclab_hiwonder.tasks.backend import TaskMechanics
from isaaclab_hiwonder.tasks.catalog import GARMENTS, TASK_NAMES
from isaaclab_hiwonder.tasks.geometry import annulus, garment_mesh
from isaaclab_hiwonder.tasks.mechanics import (
    PaperCreaseModel,
    PlasticHinge,
    cable_attachment_wrench,
    knot_determinant,
    ring_crossings,
    screw_reaction,
)
from isaaclab_hiwonder.tasks.runtime import cable_nodes_from_segments, link_state_from_com


def _initial_state(task):
    return SceneState(
        actors=copy.deepcopy(task.initial),
        joint_positions={
            joint.name: task.parameters.get("initial_joints", {}).get(joint.name, 0) for joint in task.joints
        },
        joint_velocities={joint.name: 0.0 for joint in task.joints},
        deformable_positions={spec.name: spec.vertices.copy() for spec in task.deformables},
        deformable_velocities={spec.name: np.zeros_like(spec.vertices) for spec in task.deformables},
    )


def _held_outcome(task, state):
    episode = TaskEpisode(task)
    for _ in range(18):
        result = episode.evaluate(state, 1 / 30)
    return result


def test_nut_container_completion_requires_every_released_supported_nut():
    task = build_task("nut_pick_place")
    state = _initial_state(task)
    assert not _held_outcome(task, state).success
    # Four disjoint12mm nuts lie on the3mm tray floor, within both inner walls.
    for index, (x, y) in enumerate(((0.25, 0.055), (0.27, 0.055), (0.25, 0.075), (0.27, 0.075))):
        label = f"nut_{index}"
        state.actors[label] = ActorState((x, y, 0.005))
        state.contacts.add((label, "container"))
    assert _held_outcome(task, state).success
    for violation in ("outside", "held", "unsupported"):
        bad = copy.deepcopy(state)
        if violation == "outside":
            bad.actors["nut_3"] = ActorState((0.30, 0.075, 0.005))
        elif violation == "held":
            bad.contacts.add(("nut_3", "left_finger"))
        else:
            bad.contacts.remove(("nut_3", "container"))
        assert not _held_outcome(task, bad).success


def test_keyed_connector_completion_requires_its_asymmetric_orientation_and_release():
    task = build_task("connector_insert")
    state = _initial_state(task)
    assert not _held_outcome(task, state).success
    state.actors["plug"] = ActorState((0.27, 0.04, 0.015))
    state.contacts = {("plug", "socket")}
    assert _held_outcome(task, state).success
    bad = copy.deepcopy(state)
    bad.actors["plug"].orientation = (0, 0, 1, 0)
    assert not _held_outcome(task, bad).success
    bad = copy.deepcopy(state)
    bad.contacts.add(("plug", "right_finger"))
    assert not _held_outcome(task, bad).success
    state.contacts.clear()
    assert not _held_outcome(task, state).success


def test_lock_completion_requires_rotated_inserted_key_and_actual_bolt_retraction():
    task = build_task("key_lock")
    state = _initial_state(task)
    assert not _held_outcome(task, state).success
    angle = 1.4
    orientation = (0, 0, np.sin(angle / 2), np.cos(angle / 2))
    travel = -0.015 * np.sin(angle)
    state.actors["cylinder"] = ActorState((0.25, 0.03, 0.026), orientation)
    state.actors["key"] = ActorState((0.25, 0.03, 0.026), orientation)
    state.actors["bolt"] = ActorState((0.285 + travel, 0.03, 0.018))
    state.joint_positions.update(lock_angle=angle, bolt_travel=travel)
    state.contacts = {("key", "cylinder")}
    assert _held_outcome(task, state).success
    for violation in ("bolt", "key_angle", "key_contact", "release"):
        bad = copy.deepcopy(state)
        if violation == "bolt":
            bad.joint_positions["bolt_travel"] = 0
            bad.actors["bolt"] = ActorState((0.285, 0.03, 0.018))
        elif violation == "key_angle":
            bad.actors["key"].orientation = (0, 0, 0, 1)
        elif violation == "key_contact":
            bad.contacts.clear()
        else:
            bad.contacts.add(("key", "left_finger"))
        assert not _held_outcome(task, bad).success


def test_barrel_completion_requires_seating_and_both_endpoint_frame_constraints():
    from scipy.spatial.transform import Rotation

    task = build_task("barrel_jack")
    state = _initial_state(task)
    assert not _held_outcome(task, state).success
    shift = np.array((0.27, 0.04, 0.015)) - state.actors["plug"].position
    state.deformable_positions["cable"] += shift
    nodes = state.deformable_positions["cable"]
    tangent = nodes[1] - nodes[0]
    tangent /= np.linalg.norm(tangent)
    axis = np.cross((0, 0, 1), tangent)
    axis /= np.linalg.norm(axis)
    frame = Rotation.from_rotvec(axis * np.arccos(tangent[2])).as_quat()
    state.actors["plug"] = ActorState((0.27, 0.04, 0.015))
    state.actors["cable/segment_0"] = ActorState((nodes[0] + nodes[1]) / 2, frame)
    state.contacts = {("plug", "socket")}
    assert _held_outcome(task, state).success
    for violation in ("attachment_position", "attachment_angle", "seat_contact", "release"):
        bad = copy.deepcopy(state)
        if violation == "attachment_position":
            bad.deformable_positions["cable"] += (0, 0, 0.003)
        elif violation == "attachment_angle":
            bad.actors["cable/segment_0"].orientation = (
                Rotation.from_quat(frame) * Rotation.from_euler("z", 0.2)
            ).as_quat()
        elif violation == "seat_contact":
            bad.contacts.clear()
        else:
            bad.contacts.add(("plug", "right_finger"))
        assert not _held_outcome(task, bad).success


def test_ball_capture_requires_two_jaws_retention_speed_and_capture_region():
    task = build_task("catch_ball")
    state = _initial_state(task)
    assert not _held_outcome(task, state).success
    state.actors["ball"] = ActorState((0.27, 0, 0.018))
    state.contacts = {("ball", "left_finger"), ("ball", "right_finger")}
    assert _held_outcome(task, state).success
    for violation in ("one_jaw", "moving", "region"):
        bad = copy.deepcopy(state)
        if violation == "one_jaw":
            bad.contacts.remove(("ball", "right_finger"))
        elif violation == "moving":
            bad.actors["ball"].linear_velocity = (0.02, 0, 0)
        else:
            bad.actors["ball"].position = (0.31, 0, 0.018)
        assert not _held_outcome(task, bad).success


@pytest.mark.parametrize("garment", GARMENTS)
def test_garment_fold_completion_rejects_flat_translated_and_unreleased_surfaces(garment):
    task = build_task("cloth_fold", garment)
    state = _initial_state(task)
    assert not _held_outcome(task, state).success
    spec = task.deformables[0]
    nodes = spec.vertices.copy()
    midpoint = (nodes[:, 0].min() + nodes[:, 0].max()) / 2
    moving = nodes[:, 0] > midpoint
    back = nodes[:, 2] > nodes[:, 2].min() + spec.radius
    nodes[moving, 0] = 2 * midpoint - nodes[moving, 0]
    # Model four ordered fabric layers, each separated by one full thickness.
    # Folded front/back order reverses, preventing interpenetrating garment panels.
    layer = np.where(moving, np.where(back, 2, 3), np.where(back, 1, 0))
    if garment in ("towel", "handtowel", "rag"):
        layer = moving.astype(int)
    nodes[:, 2] = spec.radius + 2 * spec.radius * layer
    state.deformable_positions["sheet"] = nodes
    assert _held_outcome(task, state).success
    translated = copy.deepcopy(state)
    translated.deformable_positions["sheet"] += (0.03, 0, 0)
    assert not _held_outcome(task, translated).success
    strained = copy.deepcopy(state)
    strained.deformable_positions["sheet"][0, 2] += 0.012
    strained_result = _held_outcome(task, strained)
    assert strained_result.metrics["fold_rms_error"] < 0.012
    assert strained_result.metrics["max_edge_strain"] > 0.2
    assert not strained_result.success
    state.contacts.add(("sheet", "left_finger"))
    assert not _held_outcome(task, state).success


def _knotted_rope(variant, spec):
    """Analytic trefoil/figure-eight arcs with length-preserving discretized tails.

    Expected knot types come from the analytic curves, not the production knot
    classifier. Walking fixed chord lengths preserves every actual rod length.
    """
    from scipy.spatial.transform import Rotation

    t = np.linspace(0.14, 2 * np.pi - 0.14, 3001)
    if variant == "overhand":
        curve = np.column_stack(
            [(2 + np.cos(3 * t)) * np.cos(2 * t), (2 + np.cos(3 * t)) * np.sin(2 * t), np.sin(3 * t)]
        )
        scale = 0.007153846153846154
    else:
        curve = np.column_stack(
            [(2 + np.cos(2 * t)) * np.cos(3 * t), (2 + np.cos(2 * t)) * np.sin(3 * t), np.sin(4 * t)]
        )
        scale = 0.005384615384615385
    curve = curve @ Rotation.from_euler("xyz", (0.4, 0.7, 0.2)).as_matrix().T
    curve = curve * scale + (0.23, 0, 0.045)
    curve = np.concatenate(
        [
            np.linspace((0.13, 0, 0.0015), curve[0], 1001)[:-1],
            curve,
            np.linspace(curve[-1], (0.38, 0, 0.0015), 1501)[1:],
        ]
    )
    nodes, index = [curve[0]], 0
    lengths = np.linalg.norm(np.diff(spec.vertices, axis=0), axis=1)
    for length in lengths:
        index += np.flatnonzero(np.linalg.norm(curve[index:] - nodes[-1], axis=1) >= length)[0]
        lo, hi = curve[index - 1], curve[index]
        vector, offset = hi - lo, lo - nodes[-1]
        roots = np.roots([vector @ vector, 2 * vector @ offset, offset @ offset - length**2])
        fraction = next(root.real for root in roots if abs(root.imag) < 1e-8 and -1e-8 <= root.real <= 1 + 1e-8)
        nodes.append(lo + fraction * vector)
    nodes = np.asarray(nodes)
    np.testing.assert_allclose(np.linalg.norm(np.diff(nodes, axis=0), axis=1), lengths, atol=1e-12)
    assert nodes[:, 2].min() >= spec.radius
    assert (
        min(
            _segment_distance(a, b, c, d)
            for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:]))
            for j, (c, d) in enumerate(zip(nodes[:-1], nodes[1:]))
            if j > i + 1
        )
        > 2 * spec.radius
    )
    return nodes


def _segment_distance(a, b, c, d):
    """Independent convex segment-clearance check for finite-radius fixtures."""
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = u @ u, u @ v, v @ v, u @ w, v @ w
    candidates = [
        (np.clip(-dd / aa, 0, 1), 0),
        (np.clip((bb - dd) / aa, 0, 1), 1),
        (0, np.clip(ee / cc, 0, 1)),
        (1, np.clip((bb + ee) / cc, 0, 1)),
    ]
    determinant = aa * cc - bb * bb
    if determinant > 1e-20:
        s, t = (bb * ee - cc * dd) / determinant, (aa * ee - bb * dd) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    return min(np.linalg.norm(w + s * u - t * v) for s, t in candidates)


@pytest.mark.parametrize("variant", ("thread", "overhand", "figure_eight"))
def test_rope_completion_requires_requested_topology_release_and_material_integrity(variant):
    task = build_task("rope_thread", variant)
    state = _initial_state(task)
    assert not _held_outcome(task, state).success
    spec = task.deformables[0]
    if variant == "thread":
        nodes = np.column_stack((np.linspace(0.12, 0.33, 41), np.full(41, 0.04), np.full(41, 0.04)))
    else:
        nodes = _knotted_rope(variant, spec)
    state.deformable_positions["rope"] = nodes
    result = _held_outcome(task, state)
    assert result.success
    if variant != "thread":
        assert result.metrics["knot_determinant"] == (3 if variant == "overhand" else 5)
    held = copy.deepcopy(state)
    held.contacts.add(("rope", "right_finger"))
    assert not _held_outcome(task, held).success
    wrong = copy.deepcopy(state)
    if variant == "thread":
        wrong.deformable_positions["rope"] += (0, 0.04, 0)
    else:
        other = "figure_eight" if variant == "overhand" else "overhand"
        wrong.deformable_positions["rope"] = _knotted_rope(other, spec)
    assert not _held_outcome(task, wrong).success
    stretched = copy.deepcopy(state)
    stretched.deformable_positions["rope"] = nodes[0] + 1.2 * (nodes - nodes[0])
    assert not _held_outcome(task, stretched).success
    if variant != "thread":
        displaced = copy.deepcopy(state)
        displaced.deformable_positions["rope"] -= (0.03, 0, 0)
        assert not _held_outcome(task, displaced).success


def test_insertion_requires_depth_release_and_continuous_stability():
    task = build_task("peg_insert")
    state = SceneState(actors=dict(task.initial))
    target = np.asarray(task.parameters["seat"])
    state.actors["peg"] = ActorState(target + [0, 0, 0.012])
    episode = TaskEpisode(task)
    assert not episode.evaluate(state, 0.6).success
    state.actors["peg"] = ActorState(target)
    state.contacts = {("peg", "socket"), ("peg", "gripper")}
    assert not episode.evaluate(state, 0.6).success
    state.contacts = {("peg", "socket")}
    assert not episode.evaluate(state, 0.1).success
    state.actors["peg"] = ActorState(target, linear_velocity=(0.1, 0, 0))
    assert not episode.evaluate(state, 0.6).success
    state.actors["peg"] = ActorState(target)
    assert not episode.evaluate(state, 0.3).success
    assert episode.evaluate(state, 0.3).success


def test_stack_rejects_floating_aligned_cubes_without_support_contacts():
    task = build_task("cube_stack")
    state = SceneState(actors=dict(task.initial))
    for index in range(3):
        state.actors[f"cube_{index}"] = ActorState((0.25, 0, 0.0125 + 0.025 * index))
    assert not TaskEpisode(task).evaluate(state, 1).success
    state.contacts = {("cube_0", "table"), ("cube_1", "cube_0"), ("cube_2", "cube_1")}
    assert TaskEpisode(task).evaluate(state, 1).success


@pytest.mark.parametrize(
    "position,contacts",
    [
        ((0.20, 0, 0.05), {("ball", "left_finger"), ("ball", "right_finger"), ("ball", "table")}),
        ((0.20, 0, 0.06), {("ball", "left_finger")}),
        ((0.20, 0, 0.08), set()),
    ],
)
def test_roll_only_rejects_grasp_and_carry_history_even_after_deposit(position, contacts):
    import copy

    task = build_task("roll_ball")
    episode = TaskEpisode(task)
    state = SceneState(actors=dict(task.initial), contacts=contacts)
    state.actors["ball"] = ActorState(position)
    assert episode.evaluate(state, 1 / 30).failure
    restored = copy.deepcopy(episode)
    state.actors["ball"] = ActorState((0.29, 0.04, 0.053))
    state.contacts = {("ball", "pocket")}
    result = restored.evaluate(state, 0.6)
    assert result.failure and not result.success
    assert TaskEpisode(task).evaluate(state, 0.6).success


def test_roll_only_allows_supported_push_and_ramp_entry():
    task = build_task("roll_ball")
    episode = TaskEpisode(task)
    state = SceneState(actors=dict(task.initial), contacts={("ball", "table"), ("ball", "left_finger")})
    assert not episode.evaluate(state, 1 / 30).failure
    # The authored entrance ramp rises10mm above the support plane.
    state.actors["ball"] = ActorState((0.24, 0.04, 0.060))
    state.contacts = {("ball", "pocket"), ("ball", "left_finger")}
    assert not episode.evaluate(state, 1 / 30).failure
    state.actors["ball"] = ActorState((0.29, 0.04, 0.053))
    state.contacts = {("ball", "pocket")}
    assert episode.evaluate(state, 0.6).success


def test_rope_passage_requires_crossing_inside_ring_aperture():
    assert ring_crossings(np.array([[-1, 0, 0], [1, 0, 0]]), (0, 0, 0), (1, 0, 0), 0.1) == 1
    assert ring_crossings(np.array([[-1, 0.2, 0], [1, 0.2, 0]]), (0, 0, 0), (1, 0, 0), 0.1) == 0
    assert ring_crossings(np.array([[-1, 0, 0], [-0.1, 0, 0]]), (0, 0, 0), (1, 0, 0), 0.1) == 0


def test_plug_attachment_transfers_orientation_and_paper_retains_crease():
    forces, torque = cable_attachment_wrench(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]),
        np.zeros((3, 3)),
        ActorState((0, 0, 0), (0, 0, 2**-0.5, 2**-0.5)),
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]),
        1000.0,
        1.0,
    )
    assert np.linalg.norm(torque) > 0.1
    assert np.linalg.norm(forces) > 1
    hinge = PlasticHinge(stiffness=0.2, yield_angle=0.2, hardening=0.0)
    hinge.advance(1.0)
    assert abs(hinge.rest_angle - 0.8) < 1e-9
    assert abs(hinge.advance(0.8)) < 1e-9
    crease = PlasticHinge(stiffness=0.2, yield_angle=0.2, hardening=0.0)
    crease.advance(np.pi - 0.01)
    before = crease.accumulated_plastic
    crease.advance(-np.pi + 0.01)
    assert crease.accumulated_plastic - before < 0.03


def test_catalog_resets_do_not_fabricate_completion():
    for name in TASK_NAMES:
        task = build_task(name, seed=22)
        state = SceneState(
            actors=dict(task.initial),
            joint_positions={j.name: task.parameters.get("initial_joints", {}).get(j.name, 0) for j in task.joints},
            joint_velocities={j.name: 0.0 for j in task.joints},
            deformable_positions={d.name: d.vertices.copy() for d in task.deformables},
            deformable_velocities={d.name: np.zeros_like(d.vertices) for d in task.deformables},
            crease_rest_angles={d.name: np.zeros(len(d.parameters.get("hinges", []))) for d in task.deformables},
        )
        forces = TaskMechanics(task).forces(state)
        outcome = TaskEpisode(task).evaluate(state, 1.0)
        assert not outcome.success, name
        assert np.isfinite(outcome.reward), name
        assert all(np.all(np.isfinite(f)) for f in forces.particle_forces.values()), name


def test_ring_convex_colliders_leave_a_real_aperture():
    # Independent geometric fact: the minimum projection of each convex wedge
    # onto its radial bisector stays outside the clear bore; convex hull cannot fill it.
    for shape in annulus(0.01, 0.02, 0.005):
        bisector = shape.vertices[:, :2].mean(axis=0)
        bisector /= np.linalg.norm(bisector)
        assert np.min(shape.vertices[:, :2] @ bisector) > 0.0099


def test_knot_detector_distinguishes_trefoil_figure_eight_and_circle():
    t = np.linspace(0, 2 * np.pi, 181)
    trefoil = np.column_stack([(2 + np.cos(3 * t)) * np.cos(2 * t), (2 + np.cos(3 * t)) * np.sin(2 * t), np.sin(3 * t)])
    eight = np.column_stack([(2 + np.cos(2 * t)) * np.cos(3 * t), (2 + np.cos(2 * t)) * np.sin(3 * t), np.sin(4 * t)])
    circle = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    assert knot_determinant(trefoil) == 3
    assert knot_determinant(eight) == 5
    assert knot_determinant(circle) == 1


def test_screw_reactions_are_reciprocal_and_paper_forces_conserve_translation():
    force, torque = screw_reaction(0.3, 0.002, 0, 0, pitch=0.003)
    assert force < 0 and torque > 0
    assert abs(force * 0.003 / (2 * np.pi) + torque) < 1e-12
    forces = PaperCreaseModel([[0, 1, 2, 3]]).forces(
        np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0.008]])
    )
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=2e-7)


def test_garments_are_connected_surfaces_with_actual_openings():
    for kind in GARMENTS:
        vertices, edges, faces = garment_mesh(kind)
        assert (
            np.min(
                np.linalg.norm(
                    np.cross(
                        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
                    ),
                    axis=1,
                )
            )
            > 0
        )
        adjacency = {i: set() for i in range(len(vertices))}
        for a, b in edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        seen = {0}
        frontier = [0]
        while frontier:
            for neighbor in adjacency[frontier.pop()] - seen:
                seen.add(neighbor)
                frontier.append(neighbor)
        assert len(seen) == len(vertices), kind
        if kind in ("shirt", "shorts", "underwear", "sock"):
            assert np.ptp(vertices[:, 2]) > 0, kind
            edge_counts = Counter(
                tuple(sorted((int(a), int(b)))) for face in faces for a, b in zip(face, np.roll(face, -1))
            )
            boundary = {}
            for (a, b), count in edge_counts.items():
                if count == 1:
                    boundary.setdefault(a, set()).add(b)
                    boundary.setdefault(b, set()).add(a)
            seen = set()
            openings = 0
            for root in boundary:
                if root in seen:
                    continue
                openings += 1
                frontier = [root]
                seen.add(root)
                while frontier:
                    for neighbor in boundary[frontier.pop()] - seen:
                        seen.add(neighbor)
                        frontier.append(neighbor)
            assert openings == {"shirt": 4, "shorts": 3, "underwear": 3, "sock": 1}[kind]
        assert len(faces) > 30, kind


def test_cable_state_uses_com_offsets_and_endpoint_angular_velocity():
    rotation = (0, 2**-0.5, 0, 2**-0.5)
    poses = np.array([[0.01, 0, 0, *rotation], [0.03, 0, 0, *rotation]])
    velocities = np.array([[0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 2]], float)
    nodes, speed = cable_nodes_from_segments(poses, velocities, [0.02, 0.02])
    np.testing.assert_allclose(nodes, [[0, 0, 0], [0.02, 0, 0], [0.04, 0, 0]], atol=1e-9)
    np.testing.assert_allclose(speed, [[0, -0.02, 0], [0, 0, 0], [0, 0.02, 0]], atol=1e-9)


def test_live_rigid_state_converts_com_velocity_to_actor_origin():
    pose = np.array([[5, 6, 7, 0, 0, 2**-0.5, 2**-0.5]])
    velocity = np.array([[1.0, 2, 3, 0, 0, 2]])
    state = link_state_from_com(pose, velocity, np.array([[0.1, 0, 0]]))
    np.testing.assert_allclose(state[0, 7:], [1.2, 2, 3, 0, 0, 2], atol=1e-9)
    np.testing.assert_array_equal(velocity, [[1, 2, 3, 0, 0, 2]])


def test_paper_sequence_accumulates_dwell_at_control_rate_before_final_release():
    task = build_task("paper_airplane")
    episode = TaskEpisode(task)
    state = SceneState(actors={}, deformable_velocities={"sheet": np.zeros_like(task.deformables[0].vertices)})
    for stage, target in enumerate(task.parameters["fold_stages"]):
        state.deformable_positions["sheet"] = target.copy()
        rest = np.zeros(len(task.parameters["hinges"]))
        rest[task.parameters["crease_masks"][stage]] = 2.0
        state.crease_rest_angles["sheet"] = rest
        for _ in range(16):
            outcome = episode.evaluate(state, 1 / 30)
        if stage < len(task.parameters["fold_stages"]) - 1:
            assert episode.stage == stage + 1
            assert not outcome.success
    assert outcome.success


def test_washer_seated_configuration_clears_table_and_ring_stand():
    task = build_task("washer_ring")
    quaternion = (0, 2**-0.5, 0, 2**-0.5)
    washer = next(b for b in task.bodies if b.name == "washer")
    points = (
        np.concatenate([s.vertices for s in washer.shapes]) @ rotation_matrix(quaternion).T + task.parameters["seat"]
    )
    assert points[:, 2].min() > 0.01
    lo, hi = points.min(axis=0), points.max(axis=0)
    ring = next(b for b in task.bodies if b.name == "ring")
    for shape in ring.shapes:
        if shape.kind == "box":
            center = np.asarray(task.initial["ring"].position) + shape.position
            half = np.asarray(shape.dimensions) / 2
            assert np.any(hi < center - half) or np.any(lo > center + half)
    state = SceneState(actors=dict(task.initial), contacts={("washer", "ring")})
    state.actors["washer"] = ActorState(task.parameters["seat"], quaternion)
    assert TaskEpisode(task).evaluate(state, 0.6).success


def test_shape_sort_accepts_physical_rotational_symmetry_but_rejects_wrong_slots():
    task = build_task("shape_sort")
    state = SceneState(actors=dict(task.initial))
    for index, (label, target) in enumerate(task.parameters["targets"].items()):
        angle = (np.pi / 2, 2 * np.pi / 3, np.pi / 3)[index]
        state.actors[label] = ActorState(target, (0, 0, np.sin(angle / 2), np.cos(angle / 2)))
        state.contacts.add((label, f"slot_{index}"))
    assert TaskEpisode(task).evaluate(state, 0.6).success
    state.actors["shape_0"].position, state.actors["shape_1"].position = (
        state.actors["shape_1"].position,
        state.actors["shape_0"].position,
    )
    assert not TaskEpisode(task).evaluate(state, 0.6).success


def test_unloaded_latch_and_verified_gear_mesh_remain_engaged_without_contact_load():
    task = build_task("latch")
    state = SceneState(
        actors=dict(task.initial), joint_positions={"bolt_travel": 0.022}, joint_velocities={"bolt_travel": 0}
    )
    state.actors["bolt"] = ActorState((0.272, 0.03, 0.02))
    assert TaskEpisode(task).evaluate(state, 0.6).success
    state.actors["bolt"].position = (0.26, 0.04, 0.02)
    assert not TaskEpisode(task).evaluate(state, 0.6).success
    task = build_task("gear_assembly")
    state = SceneState(actors=dict(task.initial), joint_positions={"drive_angle": 0}, contacts={("gear", "drive_gear")})
    episode = TaskEpisode(task)
    for drive in np.linspace(0, 0.25, 6):
        angle = np.pi / 16 - drive
        state.joint_positions["drive_angle"] = drive
        state.actors["gear"] = ActorState((0.264, 0.04, 0.006), (0, 0, np.sin(angle / 2), np.cos(angle / 2)))
        episode.evaluate(state, 0.1)
    state.contacts = set()
    assert episode.evaluate(state, 0.6).success


def test_authored_joint_names_match_runtime_contract(tmp_path):
    from isaaclab_hiwonder.tasks.backend import author_rigid_usd

    from pxr import Usd, UsdPhysics

    task = build_task("key_lock")
    path = tmp_path / "key_lock.usda"
    author_rigid_usd(task, path)
    stage = Usd.Stage.Open(str(path))
    actual = {prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)}
    assert {joint.name for joint in task.joints}.issubset(actual)


def test_task_usd_drive_gains_preserve_si_units_after_native_import(tmp_path):
    import newton
    from isaaclab_hiwonder.tasks.backend import author_rigid_usd

    from pxr import Usd

    for name in ("knob", "latch"):
        task = build_task(name)
        task.joints[0].stiffness = 0.03
        task.joints[0].damping = 0.01
        path = tmp_path / f"{name}.usda"
        author_rigid_usd(task, path)
        builder = newton.ModelBuilder()
        builder.add_usd(Usd.Stage.Open(str(path)))
        np.testing.assert_allclose(builder.joint_target_ke, [0.03], rtol=1e-6)
        np.testing.assert_allclose(builder.joint_target_kd, [0.01], rtol=1e-6)


def test_collider_leaf_names_do_not_alias_actor_contact_patterns(tmp_path):
    from isaaclab_hiwonder.tasks.backend import author_rigid_usd

    from pxr import Usd, UsdPhysics

    task = build_task("shape_sort")
    path = tmp_path / "sorter.usda"
    author_rigid_usd(task, path)
    stage = Usd.Stage.Open(str(path))
    colliders = {prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)}
    assert not colliders.intersection(body.name for body in task.bodies)


def test_circuit_connector_goal_clears_socket_rim_and_seats_on_floor():
    task = build_task("circuit_restore")
    state = SceneState(
        actors=dict(task.initial), joint_positions={"switch_angle": 0.35}, joint_velocities={"switch_angle": 0}
    )
    for index, target in enumerate(task.parameters["terminals"]):
        connector = next(body for body in task.bodies if body.name == f"connector_{index}")
        terminal = next(body for body in task.bodies if body.name == f"terminal_{index}")
        collar = next(shape for shape in connector.shapes if shape.kind == "box")
        stem = next(shape for shape in connector.shapes if shape.kind == "mesh")
        rim = max(shape.vertices[:, 2].max() + shape.position[2] for shape in terminal.shapes if shape.kind == "mesh")
        collar_bottom = target[2] + collar.position[2] - collar.dimensions[2] / 2
        assert collar_bottom >= task.initial[f"terminal_{index}"].position[2] + rim - 1e-9
        floor = terminal.shapes[-1]
        floor_top = task.initial[f"terminal_{index}"].position[2] + floor.position[2] + floor.dimensions[2] / 2
        assert abs(target[2] + stem.vertices[:, 2].min() - floor_top) < 1e-9
        state.actors[f"connector_{index}"] = ActorState(target)
        state.contacts.add((f"connector_{index}", f"terminal_{index}"))
    assert TaskEpisode(task).evaluate(state, 0.6).success


def test_key_convex_parts_preserve_the_notch_and_both_l_profile_branches():
    from scipy.spatial import ConvexHull

    task = build_task("key_lock")
    key = next(body for body in task.bodies if body.name == "key")
    hulls = [ConvexHull(shape.vertices).equations for shape in key.shapes if shape.kind == "mesh"]

    def contained(point):
        return any(np.all(equations[:, :3] @ point + equations[:, 3] <= 1e-10) for equations in hulls)

    assert not contained(np.array([0.0035, 0, 0]))
    assert contained(np.array([-0.001, 0, 0]))
    assert contained(np.array([0.004, 0.004, 0]))


def test_table_resting_knot_tails_enter_topology_check_but_an_unknot_still_fails():
    for variant in ("overhand", "figure_eight"):
        task = build_task("rope_thread", variant)
        spec = task.deformables[0]
        nodes = spec.vertices.copy()
        nodes[:, 2] = spec.radius
        nodes[0, :2], nodes[-1, :2] = (0.13, 0), (0.33, 0)
        state = SceneState(
            actors={}, deformable_positions={"rope": nodes}, deformable_velocities={"rope": np.zeros_like(nodes)}
        )
        result = TaskEpisode(task).evaluate(state, 1.0)
        # A genuine unknot determinant of one proves classification ran; the
        # old elevated endpoint target skipped it and returned the sentinel zero.
        assert result.metrics["knot_determinant"] == 1
        assert not result.success
        nodes[0, 0] -= 0.021
        assert TaskEpisode(task).evaluate(state, 1.0).metrics["knot_determinant"] == 0


def test_rope_control_frame_stability_preserves_motion_reset_and_checkpoint_history():
    import copy

    task = build_task("cable_untangle")
    lengths = np.linalg.norm(np.diff(task.deformables[0].vertices, axis=0), axis=1)
    base = np.column_stack(
        (0.12 + np.r_[0, np.cumsum(lengths)], np.zeros(len(lengths) + 1), np.full(len(lengths) + 1, 0.0015))
    )
    episode = TaskEpisode(task)
    dt = 1 / 30
    resumed = None
    for step in range(18):
        time = step * dt
        # A real small-amplitude oscillation has high instantaneous velocity,
        # but negligible displacement over the controller's observation frame.
        phase = 2 * np.pi * 190 * time
        nodes = base + [0, 0.00005 * np.sin(phase), 0]
        velocity = np.tile([0, 0.00005 * 2 * np.pi * 190 * np.cos(phase), 0], (len(nodes), 1))
        state = SceneState({}, deformable_positions={"rope": nodes}, deformable_velocities={"rope": velocity})
        result = episode.evaluate(state, dt)
        if step == 0:
            assert not result.success
        if step == 7:
            resumed = copy.deepcopy(episode)
        elif resumed is not None:
            restored = resumed.evaluate(state, dt)
            assert restored.success == result.success
            assert restored.metrics == result.metrics
    assert result.success
    assert result.metrics["max_particle_speed"] > 0.02
    assert result.metrics["max_control_frame_particle_speed"] < 0.02
    assert not TaskEpisode(task).evaluate(state, task.hold_seconds).success
    for violation in ("release", "endpoint", "straightness"):
        bad = SceneState(
            {}, deformable_positions={"rope": base.copy()}, deformable_velocities={"rope": np.zeros_like(base)}
        )
        if violation == "release":
            bad.contacts.add(("rope", "left_finger"))
        elif violation == "endpoint":
            bad.deformable_positions["rope"] -= (0.021, 0, 0)
        else:
            apex = (0.23, np.sqrt(0.11**2 - 0.10**2), 0.0015)
            bad.deformable_positions["rope"] = np.concatenate(
                [
                    np.linspace((0.13, 0, 0.0015), apex, 21)[:-1],
                    np.linspace(apex, (0.33, 0, 0.0015), 21),
                ]
            )
        assert not _held_outcome(task, bad).success
    # Genuine motion remains disqualifying while geometry is still in tolerance.
    moving = TaskEpisode(task)
    for step in range(16):
        nodes = base + [0, 0.03 * step * dt, 0]
        state = SceneState(
            {},
            deformable_positions={"rope": nodes},
            deformable_velocities={"rope": np.tile([0, 0.03, 0], (len(nodes), 1))},
        )
        assert not moving.evaluate(state, dt).success
    assert moving.held == 0
    state.deformable_velocities["rope"][-1, 0] = np.nan
    invalid = TaskEpisode(task).evaluate(state, dt)
    assert invalid.failure and invalid.metrics["nonfinite"] == 1
    state.deformable_velocities["rope"][:] = 0
    state.deformable_positions["rope"][7] = [0.49, 0, 0.003]
    assert TaskEpisode(task).evaluate(state, dt).failure
