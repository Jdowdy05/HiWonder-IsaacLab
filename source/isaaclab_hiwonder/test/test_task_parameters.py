# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commanded task settings alter actual geometry, goals and outcome contracts."""

import numpy as np
import pytest
from isaaclab_hiwonder.sim.reachability import task_landmarks
from isaaclab_hiwonder.tasks import ActorState, SceneState, TaskEpisode, build_task


def test_configured_color_stack_builds_and_accepts_only_the_requested_order():
    colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1), (0.2, 0.4, 0.6)]
    task = build_task("cube_stack", task_parameters={"count": 4, "colors": colors, "order": [3, 1, 0, 2]})
    assert len(task.bodies) == 4
    assert [body.color for body in task.bodies] == colors
    np.testing.assert_allclose(task.goal, [4, *colors[3], *colors[1], *colors[0], *colors[2]])
    state = SceneState(actors=dict(task.initial))
    previous = "table"
    for index, label in enumerate(("cube_3", "cube_1", "cube_0", "cube_2")):
        state.actors[label] = ActorState((0.25, 0, 0.0125 + 0.025 * index))
        state.contacts.add((label, previous))
        previous = label
    assert TaskEpisode(task).evaluate(state, 0.6).success
    state.actors["cube_0"], state.actors["cube_2"] = state.actors["cube_2"], state.actors["cube_0"]
    assert not TaskEpisode(task).evaluate(state, 0.6).success


def test_six_cube_goal_top_is_an_explicit_reachability_candidate():
    task = build_task("cube_stack", task_parameters={"count": 6})
    points = task_landmarks(task)
    assert any(np.allclose(point, (0.25, 0, 0.1375)) for point in points)
    assert len(task.goal) == 19


@pytest.mark.parametrize(
    "name,options,joint,target,start,position",
    [
        ("knob", {"target_angle_rad": 0.0}, "knob_angle", 0.0, -1.2, (0.25, 0.03, 0.025)),
        ("switch", {"target_state": "off"}, "switch_angle", -0.35, 0.35, (0.25, 0.03, 0.022)),
        ("latch", {"target_state": "released"}, "bolt_travel", 0.0, 0.022, (0.25, 0.03, 0.02)),
    ],
)
def test_configured_mechanism_goal_has_a_nontrivial_start_and_physical_terminal(
    name, options, joint, target, start, position
):
    task = build_task(name, task_parameters=options)
    assert task.parameters["initial_joints"][joint] == start
    np.testing.assert_allclose(task.goal, [target])
    state = SceneState(actors=dict(task.initial), joint_positions={joint: target}, joint_velocities={joint: 0})
    label = {"knob": "knob", "switch": "switch", "latch": "bolt"}[name]
    orientation = (0, np.sin(target / 2), 0, np.cos(target / 2)) if name == "switch" else (0, 0, 0, 1)
    state.actors[label] = ActorState(position, orientation)
    assert TaskEpisode(task).evaluate(state, 0.6).success
    state.joint_positions[joint] = start
    if name == "latch":
        state.actors[label] = ActorState((0.272, 0.03, 0.02))
    elif name == "knob":
        state.actors[label].orientation = (0, 0, np.sin(start / 2), np.cos(start / 2))
    else:
        state.actors[label].orientation = (0, np.sin(start / 2), 0, np.cos(start / 2))
    assert not TaskEpisode(task).evaluate(state, 0.6).success


@pytest.mark.parametrize(
    "name,options",
    [
        ("cube_stack", {"count": True}),
        ("cube_stack", {"count": 7}),
        ("cube_stack", {"colors": [[1, 0, 0], [1, 0, 0]]}),
        ("cube_stack", {"count": 3, "colors": [[1, 0, 0], [0, 1, 0]]}),
        ("cube_stack", {"order": [0, 0, 1]}),
        ("knob", {"target_angle_rad": np.nan}),
        ("knob", {"target_angle_rad": 3}),
        ("switch", {"target_state": "middle"}),
        ("latch", {"target_state": "middle"}),
        ("peg_insert", {"count": 2}),
    ],
)
def test_invalid_task_commands_are_rejected_before_scene_construction(name, options):
    with pytest.raises(ValueError):
        build_task(name, task_parameters=options)


def test_task_commands_round_trip_in_json_profiles_and_reject_unused_keys(tmp_path):
    from isaaclab_hiwonder.sim.profiles import load_profile, profile_from_dict, save_profile

    profile = profile_from_dict({"task": "cube_stack", "task_parameters": {"count": 4, "order": [3, 1, 0, 2]}})
    path = tmp_path / "stack.json"
    save_profile(profile, path)
    restored = load_profile(path)
    assert restored.task_parameters == {"count": 4, "order": [3, 1, 0, 2]}
    with pytest.raises(ValueError, match="Unsupported"):
        profile_from_dict({"task": "cube_stack", "task_parameters": {"colour": [1, 0, 0]}})
    with pytest.raises(ValueError, match="JSON|finite"):
        profile_from_dict({"task": "knob", "task_parameters": {"target_angle_rad": float("nan")}})


def test_invalid_goal_landmark_is_rejected_before_reachability_solving():
    task = build_task("cube_stack")
    task.parameters["goal_landmarks"] = [(0.25, 0, np.inf)]
    with pytest.raises(ValueError, match="finite 3D"):
        task_landmarks(task)
