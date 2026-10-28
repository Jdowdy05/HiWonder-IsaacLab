# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from isaaclab_hiwonder.config import (
    ConfettiConfig,
    DistractorConfig,
    GroupControl,
    RandomizationConfig,
    RobotRandomizationConfig,
)
from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.sim.reachability import reject_unreachable_bases
from isaaclab_hiwonder.sim.resets import make_reset
from isaaclab_hiwonder.sim.support import bootstrap_task_cohorts, physical_nuisance_overlaps, task_supported
from isaaclab_hiwonder.tasks import build_task
from isaaclab_hiwonder.tasks.api import rotation_matrix


def _six_cube_profile(physical=True, frozen=False):
    config = RandomizationConfig(
        robot=RobotRandomizationConfig(geometry_variants=("nominal",), variant_scales=(1.0,)),
        distractor=DistractorConfig(count=(6, 6), max_count=6, shapes=("box",), physical_probability=float(physical)),
        confetti=ConfettiConfig(count=(0, 0), max_count=0),
    )
    return replace(
        config,
        controls=replace(config.controls, table=GroupControl(frozen=True), distractor=GroupControl(frozen=frozen)),
    )


def _reset_record(task, manager):
    reject_unreachable_bases(manager, [task])
    sample = manager.snapshot()["base"]
    base = sample["position"][0].numpy().copy()
    base[2] = 0
    q = sample["quaternion_xyzw"][0].numpy()
    return make_reset(task, 1871, seed=5, base_position=base, base_yaw=2 * np.arctan2(q[2], q[3]) + np.pi / 2)


def _strict_box_penetration(cube, nuisance, index):
    """Exact independent OBB interior witness, not the production conservative bound."""
    from scipy.optimize import linprog

    ca = np.asarray(cube.position)
    ra = rotation_matrix(cube.orientation)
    cb = nuisance["position"][0, index].numpy().astype(float)
    size = nuisance["size"][0, index].numpy().astype(float)
    cb[2] = size[2] / 2 + 0.001
    rb = rotation_matrix(nuisance["quaternion_xyzw"][0, index])
    normals = np.concatenate((ra.T, -ra.T, rb.T, -rb.T))
    limits = np.r_[0.0125 + ra.T @ ca, 0.0125 - ra.T @ ca, size / 2 + rb.T @ cb, size / 2 - rb.T @ cb]
    result = linprog(
        [0, 0, 0, -1],
        A_ub=np.column_stack((normals, np.ones(12))),
        b_ub=limits,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    assert result.success
    return result.x[3]


def test_physical_bootstrap_rejects_seed5_penetration_in_actual_immutable_reset():
    config = _six_cube_profile()
    task = build_task("cube_stack", seed=5, task_parameters={"count": 6})
    before_task = copy.deepcopy(task)
    manager = RandomizationManager(1, config, seed=5, bootstrap_attempts=32)
    before = manager.snapshot()
    original = _reset_record(task, manager)
    assert _strict_box_penetration(original.definition.initial["cube_4"], before["distractor"], 1) > 0.007
    result = bootstrap_task_cohorts(manager, [task])
    after = manager.snapshot()
    record = _reset_record(task, manager)
    for actor in record.definition.initial.values():
        for index in range(6):
            assert _strict_box_penetration(actor, after["distractor"], index) < 0
    assert result["task_support_checks"][0] > 1
    for key in before["table"]:
        assert torch.equal(before["table"][key], after["table"][key])
    for group in ("robot", "room", "actuator"):
        for key in before[group]:
            assert torch.equal(before[group][key], after[group][key])
    for label, initial in before_task.initial.items():
        np.testing.assert_array_equal(task.initial[label].position, initial.position)
    assert [(b.mass, b.friction, b.color) for b in task.bodies] == [
        (b.mass, b.friction, b.color) for b in before_task.bodies
    ]


def test_visual_only_overlap_is_allowed_but_frozen_physical_overlap_rejects():
    task = build_task("cube_stack", seed=5, task_parameters={"count": 6})
    visual = RandomizationManager(1, _six_cube_profile(physical=False, frozen=True), seed=5, bootstrap_attempts=32)
    before = visual.snapshot()
    assert _strict_box_penetration(task.initial["cube_4"], before["distractor"], 1) > 0.004
    assert bootstrap_task_cohorts(visual, [task])["task_support_checks"] == [1]
    assert torch.equal(before["distractor"]["position"], visual.snapshot()["distractor"]["position"])
    frozen = RandomizationManager(1, _six_cube_profile(frozen=True), seed=5, bootstrap_attempts=32)
    with pytest.raises(ValueError, match="[Ff]rozen.*distractor|distractor.*frozen"):
        bootstrap_task_cohorts(frozen, [task])


def test_construction_clearance_reserves_loose_jitter_but_actual_records_do_not_double_it():
    config = _six_cube_profile()
    task = build_task("cube_stack", seed=5, task_parameters={"count": 6})
    manager = RandomizationManager(1, config, seed=5, bootstrap_attempts=32)
    record = _reset_record(task, manager)
    sample = manager.snapshot()
    sample["distractor"]["active"][0, :] = False
    sample["distractor"]["active"][0, 0] = True
    sample["distractor"]["size"][0, 0] = torch.tensor((0.02, 0.02, 0.02))
    cube = task.initial["cube_4"]
    sample["distractor"]["position"][0, 0, :2] = torch.tensor((cube.position[0], cube.position[1] - 0.026))
    sample["distractor"]["quaternion_xyzw"][0, 0] = torch.tensor((0, 0, 0, 1))
    assert _strict_box_penetration(cube, sample["distractor"], 0) < 0
    assert _strict_box_penetration(record.definition.initial["cube_4"], sample["distractor"], 0) > 0
    assert physical_nuisance_overlaps(task, sample, config, False) == set()
    assert physical_nuisance_overlaps(task, sample, config, True) == {"distractor"}
    assert physical_nuisance_overlaps(record.definition, sample, config, False) == {"distractor"}


@pytest.mark.parametrize("task_name,variant", [("rope_thread", "thread"), ("cloth_fold", "rag")])
def test_physical_card_clearance_uses_soft_thickness_and_support_relative_height(task_name, variant):
    config = RandomizationConfig(
        robot=RobotRandomizationConfig(geometry_variants=("nominal",), variant_scales=(1.0,)),
        distractor=DistractorConfig(count=(0, 0), max_count=0),
        confetti=ConfettiConfig(
            count=(1, 1), max_count=1, physical_budget=1, size_m=(0.012, 0.012), aspect_ratio=(1, 1)
        ),
    )
    manager = RandomizationManager(1, config, seed=11, bootstrap_attempts=32)
    sample = manager.snapshot()
    center = sample["confetti"]["position"][0, 0, :2].numpy()
    task = build_task(task_name, variant)
    spec = task.deformables[0]
    spec.vertices[:, :2] += center - spec.vertices[:, :2].mean(0)
    spec.vertices[:, 2] = 0.0015
    assert physical_nuisance_overlaps(task, sample, config, False) == {"confetti"}
    spec.vertices[:, 2] += 0.02
    assert physical_nuisance_overlaps(task, sample, config, False) == set()


def test_mechanism_initial_and_near_goal_transforms_are_both_reserved():
    from isaaclab_hiwonder.tasks.api import ActorState, BodySpec, JointSpec, ShapeSpec, TaskDefinition

    config = _six_cube_profile()
    sample = RandomizationManager(1, config, seed=5, bootstrap_attempts=32).snapshot()
    sample["distractor"]["active"][0, :] = False
    sample["distractor"]["active"][0, 0] = True
    sample["distractor"]["size"][0, 0] = torch.tensor((0.005, 0.005, 0.045))
    sample["distractor"]["quaternion_xyzw"][0, 0] = torch.tensor((0, 0, 0, 1))
    task = TaskDefinition(
        "knob",
        "default",
        "articulated",
        [
            BodySpec("support", [], static=True),
            BodySpec("handle", [ShapeSpec("box", (0.01, 0.004, 0.004), (0.04, 0, 0))]),
        ],
        [JointSpec("angle", "revolute", "support", "handle", (0, 0, 1), -2, 2, 1, (0.25, 0, 0.02))],
        [],
        {"support": ActorState((0.25, 0, 0.02)), "handle": ActorState((0.25, 0, 0.02))},
        np.array((1.2,)),
        {"joint": "angle", "target": 1.2, "initial_joints": {"angle": -1.2}},
    )
    sample["distractor"]["position"][0, 0, :2] = torch.tensor((0.25 + 0.04 * np.cos(-1.2), 0.04 * np.sin(-1.2)))
    assert physical_nuisance_overlaps(task, sample, config, False) == {"distractor"}
    near_angle = -1.2 + 0.65 * 2.4
    sample["distractor"]["position"][0, 0, :2] = torch.tensor(
        (0.25 + 0.04 * np.cos(near_angle), 0.04 * np.sin(near_angle))
    )
    assert physical_nuisance_overlaps(task, sample, config, False) == set()
    assert physical_nuisance_overlaps(task, sample, config, True) == {"distractor"}


def test_task_fixture_extents_cannot_overhang_a_smaller_circle():
    task = build_task("roll_ball")
    table = dict(width=0.8, depth=0.65, edge_radius=0.02)
    assert task_supported(task, table, "rectangle")
    assert task_supported(task, dict(table, edge_radius=0), "rounded_rectangle")
    assert not task_supported(task, table, "circle")
    assert task_supported(task, dict(width=1.2, depth=1.2, edge_radius=0.02), "circle")


def test_small_stack_table_must_support_the_commanded_goal_as_well_as_initial_cube():
    task = build_task("cube_stack", task_parameters={"count": 1})
    # The initial cube fits this225mm half-width. The250mm goal center does not.
    table = dict(width=0.45, depth=0.4, edge_radius=0)
    assert not task_supported(task, table, "rectangle")
    assert task_supported(task, dict(table, width=0.6), "rectangle")


def test_table_support_reserves_immutable_loose_object_jitter_at_the_edge():
    task = build_task("cube_stack", task_parameters={"count": 1})
    task.initial["cube_0"].position = (0.278, 0, 0.0125)
    # Nominal right face290.5mm fits inside300mm; an8mm reset perturbation does not.
    table = dict(width=0.6, depth=0.4, edge_radius=0)
    assert not task_supported(task, table, "rectangle")
    assert task_supported(task, dict(table, width=0.7), "rectangle")


def test_joint_supported_overhang_does_not_require_table_beneath_the_child():
    from isaaclab_hiwonder.tasks.api import ActorState, BodySpec, JointSpec, ShapeSpec, TaskDefinition

    task = TaskDefinition(
        "jointed_fixture",
        "default",
        "articulated",
        [
            BodySpec("root", [ShapeSpec("box", (0.02, 0.02, 0.01))], static=True),
            BodySpec("handle", [ShapeSpec("box", (0.02, 0.01, 0.01), (0.06, 0, 0))]),
        ],
        [JointSpec("angle", "revolute", "root", "handle", (0, 0, 1), -1, 1, 1, (0.20, 0, 0.02))],
        [],
        {"root": ActorState((0.20, 0, 0.005)), "handle": ActorState((0.20, 0, 0.02))},
        np.zeros(1),
        {},
    )
    # The fixed root has full support. Its joint carries the handle past the225mm edge.
    assert task_supported(task, dict(width=0.45, depth=0.3, edge_radius=0), "rectangle")


def test_factory_bootstrap_qualifies_initial_task_geometry_without_touching_other_nuisance():
    from dataclasses import replace

    import pytest
    import torch
    from isaaclab_hiwonder.config import (
        ConfettiConfig,
        DistractorConfig,
        GroupControl,
        RandomizationConfig,
        TableConfig,
    )
    from isaaclab_hiwonder.randomization import RandomizationManager
    from isaaclab_hiwonder.sim.support import bootstrap_task_cohorts

    cfg = RandomizationConfig(
        table=TableConfig(shapes=("circle",), width_m=(1.2, 1.2), depth_m=(0.65, 1.2)),
        distractor=DistractorConfig(count=(0, 0), max_count=0),
        confetti=ConfettiConfig(count=(0, 0), max_count=0),
    )
    manager = RandomizationManager(1, cfg, seed=3)
    before = manager.snapshot()
    result = bootstrap_task_cohorts(manager, [build_task("roll_ball")])
    assert result["accepted"] == [True] and result["task_support_checks"][0] > 1
    assert torch.equal(before["robot"]["geometry_scale"], manager.snapshot()["robot"]["geometry_scale"])
    assert torch.equal(before["room"]["light_phase"], manager.snapshot()["room"]["light_phase"])
    table = {key: value[0] for key, value in manager.snapshot()["table"].items()}
    assert task_supported(build_task("roll_ball"), table, "circle")
    frozen = replace(cfg, controls=replace(cfg.controls, table=GroupControl(frozen=True)))
    with pytest.raises(ValueError, match="support"):
        bootstrap_task_cohorts(RandomizationManager(1, frozen, seed=3), [build_task("roll_ball")])
