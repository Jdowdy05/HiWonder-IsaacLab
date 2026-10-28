# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from isaaclab_hiwonder.assets.kinematics import forward_kinematics
from isaaclab_hiwonder.sim.reachability import reject_unreachable_bases, solve_position


def test_workspace_solver_respects_mount_transform_and_limits():
    base = np.array([0.03, -0.02, 0.75])
    yaw = np.pi / 2
    target = np.array([0.20, 0.0, 0.80])
    witness = solve_position(target, base, yaw)
    assert witness is not None
    cad = forward_kinematics(witness)["tool_center"][:3, 3]
    world = base + np.array([-cad[1], cad[0], cad[2]])
    assert np.linalg.norm(world - target) < 0.004
    assert solve_position([1, 0, 0], base, yaw) is None


def test_fixed_unreachable_task_is_rejected_without_extra_dofs():
    from dataclasses import replace

    from isaaclab_hiwonder.config import GroupControl, RandomizationConfig
    from isaaclab_hiwonder.randomization import RandomizationManager
    from isaaclab_hiwonder.tasks import build_task

    config = RandomizationConfig()
    config = replace(config, controls=replace(config.controls, base=GroupControl(enabled=False)))
    manager = RandomizationManager(1, config)
    task = build_task("peg_insert")
    task.initial["peg"].position = np.array([1.0, 0, 0.01])
    with pytest.raises(ValueError, match="Unreachable fixed base"):
        reject_unreachable_bases(manager, [task])


def test_initial_rope_base_overlap_is_rejected_even_when_endpoints_are_reachable():
    """Reproduce the first-step native trace's base and original thread centerline."""
    from dataclasses import replace

    from isaaclab_hiwonder.config import GroupControl, RandomizationConfig
    from isaaclab_hiwonder.randomization import RandomizationManager
    from isaaclab_hiwonder.tasks import build_task

    config = RandomizationConfig()
    config = replace(
        config,
        base=replace(config.base, x_m=(0.0989, 0.0989), y_m=(-0.0861, -0.0861), yaw_radians=(0, 0)),
        controls=replace(config.controls, base=GroupControl(frozen=True)),
    )
    manager = RandomizationManager(1, config)
    task = build_task("rope_thread", "thread")
    vertices = task.deformables[0].vertices
    vertices[:] = np.column_stack(
        (np.linspace(0.12, 0.33, len(vertices)), np.full(len(vertices), -0.07), np.full(len(vertices), 0.003))
    )
    with pytest.raises(ValueError, match="Initial task/base overlap"):
        reject_unreachable_bases(manager, [task])

    # The same reachable material at a clear staging location is accepted.
    vertices[:, 0] = 0.22
    vertices[:, 1] = np.linspace(-0.105, 0.105, len(vertices))
    assert len(reject_unreachable_bases(manager, [task])[0]) > 0


def test_loose_object_jitter_is_reserved_but_elevated_grasp_is_not_rejected():
    from dataclasses import replace

    from isaaclab_hiwonder.config import GroupControl, RandomizationConfig
    from isaaclab_hiwonder.randomization import RandomizationManager
    from isaaclab_hiwonder.tasks import build_task

    config = RandomizationConfig()
    config = replace(config, controls=replace(config.controls, base=GroupControl(enabled=False)))
    manager = RandomizationManager(1, config)
    task = build_task("peg_insert")
    # Nominal peg is outside the yawed base box, but the allowed -8 mm X
    # perturbation can put its side into the base. Reserve the whole bank.
    task.initial["peg"].position = np.array([0.045, 0, 0.02])
    with pytest.raises(ValueError, match="Initial task/base overlap"):
        reject_unreachable_bases(manager, [task])
    task.initial["peg"].position[2] = 0.20
    assert len(reject_unreachable_bases(manager, [task])[0]) > 0


def test_base_rejection_resamples_without_changing_task_or_structural_cohort():
    from isaaclab_hiwonder.randomization import RandomizationManager
    from isaaclab_hiwonder.tasks import build_task

    manager = RandomizationManager(1, seed=0, bootstrap_attempts=32)
    before = manager.snapshot()
    task = build_task("rope_thread", "thread")
    vertices = task.deformables[0].vertices.copy()
    assert len(reject_unreachable_bases(manager, [task])[0]) > 0
    after = manager.snapshot()
    assert not np.array_equal(before["base"]["position"], after["base"]["position"])
    np.testing.assert_array_equal(vertices, task.deformables[0].vertices)
    for group in before:
        if group == "base":
            continue
        for key in before[group]:
            np.testing.assert_array_equal(before[group][key], after[group][key], err_msg=f"{group}/{key}")
