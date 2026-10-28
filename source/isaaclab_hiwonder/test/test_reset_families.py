# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from isaaclab_hiwonder.sim.resets import make_reset
from isaaclab_hiwonder.tasks import build_task


def test_cohort_bank_entry_is_immutable_and_near_goal_is_distinct():
    definition = build_task("peg_insert", seed=2)
    reaching = make_reset(definition, 0, seed=17, cohort_count=2)
    near = make_reset(definition, 2, seed=17, cohort_count=2)
    repeated = make_reset(definition, 2, seed=17, cohort_count=2)
    assert reaching.family == "reaching" and near.family == "near_goal"
    assert np.array_equal(near.robot_q, repeated.robot_q)
    assert np.array_equal(near.definition.initial["peg"].position, repeated.definition.initial["peg"].position)
    assert not np.array_equal(near.definition.initial["peg"].position, reaching.definition.initial["peg"].position)
    assert np.linalg.norm(near.definition.initial["peg"].position - near.definition.parameters["seat"]) > 0.01


def test_stable_grasp_cannot_be_claimed_without_physics_receipt():
    with pytest.raises(ValueError, match="physics-validated"):
        make_reset(build_task("peg_insert"), 0, seed=0, families=("stable_grasp",))


def test_engaged_cap_bank_does_not_jitter_into_bottle_shell():
    definition = build_task("bottle_cap", "open")
    for configuration_id in range(5):
        record = make_reset(definition, configuration_id, seed=0, families=("reaching",))
        cap = np.asarray(record.definition.initial["cap"].position)
        bottle = np.asarray(record.definition.initial["bottle"].position)
        assert np.linalg.norm(cap[:2] - bottle[:2]) < 0.0008


def test_reverse_latch_near_goal_keeps_a_nontrivial_remaining_operation():
    task = build_task("latch", task_parameters={"target_state": "released"})
    record = make_reset(task, 0, seed=0, families=("near_goal",))
    assert record.family == "near_goal"
    # From22mm engaged,65% toward the released0mm goal leaves7.7mm to slide.
    assert np.isclose(record.definition.parameters["initial_joints"]["bolt_travel"], 0.0077)
