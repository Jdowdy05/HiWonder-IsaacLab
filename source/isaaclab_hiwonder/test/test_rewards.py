# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from isaaclab_hiwonder.rewards import RewardShaper
from isaaclab_hiwonder.tasks import SceneState, TaskResult, build_task


def test_stationary_near_goal_cannot_accumulate_positive_progress_reward():
    task = build_task("peg_insert")
    state = SceneState(actors=task.initial)
    result = TaskResult(0, False, False, False, False, {"progress": 0.95}, {})
    shaper = RewardShaper(1)
    shaper.evaluate(0, task, result, state, np.array([0.18, -0.07, 0.04]), 1 / 30)
    rewards = [shaper.evaluate(0, task, result, state, np.array([0.18, -0.07, 0.04]), 1 / 30)[0] for _ in range(100)]
    assert max(rewards) <= 0
    done = TaskResult(1, True, False, True, False, {"progress": 1.0}, {})
    assert shaper.evaluate(0, task, done, state, np.array([0.27, 0.04, 0.02]), 1 / 30)[0] > 5


def test_reset_and_checkpoint_preserve_other_environments_potential():
    task = build_task("peg_insert")
    state = SceneState(actors=task.initial)
    result = TaskResult(0, False, False, False, False, {"progress": 0.5}, {})
    shaper = RewardShaper(2)
    shaper.evaluate(1, task, result, state, np.array([0.18, -0.07, 0.04]), 1 / 30)
    shaper.reset([0])
    restored = RewardShaper(2)
    restored.load_state_dict(shaper.state_dict())
    assert shaper.evaluate(1, task, result, state, np.zeros(3), 1 / 30) == restored.evaluate(
        1, task, result, state, np.zeros(3), 1 / 30
    )
