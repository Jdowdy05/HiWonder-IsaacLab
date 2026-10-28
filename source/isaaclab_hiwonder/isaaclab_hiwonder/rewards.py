# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurable sparse objective with bounded potential-based manipulation shaping."""

import math
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    dense: bool = True
    shaping_discount: float = 0.99
    progress_weight: float = 1.0
    reach_weight: float = 0.4
    grasp_weight: float = 0.2
    lift_weight: float = 0.2
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    time_cost: float = 0.01
    action_cost: float = 0.01
    velocity_cost: float = 0.0001
    effort_cost: float = 0.0001

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != "dense" and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Reward {field.name} must be finite and nonnegative")
        if not 0 < self.shaping_discount <= 1:
            raise ValueError("shaping_discount must be in (0,1]")


class RewardShaper:
    """One previous potential per environment, checkpointed and cleared only on reset.

    Match shaping_discount to PPO gamma for the usual potential-shaping guarantee.
    Ground-truth reward features never enter the actor's observation groups.
    """

    def __init__(self, num_envs, config=None):
        self.config = config or RewardConfig()
        self.previous = np.full(num_envs, np.nan)

    def evaluate(
        self, env_id, task, result, state, tool_position, dt, action_cost=0.0, velocity_cost=0.0, effort_cost=0.0
    ):
        cfg = self.config
        dynamic = [body.name for body in task.bodies if not body.static and body.shapes]
        points = [np.asarray(state.actors[name].position) for name in dynamic]
        for nodes in state.deformable_positions.values():
            points.extend(nodes)
        distance = min((np.linalg.norm(point - tool_position) for point in points), default=1.0)
        reach = 1 - np.tanh(distance / 0.08)
        grasped = [
            name for name in dynamic if state.touching(name, "left_finger") and state.touching(name, "right_finger")
        ]
        grasp = float(bool(grasped)) if task.name != "roll_ball" else 0.0
        # Only free rigid objects earn lifting potential; articulated controls remain grounded.
        connected = {name for joint in task.joints for name in (joint.parent, joint.child)}
        lift = (
            max(
                (
                    np.clip((state.actors[name].position[2] - task.initial[name].position[2]) / 0.04, 0, 1)
                    for name in grasped
                    if name not in connected
                ),
                default=0.0,
            )
            if grasp
            else 0.0
        )
        potential = (
            (
                cfg.progress_weight * float(result.components.get("progress", 0))
                + cfg.reach_weight * reach
                + cfg.grasp_weight * grasp
                + cfg.lift_weight * lift
            )
            if cfg.dense
            else 0.0
        )
        previous = 0.0 if np.isnan(self.previous[env_id]) else float(self.previous[env_id])
        terminal_potential = 0.0 if result.terminated else potential
        shaping = cfg.shaping_discount * terminal_potential - previous
        self.previous[env_id] = potential
        components = {
            "potential_shaping": float(shaping),
            "success": cfg.success_bonus * float(result.success),
            "failure": -cfg.failure_penalty * float(result.failure),
            "time": -cfg.time_cost * dt,
            "action": -cfg.action_cost * action_cost * dt,
            "velocity": -cfg.velocity_cost * velocity_cost * dt,
            "effort": -cfg.effort_cost * effort_cost * dt,
        }
        return float(sum(components.values())), components

    def reset(self, env_ids):
        self.previous[np.asarray(env_ids, dtype=int)] = np.nan

    def state_dict(self):
        return {"previous": self.previous.copy()}

    def load_state_dict(self, state):
        values = np.asarray(state["previous"], dtype=float)
        if values.shape != self.previous.shape:
            raise ValueError("Reward checkpoint environment count differs")
        self.previous[:] = values
