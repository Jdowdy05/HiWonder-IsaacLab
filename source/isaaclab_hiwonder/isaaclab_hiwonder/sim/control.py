# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sampled command latency and rate-independent play approximation of gear backlash.

This is a command-side dead band, not a measured compliant gear/contact model.
"""

import math

import torch


class MotorChannel:
    def __init__(self, num_envs, num_joints, dt, max_delay, device):
        self.dt = dt
        self.length = math.ceil(max_delay / dt) + 2
        self.buffer = torch.zeros((self.length, num_envs, num_joints), device=device)
        self.output = torch.zeros((num_envs, num_joints), device=device)
        self.cursor = 0

    def reset(self, env_ids, positions):
        self.buffer[:, env_ids] = positions
        self.output[env_ids] = positions

    def step(self, target, delay_seconds, backlash_radians):
        self.buffer[self.cursor] = target
        lag = torch.round(delay_seconds / self.dt).long().clamp(0, self.length - 1)
        rows = (self.cursor - lag) % self.length
        envs = torch.arange(target.shape[0], device=target.device)[:, None]
        joints = torch.arange(target.shape[1], device=target.device)[None, :]
        delayed = self.buffer[rows, envs, joints]
        half_slack = backlash_radians / 2
        self.output.copy_(torch.maximum(torch.minimum(self.output, delayed + half_slack), delayed - half_slack))
        self.cursor = (self.cursor + 1) % self.length
        return self.output

    def state_dict(self):
        return {"buffer": self.buffer.clone(), "output": self.output.clone(), "cursor": self.cursor}

    def load_state_dict(self, state):
        self.buffer.copy_(state["buffer"])
        self.output.copy_(state["output"])
        self.cursor = int(state["cursor"])
