# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorded image-space ISP approximation and frame latency/drop channel.

Exposure scales rendered radiance relative to an 8ms reference. This does not
simulate a physical rolling shutter or establish the real camera's response.
Capture IDs gate ISP/drop decisions; timestamps gate delivery independently of
the control rate. Reset warm-up primes history with the first available capture.
"""

import math

import torch


class CameraChannel:
    def __init__(self, profile, num_envs, device, seed=0, control_hz=None):
        self.profile = profile
        if profile.latency_seconds < 0 or not 0 <= profile.frame_drop_probability <= 1:
            raise ValueError("Invalid camera latency/drop probability")
        if profile.response_gamma <= 0 or profile.exposure_seconds <= 0 or profile.gain <= 0:
            raise ValueError("Camera ISP gamma, exposure and gain must be positive")
        self.dt = 1 / (control_hz or profile.fps)
        self.delay = math.ceil(profile.latency_seconds / self.dt - 1e-9)
        self.frames = torch.zeros(
            (self.delay + 2, num_envs, profile.height, profile.width, 3), device=device, dtype=torch.uint8
        )
        self.times = torch.full((len(self.frames), num_envs), -float("inf"), device=device, dtype=torch.float64)
        self.last_ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.age_seconds = torch.zeros(num_envs, device=device)
        self.time = 0.0
        self.generators = [
            torch.Generator(device=device).manual_seed((seed + i * 1000003) % (2**63 - 1)) for i in range(num_envs)
        ]
        self.cursor = 0
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def apply(self, image, frame_ids=None):
        p = self.profile
        value = (image.float() / 255).pow(2.2)
        balance = value.new_tensor(p.white_balance_rgb)
        value = (value * (p.exposure_seconds / 0.008 * p.gain) * balance).clamp(0, 1)
        capture = (value.pow(1 / p.response_gamma) * 255).round().to(torch.uint8)
        fresh = ~self.initialized
        self.frames[:, fresh] = capture[fresh]
        self.times[:, fresh] = self.time - p.latency_seconds
        previous = self.frames[(self.cursor - 1) % len(self.frames)]
        previous_times = self.times[(self.cursor - 1) % len(self.frames)].clone()
        ids = self.last_ids + 1 if frame_ids is None else torch.as_tensor(frame_ids, device=image.device).long()
        updated = (ids != self.last_ids) | fresh
        dropped = torch.zeros(len(image), dtype=torch.bool, device=image.device)
        # Duplicate control observations of one capture do not consume camera RNG.
        for i in updated.nonzero(as_tuple=False).flatten().tolist():
            dropped[i] = torch.rand((), device=image.device, generator=self.generators[i]) < p.frame_drop_probability
        accepted = updated & (~dropped | fresh)
        self.frames[self.cursor] = torch.where(accepted[:, None, None, None], capture, previous)
        self.times[self.cursor] = torch.where(accepted, self.time, previous_times)
        eligible = self.times <= self.time - p.latency_seconds + 1e-8
        scores = torch.where(eligible, self.times, -float("inf"))
        indices = scores.argmax(0)
        env_ids = torch.arange(len(image), device=image.device)
        result = self.frames[indices, env_ids].clone()
        self.age_seconds.copy_((self.time - self.times[indices, env_ids]).float())
        self.cursor = (self.cursor + 1) % len(self.frames)
        self.last_ids.copy_(ids)
        self.time += self.dt
        self.initialized[:] = True
        return result

    def reset(self, env_ids):
        # Avoid previous-episode visual leakage during a declared environment reset.
        self.frames[:, env_ids] = 0
        self.initialized[env_ids] = False
        self.last_ids[env_ids] = -1

    def state_dict(self):
        return {
            "frames": self.frames.clone(),
            "cursor": self.cursor,
            "initialized": self.initialized.clone(),
            "times": self.times.clone(),
            "time": self.time,
            "last_ids": self.last_ids.clone(),
            "age_seconds": self.age_seconds.clone(),
            "rng_states": [generator.get_state() for generator in self.generators],
        }

    def load_state_dict(self, state):
        if "rng_states" not in state or len(state["rng_states"]) != len(self.generators):
            raise ValueError(
                "Camera checkpoint requires one drop RNG stream per environment; legacy global RNG unsupported"
            )
        self.frames.copy_(state["frames"])
        self.cursor = state["cursor"]
        self.initialized.copy_(state["initialized"])
        self.times.copy_(state["times"])
        self.time = state["time"]
        self.last_ids.copy_(state["last_ids"])
        self.age_seconds.copy_(state["age_seconds"])
        for generator, rng_state in zip(self.generators, state["rng_states"]):
            generator.set_state(rng_state.cpu())
