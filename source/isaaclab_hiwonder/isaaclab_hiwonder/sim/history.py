# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Owned, checkpointable per-environment visual frame history."""

import torch


class FrameHistory:
    """Return BCHW stacks ordered oldest-to-newest, priming new episodes with their first frame."""

    def __init__(
        self, num_envs: int, num_frames: int, channels: int, height: int, width: int, device: str | torch.device = "cpu"
    ):
        if min(num_envs, num_frames, channels, height, width) < 1:
            raise ValueError("Frame history dimensions must be positive")
        self.frames = torch.zeros((num_envs, num_frames, channels, height, width), device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def push(self, image: torch.Tensor) -> torch.Tensor:
        """Append actual consumed BCHW pixels and return an owned flattened-channel stack."""
        if image.shape != (self.frames.shape[0], *self.frames.shape[2:]) or image.device != self.frames.device:
            raise ValueError("Frame shape/device differs from declared history")
        if image.dtype != self.frames.dtype:
            raise ValueError("Frame history expects float32 policy images")
        fresh = ~self.initialized
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = image
        self.frames[fresh] = image[fresh, None].expand(-1, self.frames.shape[1], -1, -1, -1)
        self.initialized[:] = True
        return self.frames.flatten(1, 2).clone()

    def reset(self, env_ids=None) -> None:
        """Prime only selected environments on their next captured policy frame."""
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.frames.device, dtype=torch.long)
        self.initialized[ids] = False
        self.frames[ids] = 0

    def state_dict(self) -> dict:
        return dict(version=1, frames=self.frames.clone(), initialized=self.initialized.clone())

    def load_state_dict(self, state: dict) -> None:
        if (
            state["version"] != 1
            or state["frames"].shape != self.frames.shape
            or state["initialized"].shape != self.initialized.shape
        ):
            raise ValueError("Frame history checkpoint shape/schema mismatch")
        self.frames.copy_(state["frames"])
        self.initialized.copy_(state["initialized"])
