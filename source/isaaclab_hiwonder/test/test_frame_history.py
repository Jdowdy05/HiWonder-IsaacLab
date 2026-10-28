# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_hiwonder.sim.history import FrameHistory


def test_frame_history_subset_reset_fills_current_and_checkpoint_replays_owned_observations():
    history = FrameHistory(2, 2, 3, 4, 6, device="cpu")
    first = torch.stack((torch.full((3, 4, 6), 0.1), torch.full((3, 4, 6), 0.2)))
    second = first + 0.3
    with torch.inference_mode():
        history.push(first)
        prior = history.push(second)
    assert torch.equal(prior[:, :3], first)
    assert torch.equal(prior[:, 3:], second)
    history.reset([0])
    third = second + 0.2
    with torch.inference_mode():
        current = history.push(third)
    assert torch.equal(current[0, :3], third[0])
    assert torch.equal(current[0, 3:], third[0])
    assert torch.equal(current[1, :3], second[1])
    assert torch.equal(current[1, 3:], third[1])
    assert torch.equal(prior[:, :3], first)
    restored = FrameHistory(2, 2, 3, 4, 6, device="cpu")
    saved = history.state_dict()
    history.load_state_dict(saved)
    restored.load_state_dict(saved)
    fourth = torch.full_like(first, 0.9)
    assert torch.equal(history.push(fourth), restored.push(fourth))
