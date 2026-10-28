# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import replace

import torch
from isaaclab_hiwonder.config import CameraProfile
from isaaclab_hiwonder.sim.camera_channel import CameraChannel


def test_nominal_isp_preserves_bytes_and_delay_uses_old_frame():
    profile = replace(CameraProfile(), width=2, height=2, latency_seconds=1 / 30)
    channel = CameraChannel(profile, 1, "cpu", seed=1)
    first = torch.arange(12, dtype=torch.uint8).reshape(1, 2, 2, 3) * 20
    assert torch.equal(channel.apply(first), first)
    assert torch.equal(channel.apply(torch.full_like(first, 100)), first)
    assert torch.equal(channel.apply(torch.full_like(first, 200)), torch.full_like(first, 100))


def test_frame_drop_holds_capture_across_inference_steps_reset_and_restore():
    profile = replace(CameraProfile(), width=2, height=2, frame_drop_probability=1.0)
    channel = CameraChannel(profile, 1, "cpu", seed=7)
    first = torch.full((1, 2, 2, 3), 10, dtype=torch.uint8)
    with torch.inference_mode():
        channel.apply(first)
        assert torch.equal(channel.apply(first * 2), first)
    state = channel.state_dict()
    channel.reset([0])
    channel.load_state_dict(state)
    restored = CameraChannel(profile, 1, "cpu", seed=99)
    restored.load_state_dict(state)
    for capture in range(3, 6):
        assert torch.equal(channel.apply(first * capture), first)
        assert torch.equal(restored.apply(first * capture), first)
        torch.testing.assert_close(channel.age_seconds, restored.age_seconds)


def test_latency_uses_elapsed_time_and_drop_only_new_captures():
    profile = replace(CameraProfile(), width=2, height=2, latency_seconds=1 / 30)
    channel = CameraChannel(profile, 1, "cpu", seed=7, control_hz=60)
    first = torch.full((1, 2, 2, 3), 10, dtype=torch.uint8)
    channel.apply(first, frame_ids=torch.tensor([1]))
    rng = channel.state_dict()["rng_states"][0]
    assert torch.equal(channel.apply(first * 2, frame_ids=torch.tensor([1])), first)
    assert torch.equal(channel.state_dict()["rng_states"][0], rng)
    assert torch.equal(channel.apply(first * 2, frame_ids=torch.tensor([2])), first)
    assert torch.equal(channel.apply(first * 2, frame_ids=torch.tensor([2])), first)
    assert torch.equal(channel.apply(first * 3, frame_ids=torch.tensor([3])), first * 2)


def test_other_camera_reset_cannot_change_retained_capture_drop_history():
    profile = replace(CameraProfile(), width=1, height=1, frame_drop_probability=0.5, latency_seconds=1 / 30)
    reference = CameraChannel(profile, 2, "cpu", seed=1)
    reset_other = CameraChannel(profile, 2, "cpu", seed=1)
    first = torch.full((2, 1, 1, 3), 10, dtype=torch.uint8)
    for channel in (reference, reset_other):
        channel.apply(first, torch.tensor([1, 1]))
    reset_other.reset([0])
    for channel in (reference, reset_other):
        channel.apply(first, torch.tensor([1, 1]))
    # Fresh captures after the unrelated reset must have identical delivery and age.
    for capture in range(2, 12):
        pixels = torch.full_like(first, capture * 20)
        actual = reset_other.apply(pixels, torch.tensor([capture, capture]))
        expected = reference.apply(pixels, torch.tensor([capture, capture]))
        assert torch.equal(actual[1], expected[1])
        assert torch.equal(reset_other.age_seconds[1], reference.age_seconds[1])
    restored = CameraChannel(profile, 2, "cpu", seed=99)
    restored.load_state_dict(reset_other.state_dict())
    for capture in range(12, 16):
        pixels = torch.full_like(first, capture)
        assert torch.equal(
            restored.apply(pixels, torch.tensor([capture, capture])),
            reset_other.apply(pixels, torch.tensor([capture, capture])),
        )
