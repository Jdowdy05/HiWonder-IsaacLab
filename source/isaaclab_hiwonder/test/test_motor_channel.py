# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab_hiwonder.sim.control import MotorChannel


def test_play_operator_reversal_and_independent_delay():
    channel = MotorChannel(2, 1, dt=0.01, max_delay=0.04, device="cpu")
    channel.reset(torch.tensor([0, 1]), torch.zeros(2, 1))
    delay = torch.tensor([[0.0], [0.02]])
    slack = torch.tensor([[0.2], [0.0]])
    first = channel.step(torch.ones(2, 1), delay, slack)
    assert torch.allclose(first, torch.tensor([[0.9], [0.0]]))
    second = channel.step(torch.full((2, 1), 0.95), delay, slack)
    assert torch.allclose(second, torch.tensor([[0.9], [0.0]]))
    third = channel.step(torch.full((2, 1), 0.7), delay, slack)
    assert torch.allclose(third, torch.tensor([[0.8], [1.0]]))


def test_subset_reset_does_not_erase_other_joint_history():
    channel = MotorChannel(2, 1, 0.01, 0.04, "cpu")
    channel.reset(torch.tensor([0, 1]), torch.zeros(2, 1))
    delay, slack = torch.full((2, 1), 0.01), torch.zeros(2, 1)
    channel.step(torch.tensor([[1.0], [2.0]]), delay, slack)
    channel.reset(torch.tensor([0]), torch.tensor([[3.0]]))
    result = channel.step(torch.tensor([[4.0], [5.0]]), delay, slack)
    assert torch.equal(result, torch.tensor([[3.0], [2.0]]))


def test_inference_steps_allow_subset_reset_and_checkpoint_restore_outside_inference():
    channel = MotorChannel(2, 1, 0.01, 0.04, "cpu")
    delay, slack = torch.zeros((2, 1)), torch.full((2, 1), 0.2)
    with torch.inference_mode():
        channel.step(torch.ones((2, 1)), delay, slack)
    saved = channel.state_dict()
    with torch.inference_mode():
        channel.step(-torch.ones((2, 1)), delay, slack)
    channel.reset(torch.tensor([0]), torch.tensor([[0.4]]))
    torch.testing.assert_close(channel.output, torch.tensor([[0.4], [-0.9]]))
    channel.load_state_dict(saved)
    torch.testing.assert_close(channel.output, torch.tensor([[0.9], [0.9]]))
    # Replay the delayed next command after restoring both output and ring history.
    result = channel.step(torch.zeros((2, 1)), torch.full((2, 1), 0.01), slack)
    torch.testing.assert_close(result, torch.tensor([[0.9], [0.9]]))
