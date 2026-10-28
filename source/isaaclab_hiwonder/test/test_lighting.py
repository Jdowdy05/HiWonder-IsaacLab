# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent color and optical-frame contracts used by the physical light adapter."""

import math

import pytest
import torch


@pytest.mark.parametrize(
    "controls",
    [
        {"saturation": -0.1},
        {"saturation": 1.1},
        {"power": 3.1},
        {"radius_m": 0.004},
        {"radius_m": 0.251},
        {"speed": -1},
        {"fill_intensity": 2001},
        {"power": True},
        {"speed": "1"},
        {"power": float("inf")},
        {"speed": float("nan")},
        {"typo": 0},
    ],
)
def test_preview_controls_reject_invalid_updates(controls):
    from isaaclab_hiwonder.lighting import validate_preview_controls

    with pytest.raises(ValueError):
        validate_preview_controls(controls)


def test_preview_controls_normalize_supported_partial_updates_without_mutating_input():
    from isaaclab_hiwonder.lighting import validate_preview_controls

    controls = {"saturation": 0, "power": 3, "radius_m": 0.25, "speed": 0, "fill_intensity": 2000}
    result = validate_preview_controls(controls)
    assert result == {"saturation": 0.0, "power": 3.0, "radius_m": 0.25, "speed": 0.0, "fill_intensity": 2000.0}
    assert all(type(value) is float for value in result.values())
    result["power"] = 0
    assert controls["power"] == 3


def test_rgb_wheel_primaries_white_and_half_saturation():
    from isaaclab_hiwonder.lighting import rgb_light_colors

    angle = torch.tensor([[0, 2 * math.pi / 3, 4 * math.pi / 3, math.pi / 3, 0, 0]])
    saturation = torch.tensor([[1, 1, 1, 1, 0, 0.5]])
    torch.testing.assert_close(
        rgb_light_colors(angle, saturation),
        torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0.5, 0.5]]]),
        atol=1e-6,
        rtol=0,
    )


def test_camera_link_pose_rotates_scaled_mount_translation_and_composes_xyzw():
    from isaaclab_hiwonder.lighting import camera_pose_from_link

    root_half = math.sqrt(0.5)
    link = torch.tensor([[2, 3, 4, 0, 0, root_half, root_half]])
    position, quaternion = camera_pose_from_link(
        link, torch.tensor([[0.2, 0, 0]]), torch.tensor([[root_half, 0, 0, root_half]])
    )
    torch.testing.assert_close(position, torch.tensor([[2, 3.2, 4]]))
    torch.testing.assert_close(quaternion, torch.tensor([[0.5, 0.5, 0.5, 0.5]]))


def test_wrist_targets_reproject_with_translation_rotation_and_unwrapped_motion():
    from isaaclab_hiwonder.lighting import wrist_view_targets

    # Two cameras look down, the second also rotates the image 90 degrees about world Z.
    position = torch.tensor([[0, 0, 1], [2, 3, 2]], dtype=torch.float64)
    quaternion = torch.tensor([[1, 0, 0, 0], [math.sqrt(0.5), math.sqrt(0.5), 0, 0]], dtype=torch.float64)
    angle = torch.tensor([[0, math.pi / 2, 2 * math.pi - 1e-6, 2 * math.pi + 1e-6]], dtype=torch.float64).repeat(2, 1)
    extent = torch.full_like(angle, 0.5)
    intrinsics = (100, 0, 100, 0, 100, 100, 0, 0, 1)
    targets, valid = wrist_view_targets(
        position, quaternion, angle, extent, torch.zeros(2), torch.zeros((2, 3)), intrinsics, 200, 200
    )
    assert valid.all()
    torch.testing.assert_close(
        targets[:, :2],
        torch.tensor([[[0.5, 0, 0], [0, -0.4557016383, 0]], [[2, 4, 0], [2.9114032766, 3, 0]]], dtype=torch.float64),
        atol=1e-8,
        rtol=0,
    )
    # The slower vertical oscillation must not jump when horizontal phase crosses 2pi.
    assert torch.all((targets[:, 2] - targets[:, 3]).norm(dim=-1) < 2e-6)


def test_wrist_horizon_and_behind_camera_use_bounded_support_fallback():
    from isaaclab_hiwonder.lighting import wrist_view_targets

    positions = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float64)
    orientations = torch.tensor(
        [[0, math.sqrt(0.5), 0, math.sqrt(0.5)], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=torch.float64
    )
    fallback = torch.tensor([[2, 3, 0.5]]).repeat(3, 1)
    targets, valid = wrist_view_targets(
        positions,
        orientations,
        torch.zeros((3, 1), dtype=torch.float64),
        torch.zeros((3, 1)),
        torch.tensor([0.5, 0.5, -10]),
        fallback,
        (100, 0, 100, 0, 100, 100, 0, 0, 1),
        200,
        200,
    )
    assert not valid.any()
    torch.testing.assert_close(targets[..., :2], fallback[:, None, :2].double())
    torch.testing.assert_close(targets[..., 2], torch.tensor([[0.5], [0.5], [-10]], dtype=torch.float64))
