# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure RGB light color and calibrated optical-ray geometry, without simulator imports."""

import math

import torch

from isaaclab.utils.math import quat_apply, quat_mul


def validate_preview_controls(controls: dict) -> dict[str, float]:
    """Validate a partial physical-light preview update atomically and return owned floats."""
    bounds = {
        "saturation": (0, 1),
        "power": (0, 3),
        "radius_m": (0.005, 0.25),
        "speed": (0, 3),
        "fill_intensity": (0, 2000),
    }
    if not isinstance(controls, dict) or any(key not in bounds for key in controls):
        raise ValueError("Preview controls must be an object with supported lighting keys")
    for name, value in controls.items():
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not bounds[name][0] <= value <= bounds[name][1]
        ):
            raise ValueError(f"Preview control {name} must be finite within {bounds[name]}")
    return {name: float(value) for name, value in controls.items()}


def rgb_light_colors(phase_radians: torch.Tensor, saturation: torch.Tensor) -> torch.Tensor:
    """Convert hue angle [rad] and saturation [0,1], shape [N,L], to value-one HSV RGB.

    Saturation zero is white; saturation one is the pure hue. Intensity is independent;
    equal HSV value does not imply equal perceived luminance across hues.
    """
    hue = torch.remainder(phase_radians / (2 * math.pi), 1)
    offsets = hue.new_tensor([0, 4, 2])
    channels = ((hue[..., None] * 6 + offsets) % 6 - 3).abs().sub(1).clamp(0, 1)
    return 1 - saturation[..., None] * (1 - channels)


def camera_pose_from_link(
    link_pose: torch.Tensor, translation: torch.Tensor, rotation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose world link pose [N,7] and mount translation [N,3]/rotation [N,4], xyzw.

    Translation [m] must already include the robot's launch-cohort geometry scale.
    The camera optical frame and calibration are preserved.
    """
    return link_pose[:, :3] + quat_apply(link_pose[:, 3:], translation), quat_mul(link_pose[:, 3:], rotation)


def wrist_view_targets(
    camera_position: torch.Tensor,
    camera_quaternion_xyzw: torch.Tensor,
    angle: torch.Tensor,
    view_extent: torch.Tensor,
    support_z: torch.Tensor,
    fallback_centers: torch.Tensor,
    intrinsics: tuple[float, ...],
    width: int,
    height: int,
    max_distance_m: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect moving OpenCV optical rays with a horizontal workspace support plane.

    Returns world targets [N,L,3] [m] and a valid-ray mask [N,L]. Image paths occupy
    ``view_extent`` of each half-frame dimension. The unwrapped angle [rad] keeps the
    slower vertical oscillation continuous. This plane approximation ignores object
    occlusion and raised geometry. Horizon, backward and distant intersections use
    a bounded smooth orbit around fallback centers, and report false validity.
    """
    fx, fy = intrinsics[0], intrinsics[4]
    # Reduce each oscillator independently: wrapping angle before 0.73 would jump.
    horizontal = torch.remainder(angle, 2 * math.pi)
    vertical = torch.remainder(0.73 * angle, 2 * math.pi)
    rays = torch.stack(
        (
            view_extent * (width / (2 * fx)) * horizontal.cos(),
            view_extent * (height / (2 * fy)) * vertical.sin(),
            torch.ones_like(angle),
        ),
        -1,
    )
    quaternion = camera_quaternion_xyzw.to(rays.dtype)[:, None]
    direction = quat_apply(quaternion, rays)
    position = camera_position.to(rays.dtype)[:, None]
    plane = support_z.to(rays.dtype)[:, None]
    denominator = direction[..., 2]
    safe_denominator = torch.where(denominator.abs() > 1e-6, denominator, torch.ones_like(denominator))
    distance = (plane - position[..., 2]) / safe_denominator
    targets = position + distance[..., None] * direction
    valid = (
        (denominator.abs() > 1e-6)
        & (distance > 0)
        & (distance * direction.norm(dim=-1) <= max_distance_m)
        & torch.isfinite(targets).all(-1)
    )
    centers = fallback_centers.to(rays.dtype)[:, None]
    fallback = torch.stack(
        (
            centers[..., 0] + 0.1 * view_extent * horizontal.cos(),
            centers[..., 1] + 0.1 * view_extent * vertical.sin(),
            plane.expand_as(angle),
        ),
        -1,
    )
    return torch.where(valid[..., None], targets, fallback), valid
