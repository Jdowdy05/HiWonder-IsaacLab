# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_hiwonder.config import CameraProfile, HiWonderEnvConfig


def test_camera_requires_calibration_provenance_and_qualified_optical_transform():
    with pytest.raises(ValueError, match="provenance"):
        CameraProfile(calibration_status="calibrated")
    with pytest.raises(ValueError, match="quaternion"):
        CameraProfile(optical_quaternion_xyzw=(0, 0, 0, 0))
    camera = CameraProfile()
    assert camera.fx == camera.intrinsics[0]
    assert camera.f_stop is None
    with pytest.raises(ValueError, match="control"):
        HiWonderEnvConfig(control_hz=0)


def test_backend_selection_rejects_incompatible_task_physics():
    from isaaclab_hiwonder.config import resolve_backend

    assert resolve_backend("auto", "rigid") == "physx"
    assert resolve_backend("auto", "cable") == "newton_vbd"
    assert resolve_backend("newton_vbd", "cloth") == "newton_vbd"
    with pytest.raises(ValueError, match="requires"):
        resolve_backend("physx", "cloth")


@pytest.mark.parametrize(
    "field,value",
    [
        ("optical_parent_link", "wrist"),
        ("optical_frame_convention", "opengl"),
        ("renderer_frame_convention", "opencv"),
    ],
)
def test_camera_rejects_frames_the_scene_cannot_apply(field, value):
    with pytest.raises(ValueError, match=field):
        CameraProfile(**{field: value})


def test_robot_rejects_scale_disagreeing_with_named_prebuilt_asset():
    from isaaclab_hiwonder.config import RobotRandomizationConfig

    with pytest.raises(ValueError, match="scale.*nominal|nominal.*scale"):
        RobotRandomizationConfig(geometry_variants=("nominal",), variant_scales=(1.01,))
    with pytest.raises(ValueError, match="prebuilt|Unknown"):
        RobotRandomizationConfig(geometry_variants=("missing_variant",), variant_scales=(1.0,))
    assert RobotRandomizationConfig(geometry_variants=("scale099",), variant_scales=(0.99,))


@pytest.mark.parametrize("unsupported", ["actor_mode", "joint_count"])
def test_environment_rejects_actor_or_motor_models_it_cannot_execute(unsupported):
    from dataclasses import replace

    from isaaclab_hiwonder.config import ActuatorRandomizationConfig, RandomizationConfig

    kwargs = (
        {"actor_mode": "unsupported"}
        if unsupported == "actor_mode"
        else {"randomization": replace(RandomizationConfig(), actuator=ActuatorRandomizationConfig(num_joints=7))}
    )
    with pytest.raises(ValueError, match="actor_mode|six"):
        HiWonderEnvConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color_saturation": (-0.1, 0.5)},
        {"color_saturation": (0, 1.1)},
        {"saturation_period_s": -1},
        {"saturation_period_s": float("nan")},
        {"spot_radius_m": (0, 0.1)},
        {"edge_softness": (0, 1.1)},
        {"view_extent": (0, 0.91)},
        {"white_fill_intensity": (-1, 0)},
        {"intensity": (2, 1)},
        {"angular_speed": (0, float("inf"))},
        {"count": (0, 5)},
        {"count": (0.0, 1)},
        {"max_lights": True},
        {"types": ("laser",)},
        {"types": ()},
        {"profile": "invalid"},
        {"motion_mode": "invalid"},
        {"temperature_kelvin": (0, 100)},
        {"beam_angle_degrees": (0, 180)},
        {"shadows_probability": float("nan")},
        {"emitter_radius_m": 0},
        {"emitter_radius_m": float("inf")},
    ],
)
def test_lighting_rejects_nonphysical_domains(kwargs):
    from isaaclab_hiwonder.config import LightingConfig

    with pytest.raises(ValueError):
        LightingConfig(**kwargs)
