# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Auditable registration of the supplied HiWonder follower CAD, in millimeters.

The CAD assembly is the zero joint pose. Link axes remain parallel to CAD axes at
zero; joint frames carry the revolute-axis rotations. Object indices refer to the
SHA-256-pinned source in the generated manifest, never to the generic SO101 URDF.
Joint travel and inertial quantities are conservative simulation estimates pending
physical calibration. No motor communication is performed by this package.
"""

from pathlib import Path

ASSET_DIR = Path(__file__).parent
USD_PATH = ASSET_DIR / "generated" / "hiwonder_soarm101.usdc"
MANIFEST_PATH = ASSET_DIR / "generated" / "manifest.json"
CAMERA_PATH = "gripper_body/camera_mount/Camera"
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NOMINAL_JOINT_POSITIONS_RAD = (0.0, 0.0, 0.0, 0.0, 0.0, -0.3490658503988659)
GRIPPER_OPEN_RAD = -0.3490658503988659
GRIPPER_CLOSED_RAD = 0.20943951023931956


def _indices(*groups: int | range) -> list[int]:
    return sorted(i for group in groups for i in (group if isinstance(group, range) else [group]))


LINKS = {
    "base": {
        "origin_mm": [0.0, 0.0, 0.39],
        "mass_kg": 0.35,
        "objects": _indices(range(14), 31, 42, 45, range(48, 123), 260),
    },
    "shoulder": {
        "origin_mm": [0.0, -7.23, 62.55],
        "mass_kg": 0.18,
        "objects": _indices(14, 15, range(16, 31), range(32, 42), 43, 44, 46, 47, 264),
    },
    "upper_arm": {
        "origin_mm": [0.0, -37.82, 118.89],
        "mass_kg": 0.20,
        "objects": _indices(123, 124, range(125, 154)),
    },
    "forearm": {
        "origin_mm": [0.23, 22.03, 218.26],
        "mass_kg": 0.19,
        "objects": _indices(154, 155, range(156, 184), 262, 263),
    },
    "wrist": {
        "origin_mm": [0.48, -69.64, 317.29],
        "mass_kg": 0.13,
        "objects": _indices(184, 185, range(186, 200), range(219, 231), 259),
    },
    "gripper_body": {
        "origin_mm": [0.5, -111.81, 275.47],
        "mass_kg": 0.19,
        "objects": _indices(200, 201, range(202, 216), range(241, 259), 261),
    },
    "moving_jaw": {
        "origin_mm": [-19.65, -141.51, 269.95],
        "mass_kg": 0.025,
        "objects": _indices(216, 217, 218, range(231, 241)),
    },
}

# Pivots are registered from opposed servo output horns/shaft geometry.
JOINTS = [
    dict(name="shoulder_pan", parent="base", child="shoulder", axis=[0, 0, 1], limits_deg=[-100, 100]),
    dict(name="shoulder_lift", parent="shoulder", child="upper_arm", axis=[1, 0, 0], limits_deg=[-85, 85]),
    dict(name="elbow_flex", parent="upper_arm", child="forearm", axis=[1, 0, 0], limits_deg=[-85, 85]),
    dict(name="wrist_flex", parent="forearm", child="wrist", axis=[1, 0, 0], limits_deg=[-85, 85]),
    dict(
        name="wrist_roll",
        parent="wrist",
        child="gripper_body",
        axis=[0, -0.7071067812, -0.7071067812],
        limits_deg=[-100, 100],
    ),
    dict(
        name="gripper",
        parent="gripper_body",
        child="moving_jaw",
        axis=[0, -0.7071067812, 0.7071067812],
        limits_deg=[-35, 12],
    ),
]

CAMERA = {
    "width": 640,
    "height": 480,
    "fps": 30.0,
    "intrinsics": [554.2666666667, 0, 320, 0, 554.2666666667, 240, 0, 0, 1],
    "distortion_coefficients": [0, 0, 0, 0, 0],
    "distortion_model": "brown_conrady",
    "sensor_aperture_mm": [4.8, 3.6],
    "focal_length_mm": 4.157,
    "f_stop": None,
    "focus_distance_m": 0.3,
    "clipping_range_m": [0.01, 10.0],
    "calibration_status": "estimated",
    "optical_transform_status": "CAD_mechanical_estimate",
    "optical_parent_link": "gripper_body",
    "optical_translation_m": [-0.00250012, -0.05209936, 0.04465591],
    "optical_quaternion_xyzw": [0.984807753012208, 0, 0, 0.17364817766693041],
    "renderer_quaternion_xyzw": [-0.17364817766693041, 0, 0, 0.984807753012208],
    "optical_frame_convention": "opencv",
    "renderer_frame_convention": "opengl",
    "estimated_optical_center_offset_m": [0, 0, 0],
    "lens_source_object_index": 253,
    "provenance": (
        "640x480 YUYV 30fps USB descriptor; optics illustrative 60deg HFOV; "
        "CAD lens plane normal from mesh PCA, outward sign from housing-to-lens vector; "
        "optical-center offset and roll not optically calibrated"
    ),
    "calibration_source": None,
    "reprojection_error_px": None,
}

# Structural shapes use convex decomposition so the jaw gap is not filled by one hull.
STRUCTURAL_COLLIDERS = {30, 31, 139, 201, 218, 259, 260, 261, 262, 263, 264}
SERVO_COLLIDERS = {0, 1, 11, 16, 17, 27, 125, 126, 136, 158, 168, 169, 186, 187, 197, 202, 203, 213, 252, 254}
