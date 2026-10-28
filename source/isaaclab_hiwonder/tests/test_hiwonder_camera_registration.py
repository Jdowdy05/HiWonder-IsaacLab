# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Protect OpenCV camera-to-link inversion and renderer axis conventions.

The analytical cuboid fixture would expose returning solvePnP's inverse frame
directly, a common mount-registration error. This does not qualify real optics.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "camera_calibration", Path(__file__).parents[1] / "tools" / "calibrate_hiwonder_camera.py"
)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def test_mount_fit_returns_link_from_camera_and_rejects_planar_ambiguity():
    points = [
        [0, 0, 0],
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0.1, 0.1, 0],
        [0, 0, 0.1],
        [0.1, 0, 0.1],
        [0, 0.1, 0.1],
        [0.1, 0.1, 0.1],
    ]
    # Identity camera at link (0, 0, -.4), with fx=fy=400 and center=(320,240).
    pixels = [[320, 240], [420, 240], [320, 340], [420, 340], [320, 240], [400, 240], [320, 320], [400, 320]]
    profile = {"intrinsics": [400, 0, 320, 0, 400, 240, 0, 0, 1], "distortion_coefficients": [0] * 5}
    result = calibration.register_mount(points, pixels, profile, 0.01)
    np.testing.assert_allclose(result["optical_translation_m"], [0, 0, -0.4], atol=1e-7)
    np.testing.assert_allclose(result["optical_quaternion_xyzw"], [0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(np.abs(result["renderer_quaternion_xyzw"]), [1, 0, 0, 0], atol=1e-7)
    assert result["mount_reprojection_error_px"] < 1e-6
    planar = np.array(points)
    planar[:, 2] = 0
    with pytest.raises(ValueError, match="non-coplanar"):
        calibration.register_mount(planar, pixels, profile, 1)


def test_fitted_profile_is_loadable_without_claiming_physical_validation(tmp_path):
    from isaaclab_hiwonder.config import CameraProfile

    source = tmp_path / "fit.json"
    source.write_text(
        json.dumps(
            {
                "width": 640,
                "height": 480,
                "intrinsics": [640, 0, 320, 0, 640, 240, 0, 0, 1],
                "distortion_coefficients": [0.1, 0, 0, 0, 0],
                "calibration_status": "image_fit_pending_independent_validation",
                "calibration_source": [{"file": "capture.png", "sha256": "example"}],
                "per_image_rms_px": [0.15, 0.2],
                "measured_square_m": 0.015,
            }
        )
    )
    output = calibration.environment_profile(source)
    profile = CameraProfile(**output["camera"])
    assert profile.calibration_status == "estimated"
    assert profile.calibration_source == str(source.resolve())
    assert profile.horizontal_fov_degrees == pytest.approx(53.1301023542)
    assert profile.vertical_fov_degrees == pytest.approx(41.1120904392)
    assert profile.focal_length_mm == pytest.approx(4.8)
    assert profile.f_stop is None
    assert "pending independent" in profile.provenance
