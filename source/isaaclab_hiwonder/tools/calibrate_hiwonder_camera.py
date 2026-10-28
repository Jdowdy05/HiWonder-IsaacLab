# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Offline wrist-camera intrinsics and mount registration, without hardware access.

Intrinsics: supply a directory of already captured checkerboard images, inner
corner dimensions, and measured square size. Mount: supply an intrinsic profile
and JSON with ``points_link_m`` and corresponding ``pixels`` from a target whose
points are measured in the gripper_body frame. A single planar target can be
ambiguous; use well-spread non-coplanar points and independent validation views.
This tool records measured fit residuals; physical acceptance remains separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np


def file_evidence(path: Path) -> dict:
    return {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def matrix_to_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a proper rotation to xyzw using Rodrigues, including half turns."""
    vector, _ = cv2.Rodrigues(rotation)
    vector = vector.reshape(3)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [*(vector / angle * np.sin(angle / 2)).tolist(), float(np.cos(angle / 2))]


def calibrate_intrinsics(paths: list[Path], inner_corners: tuple[int, int], square_m: float, max_error: float) -> dict:
    """Fit Brown-Conrady intrinsics and report individual capture residuals."""
    if min(inner_corners) < 3 or square_m <= 0:
        raise ValueError("Use at least 3x3 inner corners and a positive measured square size")
    target = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)
    target[:, :2] = np.mgrid[0 : inner_corners[0], 0 : inner_corners[1]].T.reshape(-1, 2) * square_m
    objects, points, accepted = [], [], []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode calibration image {path}")
        size = (image.shape[1], image.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError("Calibration captures must all have the same resolution")
        image_size = size
        found, corners = cv2.findChessboardCornersSB(image, inner_corners)
        if found:
            objects.append(target)
            points.append(corners)
            accepted.append(path)
    if len(accepted) < 10:
        raise ValueError(f"Need >=10 detected views with varied orientations; detected {len(accepted)}")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(objects, points, image_size, None, None)
    errors = []
    for observed, rotation, translation in zip(points, rotations, translations):
        projected, _ = cv2.projectPoints(target, rotation, translation, matrix, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=-1)))))
    if not np.isfinite(rms) or rms > max_error:
        raise ValueError(f"Calibration RMS {rms:.3f}px exceeds acceptance threshold {max_error:.3f}px")
    return dict(
        width=image_size[0],
        height=image_size[1],
        intrinsics=matrix.reshape(-1).tolist(),
        distortion_coefficients=distortion.reshape(-1).tolist(),
        distortion_model="brown_conrady",
        reprojection_error_px=float(rms),
        per_image_rms_px=errors,
        calibration_status="image_fit_pending_independent_validation",
        calibration_source=[file_evidence(p) for p in accepted],
        board_inner_corners=list(inner_corners),
        measured_square_m=square_m,
        f_stop=None,
        optical_transform_status="not_registered",
        qualification="Fit residual alone does not verify optics, target scale, image roll, or physical mount",
    )


def register_mount(points_link_m, pixels, profile: dict, max_error: float) -> dict:
    """Estimate link-from-OpenCV transform from measured 3D/2D correspondences."""
    points = np.asarray(points_link_m, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or pixels.shape != (len(points), 2) or len(points) < 6:
        raise ValueError("Need >=6 corresponding link-frame 3D points and 2D pixels")
    if np.linalg.matrix_rank(points - points.mean(0), tol=1e-6) < 3:
        raise ValueError("Mount registration requires non-coplanar points to reject planar pose ambiguity")
    matrix = np.asarray(profile["intrinsics"]).reshape(3, 3)
    distortion = np.asarray(profile["distortion_coefficients"])
    ok, rvec, tvec = cv2.solvePnP(points, pixels, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Pose fit failed")
    rotation, _ = cv2.Rodrigues(rvec)
    if np.any((points @ rotation.T + tvec.reshape(3))[:, 2] <= 0):
        raise ValueError("Pose places target points behind the camera")
    projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
    if rms > max_error or not np.isfinite(rms):
        raise ValueError(f"Mount reprojection RMS {rms:.3f}px exceeds {max_error:.3f}px")
    link_from_camera = rotation.T
    return dict(
        optical_parent_link="gripper_body",
        optical_translation_m=(-link_from_camera @ tvec).reshape(3).tolist(),
        optical_quaternion_xyzw=matrix_to_xyzw(link_from_camera),
        renderer_quaternion_xyzw=matrix_to_xyzw(link_from_camera @ np.diag([1, -1, -1])),
        optical_frame_convention="opencv",
        renderer_frame_convention="opengl",
        optical_transform_status="correspondence_fit_pending_independent_validation",
        mount_reprojection_error_px=rms,
    )


def environment_profile(fit_path: Path, base_profile_path: Path | None = None) -> dict:
    """Create a loadable environment override without promoting a fit to validation.

    The fit remains the provenance authority, including all image hashes and
    residuals. Only CameraProfile fields enter the runtime JSON. Intrinsics fix
    projection; the virtual filmback scale remains an unmeasured modeling choice.
    """
    from isaaclab_hiwonder.config import CameraProfile

    fit = json.loads(fit_path.read_text())
    base = json.loads(base_profile_path.read_text()) if base_profile_path else {}
    camera = asdict(CameraProfile())
    camera.update(base.get("camera", {}))
    known = {field.name for field in fields(CameraProfile)}
    camera.update({key: value for key, value in fit.items() if key in known})
    camera["calibration_source"] = str(fit_path.resolve())
    camera["calibration_status"] = "estimated"
    matrix = np.asarray(camera["intrinsics"], dtype=float).reshape(3, 3)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    width, height = camera["width"], camera["height"]
    if not np.isfinite(matrix).all() or fx <= 0 or fy <= 0 or not 0 < cx < width or not 0 < cy < height:
        raise ValueError("Fit requires finite positive intrinsics and an in-frame principal point")
    camera["horizontal_fov_degrees"] = math.degrees(math.atan(cx / fx) + math.atan((width - cx) / fx))
    camera["vertical_fov_degrees"] = math.degrees(math.atan(cy / fy) + math.atan((height - cy) / fy))
    filmback_width = float(camera["sensor_aperture_mm"][0])
    focal = fx * filmback_width / width
    camera["focal_length_mm"] = focal
    camera["sensor_aperture_mm"] = [filmback_width, height * focal / fy]
    camera["provenance"] = (
        "Image/correspondence fit pending independent held-out projection and mount validation; "
        f"source SHA256={file_evidence(fit_path)['sha256']}; "
        "FOV derived from intrinsic matrix; virtual filmback scale and physical optics remain unmeasured"
    )
    CameraProfile(**camera)
    base["camera"] = camera
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    intrinsic = sub.add_parser("intrinsics")
    intrinsic.add_argument("images", type=Path)
    intrinsic.add_argument("--inner-corners", nargs=2, type=int, required=True)
    intrinsic.add_argument("--square-m", type=float, required=True)
    mount = sub.add_parser("mount")
    mount.add_argument("correspondences", type=Path)
    mount.add_argument("--profile", type=Path, required=True)
    profile = sub.add_parser("profile", help="Convert fitted calibration to a loadable environment camera override")
    profile.add_argument("fit", type=Path)
    profile.add_argument("--base-profile", type=Path)
    profile.add_argument("--output", type=Path, required=True)
    for command in (intrinsic, mount):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-rms-px", type=float, default=1.0)
    args = parser.parse_args()
    if args.mode == "intrinsics":
        paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        result = calibrate_intrinsics(paths, tuple(args.inner_corners), args.square_m, args.max_rms_px)
    elif args.mode == "mount":
        data = json.loads(args.correspondences.read_text())
        result = json.loads(args.profile.read_text())
        result.update(register_mount(data["points_link_m"], data["pixels"], result, args.max_rms_px))
        result["mount_source"] = file_evidence(args.correspondences)
    else:
        result = environment_profile(args.fit, args.base_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
