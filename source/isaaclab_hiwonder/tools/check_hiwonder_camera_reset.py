# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Qualify first PhysX wrist-camera capture after full and indexed resets without stepping."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-directory", type=Path, required=True)
parser.add_argument("--actor-mode", choices=("state_debug", "rgb"), default="state_debug")
args = parser.parse_args()
args.output_directory.mkdir(parents=True, exist_ok=True)

from isaaclab_hiwonder.tasks import build_task
from isaaclab_hiwonder.tasks.native_backend import physics_config

from isaaclab.app import launch_simulation
from isaaclab.sim import SimulationCfg

bootstrap = SimulationCfg(
    dt=1 / 240, render_interval=8, device="cuda:0", physics=physics_config(build_task("peg_insert"))
)
with launch_simulation(bootstrap, {"device": "cuda:0", "enable_cameras": True, "require_kit": True}):
    import cv2
    import numpy as np
    import torch
    from isaaclab_hiwonder.assets.kinematics import forward_kinematics
    from isaaclab_hiwonder.sim.env import make_cfg, make_env
    from isaaclab_hiwonder.tasks.api import rotation_matrix
    from PIL import Image

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    cfg = make_cfg(
        "peg_insert", 2, seed=31, actor_mode=args.actor_mode, output_directory=args.output_directory / "scene"
    )
    cfg.sim.device = "cuda:0"
    depth = max(0.025, cfg.hiwonder_profile.camera.clipping_range_m[0] * 2)
    # Preload marker materials before sensor initialization, then relocate only
    # visual fixtures from independent FK after each reset, before its first capture.
    for file in cfg.hiwonder_metadata["room_files"]:
        stage = Usd.Stage.Open(file)
        root_path = "/Room/ResetProjectionMarkers"
        UsdGeom.Xform.Define(stage, root_path)
        for channel in range(3):
            marker = UsdGeom.Sphere.Define(stage, f"{root_path}/Marker{channel}")
            marker.CreateRadiusAttr(0.016 * depth)
            marker.AddTranslateOp().Set(Gf.Vec3d(0, 0, -1))
            material = UsdShade.Material.Define(stage, f"{root_path}/Material{channel}")
            shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*np.eye(3)[channel]) * 10)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()
    env = make_env(cfg=cfg)
    camera = env.scene["wrist_camera"]
    generations = np.zeros(2, dtype=int)
    marker_world = {}
    original_reset = env._reset_idx

    def expected_camera(root, joints, scale):
        base = np.eye(4)
        base[:3, :3] = rotation_matrix(root[3:])
        base[:3, 3] = root[:3]
        return base @ forward_kinematics(joints, float(scale))["camera_opencv"]

    def reset_with_markers(ids):
        original_reset(ids)
        roots = env.robot.data.root_pose_w.torch.detach().cpu().numpy()
        joints = env.robot.data.joint_pos.torch[:, env.joint_order].detach().cpu().numpy()
        scales = env.randomization.snapshot()["robot"]["geometry_scale"].numpy()
        for i in ids.detach().cpu().tolist():
            generations[i] += 1
            offsets = (
                [(-0.24, -0.16), (0.24, -0.16), (-0.24, 0.16)]
                if generations[i] % 2
                else [(-0.10, -0.25), (0.28, 0.12), (0.10, 0.05)]
            )
            coordinates = np.array([[x * depth, y * depth, depth] for x, y in offsets])
            transform = expected_camera(roots[i], joints[i], scales[i])
            marker_world[i] = coordinates @ transform[:3, :3].T + transform[:3, 3]
            for j, world in enumerate(marker_world[i]):
                path = f"/World/envs/env_{i}/Room/ResetProjectionMarkers/Marker{j}"
                local = world - env.scene.env_origins[i].detach().cpu().numpy()
                env.sim.stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Set(Gf.Vec3d(*local))

    env._reset_idx = reset_with_markers

    def snapshot(label):
        # First read must genuinely capture, without a checker-side refresh/repair.
        data = camera.data
        observed = data.pos_w.torch.detach().cpu().numpy().copy()
        rgb = data.output["rgb"].torch[..., :3].detach().cpu().numpy().copy()
        root = env.robot.data.root_pose_w.torch.detach().cpu().numpy().copy()
        q = env.robot.data.joint_pos.torch[:, env.joint_order].detach().cpu().numpy().copy()
        bodies = env.robot.data.body_link_pose_w.torch.detach().cpu().numpy().copy()
        scale = env.randomization.snapshot()["robot"]["geometry_scale"].numpy()
        expected = np.array(
            [
                r[:3] + rotation_matrix(r[3:]) @ forward_kinematics(p, float(s))["camera_opencv"][:3, 3]
                for r, p, s in zip(root, q, scale)
            ]
        )
        frame = camera.frame.torch.detach().cpu().numpy().copy()
        body_errors = [
            float(
                np.linalg.norm(
                    pose[:3]
                    - (
                        root[i, :3]
                        + rotation_matrix(root[i, 3:]) @ forward_kinematics(q[i], float(scale[i]))[name][:3, 3]
                    )
                )
            )
            for i, poses in enumerate(bodies)
            for name, pose in zip(env.robot.body_names, poses)
        ]
        np.savez_compressed(
            args.output_directory / (label + ".npz"),
            expected=expected,
            observed=observed,
            q=q,
            root=root,
            frame=frame,
            body_poses=bodies,
        )
        for i, pixels in enumerate(rgb):
            Image.fromarray(pixels).save(args.output_directory / f"{label}_env{i}.png")
        projection_errors = []
        detected_pixels = []
        for i, image in enumerate(rgb.astype(float)):
            transform = expected_camera(root[i], q[i], scale[i])
            coordinates = (marker_world[i] - transform[:3, 3]) @ transform[:3, :3]
            expected_pixels = cv2.projectPoints(
                coordinates,
                np.zeros(3),
                np.zeros(3),
                np.asarray(env.profile.camera.intrinsics).reshape(3, 3),
                np.asarray(env.profile.camera.distortion_coefficients),
            )[0].reshape(-1, 2)
            errors, detected = [], []
            yy, xx = np.indices(image.shape[:2])
            for channel, expected_pixel in enumerate(expected_pixels):
                mask = (
                    (image[..., channel] > 120)
                    & (image[..., channel] > image[..., (channel + 1) % 3] * 1.8)
                    & (image[..., channel] > image[..., (channel + 2) % 3] * 1.8)
                    & ((xx - expected_pixel[0]) ** 2 + (yy - expected_pixel[1]) ** 2 < 18**2)
                )
                if mask.sum() < 10:
                    raise RuntimeError(f"First reset frame lacks independently projected marker {i}/{channel}")
                centroid = np.array([xx[mask].mean(), yy[mask].mean()])
                errors.append(float(np.linalg.norm(centroid - expected_pixel)))
                detected.append(centroid.tolist())
            projection_errors.append(errors)
            detected_pixels.append(detected)
        return {
            "errors_m": np.linalg.norm(observed - expected, axis=1).tolist(),
            "frame": frame.tolist(),
            "maximum_body_fk_error_m": max(body_errors),
            "first_frame_marker_errors_px": projection_errors,
            "detected_marker_pixels": detected_pixels,
        }, (
            q,
            root,
            observed,
            rgb,
        )

    try:
        initial_count = env.sim.get_physics_step_count()
        env.reset()
        first, before = snapshot("full_reset")
        after_full_count = env.sim.get_physics_step_count()
        env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
        env._compute_observations()
        second, after = snapshot("indexed_reset")
        final_count = env.sim.get_physics_step_count()
        report = {
            "full_reset": first,
            "indexed_reset": second,
            "physics_step_counts": [initial_count, after_full_count, final_count],
            "untouched_joint_state_exact": bool(np.array_equal(before[0][1], after[0][1])),
            "untouched_root_exact": bool(np.array_equal(before[1][1], after[1][1])),
            "untouched_camera_pose_exact": bool(np.array_equal(before[2][1], after[2][1])),
            "untouched_rgb_exact": bool(np.array_equal(before[3][1], after[3][1])),
        }
        passed = (
            max(first["errors_m"] + second["errors_m"]) < 1e-5
            and np.max(first["first_frame_marker_errors_px"] + second["first_frame_marker_errors_px"]) < 3
            and initial_count == after_full_count == final_count
            and all(
                report[k]
                for k in ("untouched_joint_state_exact", "untouched_root_exact", "untouched_camera_pose_exact")
            )
        )
        report["status"] = "PASS_RESET_FIRST_CAPTURE" if passed else "FAIL_RESET_FIRST_CAPTURE"
        (args.output_directory / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise RuntimeError("Camera first capture does not reflect reset state without advancing physics")
    finally:
        env.close()
