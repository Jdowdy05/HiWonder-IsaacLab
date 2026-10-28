# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Render real wrist-camera frames and exercise per-room material isolation."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-directory", type=Path, required=True)
parser.add_argument("--task", default="peg_insert")
parser.add_argument("--robot-model", choices=("stock", "eka_v6"), default="stock")
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--camera-profile", type=Path, help="Environment JSON containing a camera profile override")
parser.add_argument(
    "--physical-clutter", type=int, default=0, help="Exercise this many physical confetti and distractor bodies"
)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import numpy as np
import torch
from PIL import Image

from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_hiwonder.assets.robot_models import get_robot_model
from isaaclab_hiwonder.config import HiWonderEnvConfig
from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.tasks.catalog import build_task
from isaaclab_hiwonder.tasks.native_backend import physics_config


def as_tensor(value):
    return value.torch if hasattr(value, "torch") else value


def main():
    profile = HiWonderEnvConfig(robot_model=args.robot_model, num_envs=args.num_envs, task=args.task)
    if args.physical_clutter:
        from dataclasses import replace

        profile = replace(
            profile,
            randomization=replace(
                profile.randomization,
                confetti=replace(profile.randomization.confetti, physical_budget=args.physical_clutter),
                distractor=replace(
                    profile.randomization.distractor,
                    count=(args.physical_clutter, args.physical_clutter),
                    physical_probability=1.0,
                ),
            ),
        )
    if args.camera_profile:
        from dataclasses import replace

        from isaaclab_hiwonder.config import CameraProfile

        camera_fields = json.loads(args.camera_profile.read_text())["camera"]
        profile = replace(profile, camera=CameraProfile(**camera_fields))
    randomization = RandomizationManager(
        profile.num_envs,
        profile.randomization,
        seed=profile.seed,
        bootstrap_attempts=32,
        robot_model=profile.robot_model,
    )
    definitions = [build_task(args.task, seed=i) for i in range(profile.num_envs)]
    from isaaclab_hiwonder.sim.reachability import reject_unreachable_bases
    from isaaclab_hiwonder.sim.support import bootstrap_task_cohorts

    support = bootstrap_task_cohorts(randomization, definitions)
    witnesses = reject_unreachable_bases(randomization, definitions, robot_model=profile.robot_model)
    sim_cfg = SimulationCfg(dt=1 / 120, render_interval=4, device=args.device, physics=physics_config(definitions[0]))
    with launch_simulation(sim_cfg, args):
        # On Windows USD must first load from Kit; pre-launch USD authoring loads
        # the standalone USD wheel's incompatible DLLs into this process.
        from isaaclab.scene import InteractiveScene

        from isaaclab_hiwonder.sim.rooms import MaterialSample, update_material
        from isaaclab_hiwonder.sim.scene_config import build_scene

        scene_cfg, metadata = build_scene(profile, definitions, randomization.snapshot(), args.output_directory)
        metadata["task_support"] = support
        metadata["position_ik_witnesses"] = {i: value.tolist() for i, value in witnesses.items()}
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(scene_cfg)
        scene.filter_collisions()
        sim.reset()
        robot = scene["robot"]
        pose = torch.tensor(metadata["robot_initial_pose"], device=sim.device, dtype=torch.float32)
        pose[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim_index(root_pose=pose)
        q = torch.tensor(get_robot_model(profile.robot_model).nominal_joint_positions_rad, device=sim.device).repeat(
            profile.num_envs, 1
        )
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        robot.actuators.target_command.set_position_index(value=q)
        for _ in range(64):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
        camera = scene["wrist_camera"]
        clutter_state = {}
        for category in ("distractor", "confetti"):
            for index, key in metadata[f"{category}_assets"].items():
                poses = as_tensor(scene[key].data.root_link_pose_w).detach().cpu().numpy()
                local = poses[:, :3] - scene.env_origins.cpu().numpy()
                if not np.isfinite(poses).all() or np.any(
                    local[:, 2] < np.asarray(metadata["support_height_m"]) - 0.005
                ):
                    raise RuntimeError(f"Clutter fell through support or became nonfinite: {key}")
                clutter_state[key] = local.tolist()

        def capture():
            for _ in range(12):
                sim.render()
                camera.update(1 / 30, force_recompute=True)
            return as_tensor(camera.data.output["rgb"]).detach().cpu().numpy()[..., :3].copy()

        a, b = capture(), capture()
        if a.shape != (profile.num_envs, 480, 640, 3) or not np.isfinite(a).all() or a.std() < 2:
            raise RuntimeError(f"Camera output is invalid: {a.shape}, standard deviation {a.std()}")
        baseline = np.abs(a.astype(float) - b.astype(float)).mean(axis=(1, 2, 3))
        stage = sim.stage
        update_material(
            stage,
            metadata["room_paths"][0]["materials"]["table"],
            MaterialSample(color=(1, 0.005, 0.005), roughness=0.8, metallic=0),
        )
        changed = capture()
        difference = np.abs(b.astype(float) - changed.astype(float)).mean(axis=(1, 2, 3))
        # Light and shadow linking are authored under each owning room. Change
        # every light in room 0 while preserving the second room's settings.
        from pxr import Gf, UsdGeom, UsdLux, UsdShade

        before_light = capture()
        for path in metadata["room_paths"][0]["lights"]:
            light = UsdLux.LightAPI(stage.GetPrimAtPath(path))
            light.CreateIntensityAttr(0)
        after_light = capture()
        light_difference = np.abs(before_light.astype(float) - after_light.astype(float)).mean(axis=(1, 2, 3))
        for i in range(profile.num_envs):
            Image.fromarray(before_light[i].astype(np.uint8)).save(
                args.output_directory / f"wrist_env{i}_lights_on.png"
            )
            Image.fromarray(after_light[i].astype(np.uint8)).save(
                args.output_directory / f"wrist_env{i}_lights_off.png"
            )
        print("LIGHT_DIFFERENCE", light_difference.tolist(), "BASELINE", baseline.tolist(), flush=True)
        light_isolation = profile.num_envs < 2 or light_difference[1] <= max(0.25, float(baseline[1]) * 3)
        if light_difference[0] < 0.25 or not light_isolation:
            raise RuntimeError(f"Light isolation/effect failed: baseline={baseline}, changes={light_difference}")

        # A deterministic spot fixture exercises the same rotation/color fields
        # used by white/RGB disco, with enough flux to produce a measurable beam.
        spot_prim = stage.GetPrimAtPath(metadata["room_paths"][0]["lights"][0])
        spot = UsdLux.LightAPI(spot_prim)
        spot.CreateNormalizeAttr(True)
        spot.CreateEnableColorTemperatureAttr(False)
        spot.CreateIntensityAttr(1200)
        spot.CreateColorAttr(Gf.Vec3f(1))
        UsdLux.ShadowAPI.Apply(spot_prim).CreateShadowEnableAttr(True)
        shaping = UsdLux.ShapingAPI.Apply(spot_prim)
        shaping.CreateShapingConeAngleAttr(30)
        shaping.CreateShapingConeSoftnessAttr(0.1)
        operations = UsdGeom.Xformable(spot_prim).GetOrderedXformOps()
        translate = next(op for op in operations if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        orient = next(op for op in operations if op.GetOpType() == UsdGeom.XformOp.TypeOrient)
        translate.Set(Gf.Vec3d(0.18, 0, metadata["support_height_m"][0] + 0.5))
        orient.Set(Gf.Quatf(1, Gf.Vec3f(0)))
        white_down = capture()
        orient.Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 25).GetQuat()))
        white_swept = capture()
        orient.Set(Gf.Quatf(1, Gf.Vec3f(0)))
        spot.CreateColorAttr(Gf.Vec3f(0.05, 0.5, 1))
        rgb_swept = capture()
        spot_direction_difference = np.abs(white_down.astype(float) - white_swept.astype(float)).mean(axis=(1, 2, 3))
        spot_color_difference = np.abs(white_down.astype(float) - rgb_swept.astype(float)).mean(axis=(1, 2, 3))
        for label, delta in (
            ("white_spot_direction", spot_direction_difference),
            ("rgb_spot_color", spot_color_difference),
        ):
            if delta[0] < 0.25 or (profile.num_envs > 1 and delta[1] > max(0.25, float(baseline[1]) * 3)):
                raise RuntimeError(f"{label} effect/isolation failed: {delta}")
        Image.fromarray(white_down[0].astype(np.uint8)).save(args.output_directory / "white_spot_down.png")
        Image.fromarray(white_swept[0].astype(np.uint8)).save(args.output_directory / "white_spot_swept.png")
        Image.fromarray(rgb_swept[0].astype(np.uint8)).save(args.output_directory / "rgb_spot_swept.png")
        spot.CreateIntensityAttr(0)

        # Projection proof uses visible emissive spheres at known camera-frame
        # coordinates. Their image centroids test rendering, not attribute reads.
        from pxr import Sdf

        marker_depth = max(0.025, profile.camera.clipping_range_m[0] * 2)
        marker_positions = [
            (-0.24 * marker_depth, 0.16 * marker_depth, -marker_depth),
            (0.24 * marker_depth, 0.16 * marker_depth, -marker_depth),
            (-0.24 * marker_depth, -0.16 * marker_depth, -marker_depth),
        ]
        marker_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        # Keep calibration markers outside the robot's stronger-than-descendant
        # material binding, which intentionally overrides its own mesh materials.
        camera_path = "/World/envs/env_0/Room/ProjectionTest"
        root = UsdGeom.Xform.Define(stage, camera_path)
        position = (as_tensor(camera.data.pos_w)[0] - scene.env_origins[0]).detach().cpu().tolist()
        quaternion = as_tensor(camera.data.quat_w_opengl)[0].detach().cpu().tolist()
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        root.AddOrientOp().Set(Gf.Quatf(quaternion[3], Gf.Vec3f(*quaternion[:3])))
        for i, (position, color) in enumerate(zip(marker_positions, marker_colors)):
            marker = UsdGeom.Sphere.Define(stage, camera_path + f"/ProjectionMarker{i}")
            marker.CreateRadiusAttr(0.016 * marker_depth)
            marker.AddTranslateOp().Set(Gf.Vec3d(*position))
            material = UsdShade.Material.Define(stage, camera_path + f"/ProjectionMaterial{i}")
            shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color) * 10)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)
        projected = capture()
        Image.fromarray(projected[0].astype(np.uint8)).save(args.output_directory / "camera_projection_markers.png")
        projection_errors = []
        for i, position in enumerate(marker_positions):
            x, y, z = position
            expected = np.array(
                [profile.camera.cx + profile.camera.fx * x / -z, profile.camera.cy - profile.camera.fy * y / -z]
            )
            # This nominal projection check requires zero distortion. Nonzero
            # profiles are projected with the same documented OpenCV convention.
            import cv2

            point = np.array([[[x, -y, -z]]], dtype=np.float64)
            expected = cv2.projectPoints(
                point,
                np.zeros(3),
                np.zeros(3),
                np.asarray(profile.camera.intrinsics).reshape(3, 3),
                np.asarray(profile.camera.distortion_coefficients),
            )[0].reshape(2)
            image = projected[0].astype(float)
            mask = (
                (image[..., i] > 120)
                & (image[..., i] > image[..., (i + 1) % 3] * 1.8)
                & (image[..., i] > image[..., (i + 2) % 3] * 1.8)
            )
            yy, xx = np.indices(mask.shape)
            mask &= (xx - expected[0]) ** 2 + (yy - expected[1]) ** 2 < 18**2
            if mask.sum() < 10:
                raise RuntimeError(f"Rendered calibration marker {i} is not visible at its predicted projection")
            measured = np.array([xx[mask].mean(), yy[mask].mean()])
            projection_errors.append(float(np.linalg.norm(measured - expected)))
        if max(projection_errors) > 3:
            raise RuntimeError(f"Rendered camera projection discrepancy: {projection_errors}")
        contacts = {}
        for actor, entry in metadata["task_contacts"].items():
            matrix = as_tensor(scene[entry["sensor_key"]].data.normal_force_matrix_w).detach().cpu()
            expected = (profile.num_envs, 1, len(entry["filter_labels"]), 3)
            if tuple(matrix.shape) != expected or not torch.isfinite(matrix).all():
                raise RuntimeError(f"Filtered contact matrix mismatch for {actor}: {matrix.shape} != {expected}")
            contacts[actor] = dict(shape=list(matrix.shape), maximum_force_n=float(matrix.norm(dim=-1).max()))
        if difference[0] < 0.25:
            raise RuntimeError(f"Changed room's rendered pixels did not respond: {difference}")
        isolation = profile.num_envs < 2 or difference[1] <= max(0.25, float(baseline[1]) * 3)
        if not isolation:
            raise RuntimeError(f"Room material leaked into another camera: baseline={baseline}, changes={difference}")
        for i in range(profile.num_envs):
            Image.fromarray(b[i].astype(np.uint8)).save(args.output_directory / f"wrist_env{i}_baseline.png")
            Image.fromarray(changed[i].astype(np.uint8)).save(args.output_directory / f"wrist_env{i}_changed.png")
            Image.fromarray(after_light[i].astype(np.uint8)).save(
                args.output_directory / f"wrist_env{i}_lights_off.png"
            )
        Image.fromarray(projected[0].astype(np.uint8)).save(args.output_directory / "camera_projection_markers.png")
        result = dict(
            status="PASS_RENDER_AND_MATERIAL_ISOLATION",
            task=args.task,
            camera_shape=list(a.shape),
            camera_dtype=str(a.dtype),
            configured_capture_hz=profile.camera.fps,
            configured_intrinsics=list(profile.camera.intrinsics),
            baseline_noise_mean_abs_px=baseline.tolist(),
            changed_mean_abs_px=difference.tolist(),
            independent_room_material_render_pass=bool(isolation),
            light_changed_mean_abs_px=light_difference.tolist(),
            independent_room_light_render_pass=bool(light_isolation),
            white_spot_direction_change_mean_abs_px=spot_direction_difference.tolist(),
            rgb_spot_color_change_mean_abs_px=spot_color_difference.tolist(),
            projection_marker_centroid_errors_px=projection_errors,
            contact_sensors=contacts,
            physical_clutter_budget=args.physical_clutter,
            clutter_body_positions_env_local_m=clutter_state,
            asset_hashes=metadata["asset_hashes"],
            support_height_m=metadata["support_height_m"],
            limitations=(
                "Material and direct-light isolation tested with scene shadows enabled; "
                "independent shadow-only perturbation, task success and physical optical calibration remain unqualified"
            ),
        )
        (args.output_directory / "scene-check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
