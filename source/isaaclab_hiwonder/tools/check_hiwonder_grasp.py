# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Qualify a simulated stable peg grasp for an SGS reset seed, without hardware.

The fixture starts with the actual task peg suspended between registered CAD
finger pads. It must close before the peg falls, dwell with measured two-jaw
contacts, then lift while retaining the peg. No seed is written on failure.
"""

import argparse
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-directory", type=Path, required=True)
parser.add_argument("--close-target", type=float, default=0.12)
parser.add_argument("--robot-model", choices=("stock", "eka_v6"), default="stock")
add_launcher_args(parser)
args = parser.parse_args()
args.require_kit = True

import numpy as np
import torch

from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_hiwonder.assets.robot_models import get_robot_model
from isaaclab_hiwonder.config import GroupControl, HiWonderEnvConfig
from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.tasks.api import rotation_matrix
from isaaclab_hiwonder.tasks.catalog import build_task
from isaaclab_hiwonder.tasks.native_backend import physics_config

# Area-weighted distal contact planes, source CAD objects201 and218 at jawq0.
PAD_MIDPOINT_GRIPPER_M = np.array([0.00002459672, -0.07043074001, -0.07097931923])


def numpy(value):
    return value.torch.detach().cpu().numpy().copy() if hasattr(value, "torch") else value.detach().cpu().numpy().copy()


def quaternion_product(a, b):
    xyz = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    return np.r_[xyz, a[3] * b[3] - np.dot(a[:3], b[:3])]


def relative_pose(parent_pose, child_pose):
    inverse_q = np.r_[-parent_pose[3:6], parent_pose[6]]
    return np.r_[
        rotation_matrix(inverse_q) @ (child_pose[:3] - parent_pose[:3]), quaternion_product(inverse_q, child_pose[3:])
    ]


def main():
    model = get_robot_model(args.robot_model)
    profile = HiWonderEnvConfig(robot_model=model.name, num_envs=1, task="peg_insert", actor_mode="state_debug")
    controls = replace(
        profile.randomization.controls,
        **{field.name: GroupControl(enabled=False) for field in fields(profile.randomization.controls)},
    )
    profile = replace(profile, randomization=replace(profile.randomization, controls=controls))
    definition = build_task("peg_insert", seed=0)
    manager = RandomizationManager(1, profile.randomization, seed=0, robot_model=model)
    sim_cfg = SimulationCfg(dt=1 / 240, device=args.device, use_fabric=False, physics=physics_config(definition))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with launch_simulation(sim_cfg, args):
        from isaaclab.scene import InteractiveScene

        from isaaclab_hiwonder.sim.scene_config import build_scene

        scene_cfg, metadata = build_scene(profile, [definition], manager.snapshot(), args.output_directory / "scene")
        scene_cfg.wrist_camera = None
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(scene_cfg)
        scene.filter_collisions()
        sim.reset()
        robot = scene["robot"]
        peg = scene[metadata["task_body_assets"]["peg"]]
        order = [robot.joint_names.index(name) for name in model.joint_names]
        target = torch.zeros((1, model.num_joints), device=sim.device)
        # The source-assembly zero pose admits the task peg between these pads.
        # The wider training reset would let it fall before closure. This is an
        # explicit grasp fixture, not a change to the environment reset pose.
        root_pose = torch.tensor(metadata["robot_initial_pose"], device=sim.device, dtype=torch.float32)
        root_pose[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_joint_position_to_sim_index(position=target)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target))
        gripper_id = robot.body_names.index("gripper_body")
        base_id = robot.body_names.index("base")

        def step(command):
            robot.actuators.target_command.set_position_index(value=command)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_cfg.dt)

        def state():
            bodies = numpy(robot.data.body_link_pose_w)[0]
            object_pose = numpy(peg.data.root_link_pose_w)[0]
            return bodies, object_pose

        def contact_force(actor):
            strengths = []
            for physical_actor, entry in metadata["task_contacts"].items():
                if actor not in entry.get("actor_aliases", (physical_actor,)):
                    continue
                indices = [i for i, label in enumerate(entry["filter_labels"]) if label == "peg"]
                forces = numpy(scene[entry["sensor_key"]].data.normal_force_matrix_w)
                strengths.extend(float(np.linalg.norm(forces[0, :, index])) for index in indices)
            return max(strengths, default=0.0)

        for _ in range(160):
            step(target)
        bodies, _ = state()
        gripper_pose = bodies[gripper_id]
        center = gripper_pose[:3] + rotation_matrix(gripper_pose[3:]) @ PAD_MIDPOINT_GRIPPER_M
        if model.finger_pad_offsets_m:
            center = np.mean(
                [
                    bodies[robot.body_names.index(name), :3]
                    + rotation_matrix(bodies[robot.body_names.index(name), 3:]) @ np.asarray(offset)
                    for name, offset in model.finger_pad_offsets_m.items()
                ],
                axis=0,
            )
        object_pose = torch.tensor([[*center, 0, 0, 0, 1]], device=sim.device, dtype=torch.float32)
        peg.write_root_pose_to_sim_index(root_pose=object_pose)
        peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=sim.device))
        finger_ids = [order[i] for i in model.gripper_joint_indices]
        open_positions = target[0, finger_ids].clone()
        closed_positions = torch.tensor(
            (args.close_target,) if model.name == "stock" else model.gripper_closed_positions_rad,
            device=sim.device,
        )
        if model.name == "stock":
            target[0, finger_ids] = closed_positions
        records, consecutive, stable = [], 0, False
        required_steps = int(np.ceil(0.5 / sim_cfg.dt))
        close_steps = max(1, round(0.05 / sim_cfg.dt))
        for step_id in range(480):
            if model.name != "stock":
                target[0, finger_ids] = open_positions + min(1.0, (step_id + 1) / close_steps) * (
                    closed_positions - open_positions
                )
            step(target)
            bodies, object_state = state()
            left, right = contact_force("left_finger"), contact_force("right_finger")
            if not np.isfinite(object_state).all():
                raise RuntimeError("Peg became nonfinite")
            consecutive = consecutive + 1 if min(left, right) > 0.02 else 0
            records.append([step_id * sim_cfg.dt, left, right, *object_state[:3].tolist()])
            if consecutive >= required_steps:
                stable = True
                break
        qualification = dict(
            stable_two_jaw_dwell=stable,
            dwell_seconds=consecutive * sim_cfg.dt,
            minimum_contact_threshold_n=0.02,
            close_target_rad=args.close_target if model.name == "stock" else None,
            finger_close_targets_rad=closed_positions.detach().cpu().tolist(),
            closure_ramp_seconds=close_steps * sim_cfg.dt if model.name != "stock" else 0.0,
            fixture_initial_finger_targets_rad=open_positions.detach().cpu().tolist(),
            task="peg_insert",
            geometry_variant="nominal",
            robot_model=model.name,
            joint_names=model.joint_names,
            source_robot_sha256=hashlib.sha256(model.asset_path().read_bytes()).hexdigest(),
            source_task_sha256=hashlib.sha256(
                (args.output_directory / "scene" / "task_0.usda").read_bytes()
            ).hexdigest(),
            source_cad_sha256=(
                "f675f6049f94206c7b74ab73c40485913916babff1a6f4fc145dc9040672a2f8"
                if model.name == "stock"
                else hashlib.sha256((model.asset_directory / "source.step").read_bytes()).hexdigest()
            ),
            physics_dt=sim_cfg.dt,
            initial_spawn_world_m=center.tolist(),
        )
        if not stable:
            qualification.update(status="FAIL_CLOSE_OR_DWELL", contacts=records)
            (args.output_directory / "receipt.json").write_text(json.dumps(qualification, indent=2))
            raise RuntimeError("No continuous two-jaw contact for 0.5 seconds; stable-grasp seed not qualified")
        bodies, object_state = state()
        relative_before = relative_pose(bodies[gripper_id], object_state)
        seed = dict(
            robot_joint_positions=numpy(robot.data.joint_pos)[0, order],
            robot_joint_velocities=numpy(robot.data.joint_vel)[0, order],
            robot_joint_targets=numpy(target)[0, order],
            object_pose_robot_base=relative_pose(bodies[base_id], object_state),
            object_velocity_world=numpy(peg.data.root_link_vel_w)[0],
            robot_base_pose_world=bodies[base_id],
            gripper_relative_object_pose=relative_before,
        )
        initial_height = object_state[2]
        lift_contacts = []
        for step_id in range(300):
            target[0, order[model.joint_names.index("shoulder_lift")]] = -0.15 * min(1.0, (step_id + 1) / 150)
            step(target)
            lift_contacts.append([contact_force("left_finger"), contact_force("right_finger")])
        bodies, final_object = state()
        relative_after = relative_pose(bodies[gripper_id], final_object)
        height_gain = float(final_object[2] - initial_height)
        slip = float(np.linalg.norm(relative_after[:3] - relative_before[:3]))
        retained_fraction = float((np.min(np.asarray(lift_contacts)[-required_steps:], axis=1) > 0.02).mean())
        qualified = height_gain > 0.01 and slip < 0.008 and retained_fraction > 0.95
        qualification.update(
            status="PASS_PHYSICS_GRASP" if qualified else "FAIL_LIFT_RETENTION",
            fixture_status="PASS_STABLE_GRASP_AND_LIFT" if qualified else "FAIL_LIFT_RETENTION",
            hold_seconds=consecutive * sim_cfg.dt,
            jaw_contacts={"left_finger": bool(qualified), "right_finger": bool(qualified)},
            robot_asset_sha256=qualification["source_robot_sha256"],
            joint_positions=seed["robot_joint_positions"].tolist(),
            joint_targets=seed["robot_joint_targets"].tolist(),
            object_name="peg",
            object_pose_robot_base=seed["object_pose_robot_base"].tolist(),
            requested_actuator_parameters={
                key: value[0].tolist() for key, value in manager.snapshot()["actuator"].items()
            },
            actuator_parameters={
                name: numpy(getattr(robot.data, attribute))[0].tolist()
                for name, attribute in {
                    "stiffness": "joint_stiffness",
                    "damping": "joint_damping",
                    "armature": "joint_armature",
                    "friction": "joint_friction_coeff",
                    "effort_limit": "joint_effort_limits",
                    "velocity_limit": "joint_vel_limits",
                }.items()
            },
            command_channel="direct solver targets; no encoder noise, delay or backlash injection in this fixture",
            height_gain_m=height_gain,
            gripper_relative_slip_m=slip,
            lift_final_dwell_two_jaw_fraction=retained_fraction,
            measured_joint_positions=seed["robot_joint_positions"].tolist(),
            commanded_joint_targets=seed["robot_joint_targets"].tolist(),
            contacts=records,
            lift_contacts=lift_contacts,
            limitations=(
                "Simulation-only seed for this exact nominal asset/task/actuator profile; "
                "physical grasp and randomized-profile robustness not established"
            ),
        )
        if qualified:
            seed_path = args.output_directory / "stable_peg_grasp.npz"
            np.savez(seed_path, **seed)
            qualification["seed_file"] = str(seed_path.resolve())
            qualification["seed_sha256"] = hashlib.sha256(seed_path.read_bytes()).hexdigest()
        (args.output_directory / "receipt.json").write_text(json.dumps(qualification, indent=2))
        print(json.dumps({k: v for k, v in qualification.items() if k not in ("contacts", "lift_contacts")}, indent=2))
        if not qualified:
            raise RuntimeError("Grasp lost contact, slipped or failed to lift; stable-grasp seed not qualified")


if __name__ == "__main__":
    main()
