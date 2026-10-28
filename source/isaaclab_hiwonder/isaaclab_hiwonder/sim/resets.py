# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable reaching/near-goal resets and physics-receipted stable grasp seeds."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab_hiwonder.assets.kinematics import forward_kinematics
from isaaclab_hiwonder.assets.robot_models import get_robot_model
from isaaclab_hiwonder.tasks.api import ActorState

from .reachability import solve_position, task_landmarks


class UnreachableReset(ValueError):
    """A sampled base nuisance cannot reach this immutable reset entry."""


@dataclass
class ResetRecord:
    definition: object
    robot_q: np.ndarray
    robot_target_q: np.ndarray
    family: str
    ik_witnesses: np.ndarray


def load_grasp_library(path, robot_model="stock"):
    """Require measured retention and both jaw contacts, plus the exact robot asset hash."""
    data = copy.deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text(encoding="utf-8"))
    model = get_robot_model(robot_model)
    if data.get("status") != "PASS_PHYSICS_GRASP" or data.get("hold_seconds", 0) < 0.5:
        raise ValueError("Stable grasp library lacks a passing >=0.5s physical retention receipt")
    if not all(data.get("jaw_contacts", {}).get(key, False) for key in ("left_finger", "right_finger")):
        raise ValueError("Stable grasp must have measured contact with both jaws")
    if data.get("robot_asset_sha256") != hashlib.sha256(model.asset_path().read_bytes()).hexdigest():
        raise ValueError("Stable grasp receipt is for a different robot asset")
    for field in ("joint_positions", "joint_targets"):
        values = np.asarray(data.get(field, []), dtype=float)
        if values.shape != (model.num_joints,):
            raise ValueError(f"Stable grasp requires measured and commanded {model.num_joints}-joint states")
        if not np.isfinite(values).all():
            raise ValueError("Stable grasp joint states must be finite")
    return data


def make_reset(
    definition,
    configuration_id,
    seed,
    families=("reaching", "near_goal"),
    grasp_library=None,
    cohort_count=1,
    base_position=(0, 0, 0),
    base_yaw=np.pi / 2,
    geometry_scale=1.0,
    robot_model="stock",
):
    """Generate a deterministic bank entry; stable seeds never fabricate held objects.

    Near-goal resets are currently defined for rigid insertion and single-joint
    mechanisms; other task families retain their validated native initial state.
    """
    if not families or any(name not in ("reaching", "near_goal", "stable_grasp") for name in families):
        raise ValueError("Reset families must select reaching, near_goal and/or stable_grasp")
    task = copy.deepcopy(definition)
    rng = np.random.default_rng(seed + int(configuration_id) * 1009 + 7919)
    model = get_robot_model(robot_model)
    q = np.asarray(model.nominal_joint_positions_rad, dtype=float).copy()
    q[list(model.arm_joint_indices)] += rng.uniform(-0.035, 0.035, len(model.arm_joint_indices))
    family = families[(int(configuration_id) // cohort_count) % len(families)]
    connected = {name for joint in task.joints for name in (joint.parent, joint.child)}
    if task.name == "bottle_cap" and task.parameters["operation"] == "open":
        connected.add("cap")  # Custom screw coupling is physical despite having no SDK joint.
    for body in task.bodies:
        if not body.static and body.name not in connected and not task.deformables:
            position = np.asarray(task.initial[body.name].position).copy()
            position[:2] += rng.uniform(-0.008, 0.008, 2)
            task.initial[body.name].position = position
    if family == "near_goal":
        if task.name in ("peg_insert", "connector_insert"):
            moving, seat = task.parameters["moving"], np.asarray(task.parameters["seat"])
            task.initial[moving] = ActorState(seat + (0.022, -0.012, 0.035))
        elif task.name in ("knob", "latch"):
            key = task.parameters["joint"]
            target = task.parameters["target"]
            initial = task.parameters.setdefault("initial_joints", {})
            start = initial.get(key, 0.0)
            initial[key] = start + 0.65 * (target - start)
        else:
            family = "reaching"
    target_q = q.copy()
    if family == "stable_grasp":
        if grasp_library is None:
            raise ValueError("stable_grasp requires a physics-validated grasp_library")
        receipt = load_grasp_library(grasp_library, robot_model=model)
        if receipt["task"] != task.name or not np.isclose(geometry_scale, 1.0):
            family = "reaching"
        else:
            q = np.asarray(receipt["joint_positions"], dtype=float).copy()
            target_q = np.asarray(receipt["joint_targets"], dtype=float).copy()
            reference_q = q.copy()
            delta = rng.uniform(-0.06, 0.06)
            q[0] += delta
            target_q[0] += delta
            transform = forward_kinematics(q, robot_model=model)["gripper_body"] @ np.linalg.inv(
                forward_kinematics(reference_q, robot_model=model)["gripper_body"]
            )
            body_pose = np.eye(4)
            body_pose[:3, 3] = receipt["object_pose_robot_base"][:3]
            body_pose[:3, :3] = Rotation.from_quat(receipt["object_pose_robot_base"][3:]).as_matrix()
            base = np.eye(4)
            base[:3, 3] = base_position
            base[:3, :3] = Rotation.from_euler("z", base_yaw).as_matrix()
            pose = base @ transform @ body_pose
            task.initial[receipt["object_name"]] = ActorState(pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat())
    witnesses = []
    for point in task_landmarks(task):
        witness = solve_position(point, base_position, base_yaw, geometry_scale, robot_model=model)
        if witness is None:
            raise UnreachableReset(f"Sampled base cannot reach {family} reset for {task.name}")
        witnesses.append(witness)
    return ResetRecord(task, q, target_q, family, np.asarray(witnesses))
