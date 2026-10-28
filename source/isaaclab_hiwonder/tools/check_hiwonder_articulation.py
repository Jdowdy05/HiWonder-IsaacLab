# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Headless PhysX qualification of the real converted USD at a translated support.

This checks backend joint discovery, fixed-base anchoring, sampled link poses,
and finite driven motion. It is not a contact/grasp or physical robot test.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch
from isaaclab_hiwonder.assets.kinematics import forward_kinematics
from isaaclab_hiwonder.assets.registration import JOINT_NAMES, NOMINAL_JOINT_POSITIONS_RAD, USD_PATH
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


def main():
    cfg = sim_utils.SimulationCfg(dt=1 / 240, device=args.device, use_fabric=False)
    with launch_simulation(cfg, args):
        sim = sim_utils.SimulationContext(cfg)
        translation = np.array([0.45, 0.2, 0.72])
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(USD_PATH),
                    rigid_props=PhysxRigidBodyCfg(disable_gravity=False),
                    articulation_props=PhysxArticulationCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(translation), joint_pos=dict(zip(JOINT_NAMES, NOMINAL_JOINT_POSITIONS_RAD))
                ),
                actuators={
                    "all": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=8.0,
                        damping=0.35,
                        joint_effort_limit=1.0,
                        joint_velocity_limit=5.0,
                    )
                },
            )
        )
        sim.reset()
        expected_names = set(JOINT_NAMES)
        if set(robot.joint_names) != expected_names:
            raise RuntimeError(f"Backend joint names disagree: {robot.joint_names}")
        order = [robot.joint_names.index(name) for name in JOINT_NAMES]
        max_position_error, max_base_drift = 0.0, 0.0
        samples = [
            list(NOMINAL_JOINT_POSITIONS_RAD),
            [0.2, -0.15, 0.1, -0.1, 0.15, -0.2],
            [-0.2, 0.1, -0.1, 0.15, -0.2, 0.1],
        ]
        records = []
        for sample in samples:
            target = torch.zeros((1, 6), device=sim.device)
            target[0, order] = torch.tensor(sample, device=sim.device)
            robot.write_joint_position_to_sim_index(position=target)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target))
            robot.actuators.target_command.set_position_index(value=target)
            for _ in range(80):
                robot.write_data_to_sim()
                sim.step()
                robot.update(cfg.dt)
            q = robot.data.joint_pos.torch[0, order].detach().cpu().numpy()
            body_pose = robot.data.body_link_pose_w.torch[0].detach().cpu().numpy()
            if not np.isfinite(q).all() or not np.isfinite(body_pose).all():
                raise RuntimeError("Nonfinite articulation state")
            expected = forward_kinematics(q)
            errors = {
                name: float(np.linalg.norm(body_pose[i, :3] - translation - expected[name][:3, 3]))
                for i, name in enumerate(robot.body_names)
            }
            max_position_error = max(max_position_error, max(errors.values()))
            drift = np.linalg.norm(body_pose[robot.body_names.index("base"), :3] - translation)
            max_base_drift = max(max_base_drift, float(drift))
            records.append(dict(requested_q=sample, observed_q=q.tolist(), body_position_errors_m=errors))
        if max_position_error > 0.002 or max_base_drift > 0.001:
            raise RuntimeError(f"Registration/fixed-base mismatch: body {max_position_error}m, base {max_base_drift}m")
        result = dict(
            status="PASS_PHYSX_ARTICULATION_KINEMATICS",
            steps=240,
            physics_dt=cfg.dt,
            support_translation_m=translation.tolist(),
            backend_joint_order=robot.joint_names,
            body_names=robot.body_names,
            maximum_body_position_error_m=max_position_error,
            maximum_base_drift_m=max_base_drift,
            sampled_poses=records,
            limitations=(
                "Self-collision disabled; no tabletop/contact/grasp qualification; estimated dynamics and joint limits"
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
