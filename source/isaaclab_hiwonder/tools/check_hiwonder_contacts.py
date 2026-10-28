# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Check real two-room shape-sort contact filters, aperture support and separation."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-directory", type=Path, required=True)
add_launcher_args(parser)
args = parser.parse_args()
args.require_kit = True

import torch
from isaaclab_hiwonder.config import HiWonderEnvConfig
from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.tasks.catalog import build_task
from isaaclab_hiwonder.tasks.native_backend import physics_config

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    definitions = [build_task("shape_sort", seed=i) for i in range(2)]
    profile = HiWonderEnvConfig(num_envs=2, task="shape_sort", actor_mode="state_debug")
    sim_cfg = SimulationCfg(dt=1 / 240, device=args.device, physics=physics_config(definitions[0]))
    with launch_simulation(sim_cfg, args):
        from isaaclab_hiwonder.sim.scene_config import build_scene
        from isaaclab_hiwonder.tasks.runtime import TaskRuntime

        from isaaclab.scene import InteractiveScene

        scene_cfg, metadata = build_scene(
            profile, definitions, RandomizationManager(2).snapshot(), args.output_directory
        )
        scene_cfg.wrist_camera = None
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(scene_cfg)
        scene.filter_collisions()
        sim.reset()
        runtime = TaskRuntime(scene, metadata, definitions)
        robot = scene["robot"]
        robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch)

        def advance(count):
            for _ in range(count):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim_cfg.dt)
            return runtime.states()

        initial = advance(160)
        for state in initial:
            for name in ("shape_0", "shape_1", "shape_2"):
                if not state.touching(name, "table"):
                    raise RuntimeError(f"Expected measured table support for {name}: {state.contacts}")
        shape = scene[metadata["task_body_assets"]["shape_0"]]
        seat = definitions[0].parameters["targets"]["shape_0"]
        center = torch.tensor(seat, device=sim.device) + scene.env_origins[0]
        center[2] += metadata["support_height_m"][0]
        pose = torch.cat((center, torch.tensor([0, 0, 0, 1], device=sim.device))).reshape(1, 7)
        env_ids = torch.tensor([0], device=sim.device)
        shape.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        shape.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=sim.device), env_ids=env_ids)
        seated = advance(160)
        if not seated[0].touching("shape_0", "slot_0") or seated[1].touching("shape_0", "slot_0"):
            raise RuntimeError(
                f"Socket-floor contact was missing or leaked across environments: {[s.contacts for s in seated]}"
            )
        if not seated[1].touching("shape_0", "table"):
            raise RuntimeError("Unchanged environment lost its independent table support")
        pose[0, 2] += 0.08
        shape.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        shape.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=sim.device), env_ids=env_ids)
        separated = advance(2)
        if separated[0].touching("shape_0", "slot_0") or separated[0].touching("shape_0", "table"):
            raise RuntimeError("Separated object retained stale contact evidence")
        filters = {}
        for actor, entry in metadata["task_contacts"].items():
            sensor = scene[entry["sensor_key"]]
            matrix = sensor.data.force_matrix_w.torch
            if tuple(matrix.shape) != (2, 1, len(entry["filter_labels"]), 3):
                raise RuntimeError(f"Unexpected actor contact matrix {actor}: {matrix.shape}")
            filters[actor] = dict(
                matrix_shape=list(matrix.shape),
                actor_labels=entry["filter_labels"],
                paths=entry["filter_paths"],
                resolved_filter_paths=sensor.contact_view.filter_paths,
            )
        receipt = dict(
            status="PASS_COMPOUND_ACTOR_CONTACT_FILTERS",
            environments=2,
            task="shape_sort",
            initial_contacts=[sorted(state.contacts) for state in initial],
            seated_contacts=[sorted(state.contacts) for state in seated],
            separated_contacts=[sorted(state.contacts) for state in separated],
            filters=filters,
            geometry_changed=False,
            aperture_floor_support_verified=True,
            independent_environment_and_separation_verified=True,
        )
        (args.output_directory / "contact-check.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
