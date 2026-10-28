# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute task physics/reset contracts in isolated SDK processes.

No learned policy or physical robot is used. Passing means the actual scene,
measured observations, physics steps and indexed resets execute with finite state;
it does not establish that a policy solves the manipulation task.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def native_diagnostic_snapshot(env):
    """Optional synchronized copies for comparing eager/graph initialization."""
    if not getattr(env, "hiwonder_native_diagnostics", False):
        return {}
    import numpy as np
    from isaaclab_hiwonder.tasks.vbd_contact import NewtonHiWonderVBDManager
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    arrays = {"native_all_body_labels": np.array(model.body_label)}
    # Read-only, version-pinned diagnostics; these do not change solver arrays.
    solver = NewtonHiWonderVBDManager._solver
    solver_names = tuple(
        name
        for name, value in vars(solver).items()
        if hasattr(value, "numpy")
        and any(token in name for token in ("mass", "inertia", "rho", "penalty", "material", "joint", "body_q_prev"))
    )
    owners = {
        "solver": (solver, solver_names),
        "model": (
            model,
            (
                "body_q",
                "body_qd",
                "body_mass",
                "body_inv_mass",
                "body_inertia",
                "body_inv_inertia",
                "body_world",
                "gravity",
                "joint_type",
                "joint_parent",
                "joint_child",
                "joint_X_p",
                "joint_X_c",
                "joint_q_start",
                "joint_qd_start",
                "joint_target_q_start",
                "joint_axis",
                "joint_target_ke",
                "joint_target_kd",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_effort_limit",
                "joint_velocity_limit",
                "joint_friction",
                "joint_armature",
                "shape_body",
                "shape_transform",
                "shape_geo_scale",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_mu",
                "shape_collision_filter_pairs",
            ),
        ),
        "state0": (NewtonManager.get_state_0(), ("body_q", "body_qd", "body_f", "joint_q", "joint_qd")),
        "state1": (NewtonManager.get_state_1(), ("body_q", "body_qd", "body_f", "joint_q", "joint_qd")),
        "control": (NewtonManager.get_control(), ("joint_target_q", "joint_target_qd", "joint_f")),
        "motors": (
            NewtonHiWonderVBDManager.get_robot_motors(),
            (
                "parameters",
                "targets",
                "feedforward",
                "inertia_floor",
                "applied_effort",
                "q_indices",
                "v_indices",
                "t_indices",
            ),
        ),
    }
    for prefix, (owner, names) in owners.items():
        for name in names:
            value = getattr(owner, name, None)
            if value is not None and hasattr(value, "numpy"):
                arrays[f"diagnostic_{prefix}_{name}"] = value.numpy().copy()
    arrays["diagnostic_solver_dt"] = np.array(NewtonManager.get_solver_dt())
    arrays["diagnostic_use_cuda_graph"] = np.array(env.cfg.sim.physics.use_cuda_graph)
    return arrays


def camera_snapshot(env, definition, output_directory, task, label):
    import numpy as np
    from isaaclab_hiwonder.assets.kinematics import forward_kinematics
    from isaaclab_hiwonder.tasks.api import rotation_matrix
    from PIL import Image

    # state_debug does not consume camera observations during environment steps.
    # Explicitly refresh the public sensor after rendering before measuring it.
    env.sim.render()
    env.scene["wrist_camera"].update(0.0, force_recompute=True)
    roots = env.robot.data.root_pose_w.torch.detach().cpu().numpy()
    joints = env.robot.data.joint_pos.torch[:, env.joint_order].detach().cpu().numpy()
    observed = env.scene["wrist_camera"].data.pos_w.torch.detach().cpu().numpy()
    scales = env.randomization.snapshot()["robot"]["geometry_scale"].numpy()
    expected = np.array(
        [
            root[:3] + rotation_matrix(root[3:]) @ forward_kinematics(q, float(scale))["camera_opencv"][:3, 3]
            for root, q, scale in zip(roots, joints, scales)
        ]
    )
    arrays = dict(
        robot_q=joints,
        robot_qd=env.robot.data.joint_vel.torch.detach().cpu().numpy(),
        robot_root=roots,
        expected_camera=expected,
        observed_camera=observed,
        camera_quat=env.scene["wrist_camera"].data.quat_w_ros.torch.detach().cpu().numpy(),
        robot_body_poses=env.robot.data.body_link_pose_w.torch.detach().cpu().numpy(),
        robot_body_names=np.array(env.robot.body_names),
    )
    if definition.deformables:
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        indices = [i for i, path in enumerate(model.body_label) if "/Room/Robot/" in path]
        arrays["native_robot_labels"] = np.array([model.body_label[i] for i in indices])
        arrays["native_robot_body_q"] = NewtonManager.get_state_0().body_q.numpy()[indices]
        arrays["native_robot_body_qd"] = NewtonManager.get_state_0().body_qd.numpy()[indices]
        arrays["native_robot_com"] = model.body_com.numpy()[indices]
        arrays["native_joint_q"] = NewtonManager.get_state_0().joint_q.numpy()
        arrays["native_joint_qd"] = NewtonManager.get_state_0().joint_qd.numpy()
        arrays.update(native_diagnostic_snapshot(env))
    np.savez_compressed(output_directory / (task.replace(":", "_") + "_camera_" + label + ".npz"), **arrays)
    frames = env.scene["wrist_camera"].data.output["rgb"].torch[..., :3].detach().cpu().numpy()
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(output_directory / f"{task.replace(':', '_')}_wrist_{label}_env{i}.png")
    errors = np.linalg.norm(expected - observed, axis=-1).tolist()
    if max(errors) > 0.002:
        raise RuntimeError(f"{label} camera/registered moving-arm transform mismatch: {errors}m")
    return errors


def imported_rest_errors(env):
    """Check imported rest vertices, separately from the mutable reset state."""
    import numpy as np
    from isaaclab_newton.physics import NewtonManager

    if not any(
        spec.kind != "cable"
        for _, definition, _, _ in env.cfg.hiwonder_metadata["native_task_entries"]
        for spec in definition.deformables
    ):
        return {}
    model_rest = NewtonManager.get_model().particle_q.numpy()
    ranges = NewtonManager.backend.deformable_ranges
    metrics = {}
    for i, imported, _, paths in env.cfg.hiwonder_metadata["native_task_entries"]:
        for spec in imported.deformables:
            if spec.kind == "cable":
                continue
            matches = [
                value
                for key, value in ranges.items()
                if key == paths[spec.name] or key.startswith(paths[spec.name] + "/")
            ]
            if len(matches) != 1 or matches[0][1] != len(spec.vertices):
                raise RuntimeError(f"Exact imported rest mapping unavailable for env{i}/{spec.name}")
            offset, count, _ = matches[0]
            expected = spec.vertices + env.runtime.offsets[i]
            error = float(np.max(np.abs(model_rest[offset : offset + count] - expected)))
            metrics[f"{i}/{spec.name}/imported_rest_error_m"] = error
            if error > 1e-5:
                raise RuntimeError(f"Imported rest geometry differs for env{i}/{spec.name}: {error}m")
    return metrics


def check_finite_observations(observations, reward):
    import torch

    if not all(torch.isfinite(value).all() for value in observations.values()) or not torch.isfinite(reward).all():
        raise RuntimeError("Nonfinite observation or reward")


def verify_live_rod_inertias(env, output_directory, task):
    import numpy as np
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    labels = {label: i for i, label in enumerate(model.body_label)}
    inertia, mass = model.body_inertia.numpy(), model.body_mass.numpy()
    records = env.native_binding.authored_rod_inertias
    for label, record in records.items():
        np.testing.assert_array_equal(inertia[labels[label]], record["inertia"].astype(np.float32))
        np.testing.assert_allclose(mass[labels[label]], record["mass"], rtol=2e-6, atol=0)
    receipt = env.native_binding.rod_inertia_receipt
    audit_path = output_directory / (task.replace(":", "_") + "_rod_inertia.json")
    audit_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {
        "status": receipt.get("status", "NO_ROD_BODIES"),
        "body_count": len(records),
        "live_tensors_verified_after_reset_steps": True,
        "audit_file": str(audit_path),
    }


def retained_physics_snapshot(env):
    """Measured retained-world state, including twist invisible in cable nodes."""
    import re

    import numpy as np

    def array(value):
        if hasattr(value, "torch"):
            value = value.torch
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value).copy()

    snapshot = {
        "robot_q": array(env.robot.data.joint_pos)[1:],
        "robot_qd": array(env.robot.data.joint_vel)[1:],
        "robot_root": array(env.robot.data.root_pose_w)[1:],
        "robot_body_poses": array(env.robot.data.body_link_pose_w)[1:],
    }
    for env_id, state in enumerate(env.runtime.states()):
        if env_id == 0:
            continue
        prefix = f"env{env_id}"
        for name, actor in state.actors.items():
            snapshot[f"{prefix}_actor_{name}"] = np.r_[
                actor.position, actor.orientation, actor.linear_velocity, actor.angular_velocity
            ]
        for field in ("deformable_positions", "deformable_velocities", "crease_rest_angles"):
            for name, value in getattr(state, field).items():
                snapshot[f"{prefix}_{field}_{name}"] = array(value)
        for field in ("joint_positions", "joint_velocities", "mechanisms"):
            for name, value in getattr(state, field).items():
                snapshot[f"{prefix}_{field}_{name}"] = array(value)
        snapshot[f"{prefix}_contacts"] = np.asarray(sorted(state.contacts), dtype="U")
    if env.native_binding is not None:
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        indices = [
            index
            for index, label in enumerate(model.body_label)
            if (match := re.search(r"/env_(\d+)/", label)) and int(match[1]) >= 1
        ]
        state = NewtonManager.get_state_0()
        snapshot["native_all_retained_body_q"] = state.body_q.numpy()[indices].copy()
        snapshot["native_all_retained_body_qd"] = state.body_qd.numpy()[indices].copy()
        hook = env.native_binding.hook
        for (env_id, name), (start, end) in hook.surface_ranges.items():
            if env_id >= 1:
                snapshot[f"env{env_id}_{name}_plastic_rest"] = hook.rest.numpy()[start:end].copy()
                snapshot[f"env{env_id}_{name}_plastic_accumulated"] = hook.accumulated.numpy()[start:end].copy()
    return snapshot


def check_measured_task_health(env):
    """Check actual physical state after integration following an indexed reset."""
    import numpy as np
    from isaaclab_hiwonder.tasks.api import rotation_matrix

    metrics = {}
    for env_id, (definition, state) in enumerate(zip(env.runtime.definitions, env.runtime.states())):
        for name, actor in state.actors.items():
            if not np.isfinite(
                np.r_[actor.position, actor.orientation, actor.linear_velocity, actor.angular_velocity]
            ).all():
                raise RuntimeError(f"Nonfinite post-indexed-reset actor {env_id}/{name}")
        for spec in definition.deformables:
            nodes = state.deformable_positions[spec.name]
            if not np.isfinite(nodes).all() or not np.isfinite(state.deformable_velocities[spec.name]).all():
                raise RuntimeError(f"Nonfinite post-indexed-reset deformable {env_id}/{spec.name}")
            lengths = np.linalg.norm(nodes[spec.edges[:, 0]] - nodes[spec.edges[:, 1]], axis=1)
            rest = np.linalg.norm(spec.vertices[spec.edges[:, 0]] - spec.vertices[spec.edges[:, 1]], axis=1)
            strain = float(np.max(np.abs(lengths / rest - 1)))
            metrics[f"{env_id}/{spec.name}/max_edge_strain"] = strain
            if strain > 0.25:
                raise RuntimeError(f"Post-indexed-reset strain {env_id}/{spec.name}: {strain}")
            if spec.kind == "plastic_shell" and not np.isfinite(state.crease_rest_angles[spec.name]).all():
                raise RuntimeError("Nonfinite post-indexed-reset plastic state")
        if definition.name == "barrel_jack":
            plug = state.actors[definition.parameters["moving"]]
            frame = definition.deformables[0].attachments[0]["parent_frame"]
            expected = plug.position + rotation_matrix(plug.orientation) @ np.asarray(frame[:3])
            error = float(np.linalg.norm(state.deformable_positions["cable"][0] - expected))
            segment = state.actors["cable/segment_0"]
            relative = (
                rotation_matrix(plug.orientation) @ rotation_matrix(definition.parameters["attachment_orientation"])
            ).T @ rotation_matrix(segment.orientation)
            angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
            metrics[f"{env_id}/attachment_error_m"] = error
            metrics[f"{env_id}/attachment_error_rad"] = angle
            if error > 0.003 or angle > 0.15:
                raise RuntimeError(f"Post-indexed-reset cable bond failed: {error}m, {angle}rad")
    return metrics


def verify_indexed_reset(env, output_directory, task):
    import numpy as np
    import torch

    before = retained_physics_snapshot(env)
    start = env.sim.get_physics_step_count()
    env._reset_idx(torch.tensor([0], dtype=torch.long, device=env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)
    after_reset = env.sim.get_physics_step_count()
    after = retained_physics_snapshot(env)
    np.savez_compressed(
        output_directory / (task.replace(":", "_") + "_indexed_reset.npz"),
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
    )
    if before.keys() != after.keys():
        raise RuntimeError("Indexed reset changed retained physical-state schema")
    maximum = 0.0
    for key in before:
        if before[key].dtype.kind in "biufc" and before[key].size:
            maximum = max(maximum, float(np.max(np.abs(after[key].astype(float) - before[key].astype(float)))))
        np.testing.assert_array_equal(after[key], before[key], err_msg=f"Indexed reset changed retained {key}")
    if after_reset != start:
        raise RuntimeError(f"Indexed reset integrated physics: {start} -> {after_reset}")
    for _ in range(8):
        with torch.inference_mode():
            observations, reward, _, _, _ = env.step(torch.zeros((env.num_envs, 6), device=env.device))
        check_finite_observations(observations, reward)
    return {
        "reset_env_ids": [0],
        "retained_env_ids": list(range(1, env.num_envs)),
        "physics_steps_before_reset": start,
        "physics_steps_after_reset": after_reset,
        "integration_steps_during_reset": after_reset - start,
        "max_retained_state_delta": maximum,
        "post_indexed_reset_control_steps": 8,
        "post_indexed_reset_metrics": check_measured_task_health(env),
    }


def measure_deformable_fixture(index, state, spec, dump_failure):
    """Measure real edge strain and node speed without a reset replacing them."""
    import numpy as np

    positions = state.deformable_positions[spec.name]
    current = np.linalg.norm(positions[spec.edges[:, 0]] - positions[spec.edges[:, 1]], axis=1)
    rest = np.linalg.norm(spec.vertices[spec.edges[:, 0]] - spec.vertices[spec.edges[:, 1]], axis=1)
    strain = float(np.max(np.abs(current / rest - 1)))
    speed = float(np.max(np.linalg.norm(state.deformable_velocities[spec.name], axis=1)))
    if not np.isfinite(speed):
        dump_failure("velocity")
        raise RuntimeError(f"Nonfinite {spec.name} node velocity")
    if not np.isfinite(strain) or strain > 0.25:
        dump_failure("strain")
        raise RuntimeError(f"Unstable {spec.name} edge strain: {strain}")
    return {
        f"{index}/{spec.name}/max_edge_strain": strain,
        f"{index}/{spec.name}/max_node_speed_m_s": speed,
    }


class NativeMotionTrace:
    """Record the same instantaneous node-speed gate evaluated by the tasks."""

    def __init__(self, env, required_window, required_metric="raw"):
        self.env = env
        self.required_window = required_window
        self.required_metric = required_metric
        self.enabled = bool(env.runtime.definitions[0].deformables)
        self.samples = {}

    def record(self):
        if not self.enabled:
            return
        import numpy as np

        for index, state in enumerate(self.env.runtime.states()):
            configuration = int(self.env.config_ids[index])
            for spec in self.env.runtime.definitions[index].deformables:
                key = f"{index}/{spec.name}"
                speed = float(np.max(np.linalg.norm(state.deformable_velocities[spec.name], axis=1)))
                positions = np.asarray(state.deformable_positions[spec.name]).copy()
                if not np.isfinite(speed) or not np.isfinite(positions).all():
                    raise RuntimeError(f"Nonfinite native motion trace: {key}")
                episode_step = int(self.env.episode_length_buf[index])
                self.samples.setdefault(key, []).append((configuration, speed, positions, episode_step))

    @staticmethod
    def longest_qualified(values, episode_steps, discontinuity):
        consecutive = longest = 0
        for index, speed in enumerate(values):
            if index and discontinuity[index - 1]:
                consecutive = 0
            consecutive = consecutive + 1 if speed < 0.02 and episode_steps[index] > 0 else 0
            longest = max(longest, consecutive)
        return longest

    def write(self, output_directory, task):
        import math

        import numpy as np

        if not self.enabled:
            if self.required_window:
                raise ValueError("A quiet-window check requires a native deformable task")
            return {}
        dt = self.env.step_dt
        window = self.required_window or 0.5
        count = math.ceil(window / dt - 1e-9)
        summary, arrays = {}, {}
        for key, records in self.samples.items():
            configurations = np.array([row[0] for row in records])
            raw = np.array([row[1] for row in records])
            positions = np.stack([row[2] for row in records])
            episode_steps = np.array([row[3] for row in records])
            discontinuity = (configurations[1:] != configurations[:-1]) | (episode_steps[1:] <= episode_steps[:-1])
            displacement = np.max(np.linalg.norm(np.diff(positions, axis=0), axis=-1), axis=-1) / dt
            displacement[discontinuity] = np.nan
            control_speed = np.r_[np.nan, displacement]
            longest = self.longest_qualified(raw, episode_steps, discontinuity)
            longest_control = self.longest_qualified(control_speed, episode_steps, discontinuity)
            recent = raw[-count:]
            recent_control = control_speed[-count:]
            same_episode = (
                len(records) >= count
                and np.all(configurations[-count:] == configurations[-1])
                and np.all(np.diff(episode_steps[-count:]) > 0)
                and np.all(episode_steps[-count:] > 0)
            )
            raw_qualified = bool(same_episode and len(recent) == count and np.all(recent < 0.02))
            control_qualified = bool(same_episode and len(recent_control) == count and np.all(recent_control < 0.02))
            qualified = raw_qualified if self.required_metric == "raw" else control_qualified
            summary[key] = {
                "samples": len(records),
                "sample_hz": 1 / dt,
                "threshold_m_s": 0.02,
                "window_seconds": window,
                "window_raw_speed_min_m_s": float(recent.min()),
                "window_raw_speed_max_m_s": float(recent.max()),
                "window_fraction_below_threshold": float(np.mean(recent < 0.02)),
                "longest_consecutive_qualified_seconds": longest * dt,
                "window_control_speed_max_m_s": (
                    float(np.nanmax(recent_control)) if np.isfinite(recent_control).any() else None
                ),
                "window_control_fraction_below_threshold": float(np.mean(recent_control < 0.02)),
                "longest_consecutive_control_qualified_seconds": longest_control * dt,
                "final_raw_window_qualified": raw_qualified,
                "final_control_window_qualified": control_qualified,
                "required_metric": self.required_metric,
                "quiet_window_required": bool(self.required_window),
                "final_window_qualified": qualified,
            }
            prefix = key.replace("/", "_")
            arrays[prefix + "_configuration_ids"] = configurations
            arrays[prefix + "_episode_steps"] = episode_steps
            arrays[prefix + "_raw_node_speed_m_s"] = raw
            arrays[prefix + "_control_displacement_speed_m_s"] = displacement
            arrays[prefix + "_positions_m"] = positions
        stem = task.replace(":", "_") + "_native_motion"
        np.savez_compressed(output_directory / (stem + ".npz"), **arrays)
        (output_directory / (stem + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
        if self.required_window and (
            not summary or not all(value["final_window_qualified"] for value in summary.values())
        ):
            raise RuntimeError(f"Native resting fixture failed the continuous quiet window: {summary}")
        return summary


def install_failure_guard(env, output_directory, task, dump_failure):
    """Observe real failures before DirectRLEnv replaces their physical state."""
    import torch

    original_reset = env._reset_idx

    def reject_failed_reset(ids):
        ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        failed = ids[env.has_started[ids] & env.failures[ids]]
        if len(failed):
            details = {
                "task": task,
                "failed_env_ids": failed.tolist(),
                "configuration_ids": env.config_ids[failed].tolist(),
                "reset_family_ids": env.reset_family_ids[failed].tolist(),
                "control_step": int(env.common_step_counter),
                "episode_elapsed_seconds": [env.episodes[i].elapsed for i in failed.tolist()],
                "observation_boundary": "measured_before_automatic_reset",
            }
            failure_path = output_directory / (task.replace(":", "_") + "_failure.json")
            failure_path.write_text(json.dumps(details, indent=2) + "\n")
            dump_failure("pre_autoreset_failure")
            raise RuntimeError(f"Zero-action fixture failed before automatic reset: {details}")
        return original_reset(ids)

    env._reset_idx = reject_failed_reset


def child_command(args, name):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--task",
        name,
        "--steps",
        str(args.steps),
        "--num-envs",
        str(args.num_envs),
        "--device",
        args.device,
        "--output-directory",
        str(args.output_directory),
    ]
    if args.rgb:
        command.append("--rgb")
    if args.native_diagnostics:
        command.append("--native-diagnostics")
    if args.quiet_window:
        command.extend(("--quiet-window", str(args.quiet_window), "--quiet-metric", args.quiet_metric))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Catalog plus every garment/knot variant")
    parser.add_argument(
        "--quiet-window", type=float, default=0.0, help="Require final native quiet seconds below20mm/s"
    )
    parser.add_argument("--quiet-metric", choices=("raw", "control_frame"), default="raw")
    parser.add_argument(
        "--native-diagnostics", action="store_true", help="Synchronized initial/final native model/control snapshots"
    )
    parser.add_argument("--task", default="peg_insert")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.all:
        from isaaclab_hiwonder.tasks import GARMENTS, KNOT_VARIANTS, TASK_NAMES

        names = [name for name in TASK_NAMES if name not in ("cloth_fold", "rope_thread")]
        names += [f"cloth_fold:{variant}" for variant in GARMENTS]
        names += [f"rope_thread:{variant}" for variant in KNOT_VARIANTS]
        names += ["bottle_cap:close", "roll_ball:table"]
        results = []
        for name in names:
            command = child_command(args, name)
            with (args.output_directory / (name.replace(":", "_") + ".log")).open("w") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
            report = args.output_directory / (name.replace(":", "_") + ".json")
            result = json.loads(report.read_text()) if report.exists() else {"task": name, "status": "FAILED"}
            result["exit_code"] = process.returncode
            results.append(result)
            print(f"{name}: {result['status']}", flush=True)
        (args.output_directory / "catalog-results.json").write_text(json.dumps(results, indent=2) + "\n")
        if any(result["exit_code"] for result in results):
            raise SystemExit(1)
        return

    from isaaclab_hiwonder.tasks import build_task
    from isaaclab_hiwonder.tasks.native_backend import physics_config

    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg

    name, _, variant = args.task.partition(":")
    definition = build_task(name, variant or None)
    report_path = (args.output_directory / (args.task.replace(":", "_") + ".json")).resolve()
    if report_path.parent != args.output_directory.resolve():
        raise ValueError("Task report must stay inside the selected output directory")
    report_path.unlink(missing_ok=True)  # A failed rerun must not leave a stale PASS receipt.
    bootstrap = SimulationCfg(dt=1 / 240, render_interval=8, device=args.device, physics=physics_config(definition))
    with launch_simulation(bootstrap, {"device": args.device, "enable_cameras": True, "require_kit": True}):
        import torch
        from isaaclab_hiwonder.sim.env import make_cfg, make_env

        cfg = make_cfg(
            args.task,
            args.num_envs,
            seed=31,
            actor_mode="rgb" if args.rgb else "state_debug",
            output_directory=args.output_directory / args.task.replace(":", "_"),
        )
        cfg.sim.device = args.device
        env = make_env(cfg=cfg)
        env.hiwonder_native_diagnostics = args.native_diagnostics

        def dump_native_failure(reason):
            import numpy as np

            arrays = {}
            for i, measured in enumerate(env.runtime.states()):
                for name, actor in measured.actors.items():
                    arrays[f"env{i}_actor_{name.replace('/', '_')}"] = np.r_[
                        actor.position, actor.orientation, actor.linear_velocity, actor.angular_velocity
                    ]
                for spec in env.runtime.definitions[i].deformables:
                    prefix = f"env{i}_{spec.name}"
                    arrays[prefix + "_rest"] = spec.vertices
                    arrays[prefix + "_edges"] = spec.edges
                    arrays[prefix + "_measured"] = measured.deformable_positions[spec.name]
            for name, key in env.cfg.hiwonder_metadata["task_deformables"].items():
                asset = env.scene[key]
                if hasattr(asset.data, "segment_pose_w"):
                    arrays[name + "_segment_poses"] = asset.data.segment_pose_w.torch.detach().cpu().numpy()
                    arrays[name + "_segment_velocities"] = asset.data.segment_velocity_w.torch.detach().cpu().numpy()
            if env.native_binding is not None and env.native_binding.hook is not None:
                from isaaclab_newton.physics import NewtonManager

                model = NewtonManager.get_model()
                joints = np.array([i for i, path in enumerate(model.joint_label) if "/CableBond_" in path], dtype=int)
                arrays["bond_joint_indices"] = joints
                for name in ("joint_parent", "joint_child", "joint_X_p", "joint_X_c", "joint_world"):
                    arrays["bond_" + name] = getattr(model, name).numpy()[joints]
                pairs = np.r_[model.joint_parent.numpy()[joints], model.joint_child.numpy()[joints]]
                indices = np.unique(pairs[pairs >= 0])
                arrays["bond_body_indices"] = indices
                arrays["model_body_q_bonds"] = model.body_q.numpy()[indices]
                arrays["model_body_com_bonds"] = model.body_com.numpy()[indices]
                arrays["live_body_q_bonds"] = NewtonManager.get_state_0().body_q.numpy()[indices]
                arrays["live_body_qd_bonds"] = NewtonManager.get_state_0().body_qd.numpy()[indices]
            output = args.output_directory / (args.task.replace(":", "_") + "_" + reason + ".npz")
            np.savez_compressed(output, **arrays)

        # Automatic resets must not hide an unstable zero-action fixture.
        install_failure_guard(env, args.output_directory, args.task, dump_native_failure)

        try:
            observations, _ = env.reset()
            initial_camera_errors = camera_snapshot(env, definition, args.output_directory, args.task, "initial")
            native_rest_metrics = {}
            if definition.deformables:
                import numpy as np

                native_rest_metrics = imported_rest_errors(env)
                for i, measured in enumerate(env.runtime.states()):
                    for spec in env.runtime.definitions[i].deformables:
                        error = np.max(np.abs(measured.deformable_positions[spec.name] - spec.vertices))
                        if error > 1e-5:
                            dump_native_failure("reset_mismatch")
                            raise RuntimeError(
                                f"Environment {i} reset geometry differs from its own {spec.name}: {error}m"
                            )
            reward_total = torch.zeros(args.num_envs, device=env.device)
            terminal_count = 0
            motion_trace = NativeMotionTrace(env, args.quiet_window, args.quiet_metric)
            for _ in range(args.steps):
                # Match PPO collection, then deliberately reset/restore outside
                # inference mode below to qualify persistent-buffer ownership.
                with torch.inference_mode():
                    observations, reward, terminal, timeout, _ = env.step(
                        torch.zeros((args.num_envs, 6), device=env.device)
                    )
                check_finite_observations(observations, reward)
                reward_total += reward
                terminal_count += int((terminal | timeout).sum())
                motion_trace.record()
            native_motion = motion_trace.write(args.output_directory, args.task)
            camera_errors = camera_snapshot(env, definition, args.output_directory, args.task, "final")
            native_metrics = dict(native_rest_metrics)
            if definition.deformables:
                import numpy as np
                from isaaclab_hiwonder.tasks.api import rotation_matrix

                if env.native_binding is None or env.native_binding.hook is None:
                    raise RuntimeError("Native task constitutive binding was never initialized")
                measured = env.runtime.states()
                for i, state in enumerate(measured):
                    for spec in env.runtime.definitions[i].deformables:
                        native_metrics.update(measure_deformable_fixture(i, state, spec, dump_native_failure))
                        if spec.kind == "plastic_shell" and spec.name not in state.crease_rest_angles:
                            raise RuntimeError("Paper plastic crease state is missing")
                    if definition.name == "barrel_jack":
                        plug = state.actors[definition.parameters["moving"]]
                        attachment = definition.deformables[0].attachments[0]
                        expected = (
                            rotation_matrix(plug.orientation) @ np.asarray(attachment["parent_frame"][:3])
                            + plug.position
                        )
                        error = float(np.linalg.norm(state.deformable_positions["cable"][0] - expected))
                        segment = state.actors["cable/segment_0"]
                        relative = (
                            rotation_matrix(plug.orientation)
                            @ rotation_matrix(definition.parameters["attachment_orientation"])
                        ).T @ rotation_matrix(segment.orientation)
                        angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
                        native_metrics[f"{i}/attachment_error_m"] = error
                        native_metrics[f"{i}/attachment_error_rad"] = angle
                        if error > 0.003 or angle > 0.15:
                            dump_native_failure("attachment")
                            raise RuntimeError(f"Cable plug-frame attachment failed: {error}m, {angle}rad")
            # Checkpoint a running episode, including its motion/plastic history.
            state = env.state_dict()
            before = env.sampler.counts.clone()
            env.reset()  # Interrupted episodes must not train SGS.
            if not torch.equal(before, env.sampler.counts):
                raise RuntimeError("External reset incorrectly added an SGS outcome")
            env.load_state_dict(state)
            # Exercise integration after discontinuous reset/restore, not only
            # the immediate written state, to catch stale solver/contact history.
            for _ in range(8):
                with torch.inference_mode():
                    observations, reward, _, _, _ = env.step(torch.zeros((args.num_envs, 6), device=env.device))
                check_finite_observations(observations, reward)
            indexed_reset = verify_indexed_reset(env, args.output_directory, args.task) if args.num_envs > 1 else None
            report = {
                "task": args.task,
                "status": "PASS_PHYSICS_RESET_EXECUTION",
                "steps": args.steps,
                "post_reset_restore_steps": 8,
                "checkpoint_from_running_episode": True,
                "rollout_inference_mode": True,
                "num_envs": args.num_envs,
                "physics": definition.physics_family,
                "reward_sum": reward_total.tolist(),
                "terminal_count": terminal_count,
                "pre_autoreset_failure_guard": True,
                "unexpected_failure_count": 0,
                "bank_hash": env.bank_hash,
                "implementation_sha256": hashlib.sha256(
                    json.dumps(env.bank_definition["implementation_sha256"], sort_keys=True).encode()
                ).hexdigest(),
                "camera_fk_errors_m": camera_errors,
                "initial_camera_fk_errors_m": initial_camera_errors,
                "native_metrics": native_metrics,
                "native_motion": native_motion,
                "qualification": "execution_only_no_task_or_hardware_success_claim",
                "indexed_reset": indexed_reset,
            }
            if definition.deformables:
                report["native_rod_inertia"] = verify_live_rod_inertias(env, args.output_directory, args.task)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
