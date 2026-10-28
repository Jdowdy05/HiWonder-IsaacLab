# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RGB/proprioceptive RL with measured task outcomes and fixed-bank SGS.

Geometry/material cohorts are built once per launch. Episode resets randomize
supported base poses, actuator channels and image nuisance parameters. Rebuilding
with another seed redraws the structural cohort without silently scaling live
physics shapes or their calibrated camera frames.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.math import quat_apply

from isaaclab_hiwonder.assets.registration import ASSET_DIR, JOINT_NAMES, LINKS
from isaaclab_hiwonder.config import HiWonderEnvConfig, TaskSwitchConfig, resolve_backend
from isaaclab_hiwonder.images import ImageRandomizer
from isaaclab_hiwonder.lighting import camera_pose_from_link
from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.rewards import RewardShaper
from isaaclab_hiwonder.sgs import SuccessGuidedSampler, cohort_probabilities
from isaaclab_hiwonder.tasks.catalog import TASK_NAMES, build_task
from isaaclab_hiwonder.tasks.native_backend import NativeTaskBinding, physics_config
from isaaclab_hiwonder.tasks.outcomes import TaskEpisode

from .camera_channel import CameraChannel
from .checkpoint import structural_snapshot_hash, task_definition_hash
from .control import MotorChannel
from .history import FrameHistory
from .lighting import SceneLightingRig
from .observations import pack_privileged_rows
from .reachability import reject_unreachable_bases
from .resets import UnreachableReset, load_grasp_library, make_reset
from .support import bootstrap_task_cohorts, physical_nuisance_overlaps


def make_cfg(
    task=None,
    num_envs=4,
    seed=0,
    actor_mode="rgb",
    output_directory=None,
    profile=None,
    sampling="sgs",
    split="train",
    bank_size=32768,
    **kwargs,
):
    from .scene_config import build_scene

    if task is None:
        task = profile.task if profile is not None else "peg_insert"
    name, _, variant = task.partition(":")
    name = {"peg_in_hole": "peg_insert", "color_cube_stacking": "cube_stack"}.get(name, name)
    if name not in TASK_NAMES:
        raise ValueError(f"Unknown task {name}. Choose {TASK_NAMES}")
    profile = profile or HiWonderEnvConfig(
        num_envs=num_envs, seed=seed, task=name, actor_mode=actor_mode, physics_substeps=8
    )
    random_config = replace(profile.randomization, task=TaskSwitchConfig(task_names=(name,), weights=(1.0,)))
    profile = replace(profile, randomization=random_config, task=name + (":" + variant if variant else ""))
    manager = RandomizationManager(
        profile.num_envs, profile.randomization, seed=profile.seed, device="cpu", bootstrap_attempts=32
    )
    definitions = [
        build_task(name, variant or None, seed=profile.seed + i, task_parameters=profile.task_parameters)
        for i in range(profile.num_envs)
    ]
    if profile.episode_seconds is not None:
        for definition in definitions:
            definition.horizon_seconds = profile.episode_seconds
    backend = resolve_backend(profile.backend, definitions[0].physics_family)
    profile = replace(profile, backend=backend)
    support = bootstrap_task_cohorts(manager, definitions)
    witnesses = reject_unreachable_bases(manager, definitions)
    directory = Path(output_directory or tempfile.mkdtemp(prefix="hiwonder_scene_"))
    scene, metadata = build_scene(profile, definitions, manager.snapshot(), directory)
    metadata["task_support"] = support
    metadata["structural_snapshot_sha256"] = structural_snapshot_hash(manager.snapshot())
    chosen = manager.snapshot()["robot"]["geometry_variant_id"].tolist()
    variants = [profile.randomization.robot.geometry_variants[index] for index in chosen]
    metadata["task_definition_sha256"] = [task_definition_hash(definition) for definition in definitions]
    metadata["robot_asset_sha256"] = [
        hashlib.sha256(
            (
                ASSET_DIR
                / "generated"
                / ("hiwonder_soarm101" + ("" if variant == "nominal" else "_" + variant) + ".usdc")
            ).read_bytes()
        ).hexdigest()
        for variant in variants
    ]
    metadata["position_ik_witnesses"] = {i: value.tolist() for i, value in witnesses.items()}
    sim = SimulationCfg(
        dt=1 / (profile.control_hz * profile.physics_substeps),
        render_interval=profile.physics_substeps,
        physics=physics_config(definitions[0]),
        # PhysX camera FrameView must consume live Fabric transforms. Its USD
        # fallback reads default-time transforms, which remain at initialization.
        use_fabric=not bool(definitions[0].deformables),
    )
    cfg = DirectRLEnvCfg(
        decimation=profile.physics_substeps,
        episode_length_s=profile.episode_seconds or definitions[0].horizon_seconds,
        action_space=6,
        observation_space={
            "policy_image": [3 * profile.frame_stack, profile.camera.height, profile.camera.width]
            if profile.actor_mode == "rgb"
            else 18,
            "proprio": 18,
            "command": 40,
            "critic": 256,
        },
        state_space=0,
        sim=sim,
        scene=scene,
        seed=profile.seed,
        num_rerenders_on_reset=1 if profile.actor_mode == "rgb" else 0,
    )
    cfg.hiwonder_profile, cfg.hiwonder_metadata = profile, metadata
    cfg.hiwonder_definitions, cfg.hiwonder_randomization = definitions, manager
    cfg.hiwonder_sampling, cfg.hiwonder_split = sampling, split
    cfg.hiwonder_grasp_library = load_grasp_library(profile.grasp_library) if profile.grasp_library else None
    cfg.hiwonder_grasp_library_sha256 = (
        hashlib.sha256(Path(profile.grasp_library).read_bytes()).hexdigest() if profile.grasp_library else None
    )
    if bank_size < profile.num_envs:
        raise ValueError("SGS bank must contain at least one reset per structural cohort")
    cfg.hiwonder_bank_size = bank_size
    cfg.hiwonder_directory = str(directory)
    return cfg


def make_env(task=None, num_envs=4, seed=0, actor_mode="rgb", cfg=None, **kwargs):
    return HiWonderEnv(cfg or make_cfg(task, num_envs, seed, actor_mode, **kwargs))


class HiWonderEnv(DirectRLEnv):
    reset_family_names = ("reaching", "near_goal", "stable_grasp")

    def _setup_scene(self):
        metadata = self.cfg.hiwonder_metadata
        entries = [entry for entry in metadata.get("native_task_entries", []) if entry[1].deformables]
        self.native_binding = NativeTaskBinding(entries) if entries else None

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        from isaaclab_hiwonder.tasks.runtime import TaskRuntime

        self.profile = cfg.hiwonder_profile
        self.definitions = cfg.hiwonder_definitions
        self.randomization = cfg.hiwonder_randomization
        self.robot = self.scene["robot"]
        self.joint_order = [self.robot.joint_names.index(name) for name in JOINT_NAMES]
        self.tool_body_id = self.robot.body_names.index("gripper_body")
        self.tool_offset = torch.as_tensor(
            (np.array([-2.0, -181.0, 207.0]) - np.asarray(LINKS["gripper_body"]["origin_mm"])) * 0.001,
            device=self.device,
            dtype=torch.float32,
        )
        self.runtime = TaskRuntime(self.scene, cfg.hiwonder_metadata, self.definitions)
        self.runtime.native_binding = self.native_binding
        self.sampler = SuccessGuidedSampler(cfg.hiwonder_bank_size, seed=self.profile.seed, device=self.device)
        package = Path(__file__).resolve().parents[1]
        self.bank_definition = {
            "version": "cohort-reset-families-v3-commanded-goals",
            "profile": asdict(self.profile),
            "variant": self.definitions[0].variant,
            "size": cfg.hiwonder_bank_size,
            "cohort_rule": "id % num_envs",
            "seed_rule": "seed + id*1009 + 7919",
            "loose_xy_jitter_m": [-0.008, 0.008],
            "arm_jitter_rad": [-0.035, 0.035],
            "task_definition_sha256": cfg.hiwonder_metadata["task_definition_sha256"],
            "robot_asset_sha256": cfg.hiwonder_metadata["robot_asset_sha256"],
            "robot_collision_profiles": cfg.hiwonder_metadata["robot_collision_profiles"],
            "reset_algorithm_sha256": hashlib.sha256(Path(__file__).with_name("resets.py").read_bytes()).hexdigest(),
            "structural_snapshot_sha256": cfg.hiwonder_metadata["structural_snapshot_sha256"],
            "reset_families": self.profile.reset_families,
            "grasp_library_sha256": cfg.hiwonder_grasp_library_sha256,
            "implementation_sha256": {
                path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(package.rglob("*.py"))
            },
            "backend_versions": {
                name: importlib.metadata.version(name) for name in ("isaacsim", "newton", "warp-lang")
            },
        }
        self.bank_hash = hashlib.sha256(json.dumps(self.bank_definition, sort_keys=True).encode()).hexdigest()
        self.config_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_family_ids = torch.zeros_like(self.config_ids)
        self.episodes = [TaskEpisode(d) for d in self.definitions]
        self.reward_shaper = RewardShaper(self.num_envs, self.profile.reward)
        self.episode_reward_components = {}
        self.actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.targets = self.robot.data.default_joint_pos.torch.clone()
        self.channel = MotorChannel(
            self.num_envs, 6, self.step_dt, self.profile.randomization.actuator.delay_seconds[1], self.device
        )
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success = torch.zeros_like(self.completed)
        self.rewards = torch.zeros(self.num_envs, device=self.device)
        self.failures = torch.zeros_like(self.completed)
        self.timeouts = torch.zeros_like(self.completed)
        self.has_started = torch.zeros_like(self.completed)
        self.result_step = -1
        self.image_randomizer = ImageRandomizer(self.randomization.image_config, self.randomization.smudge_config)
        self.image_parameters = self.image_randomizer.sample_from_snapshot(
            self.randomization.snapshot(),
            self.profile.camera.height,
            self.profile.camera.width,
            device=self.device,
        )
        self.observation_generator = torch.Generator(device=self.device).manual_seed(self.profile.seed + 103)
        self.camera_channel = CameraChannel(
            self.profile.camera, self.num_envs, self.device, self.profile.seed + 107, control_hz=self.profile.control_hz
        )
        self.frame_history = (
            FrameHistory(
                self.num_envs,
                self.profile.frame_stack,
                3,
                self.profile.camera.height,
                self.profile.camera.width,
                self.device,
            )
            if self.profile.actor_mode == "rgb"
            else None
        )
        self._lighting_mount_translation = (
            torch.tensor(self.profile.camera.optical_translation_m)
            * self.randomization.snapshot(groups=("robot",))["robot"]["geometry_scale"][:, None]
        )
        self._lighting_mount_rotation = torch.tensor(self.profile.camera.optical_quaternion_xyzw).repeat(
            self.num_envs, 1
        )
        room_control = self.profile.randomization.controls.room
        lighting_config = self.profile.randomization.lighting
        if not room_control.enabled:
            lighting_config = replace(lighting_config, profile="stationary", motion_mode="room_sweep")
        self.lighting = SceneLightingRig(
            self.sim.stage,
            cfg.hiwonder_metadata["room_paths"],
            cfg.hiwonder_metadata["fill_light_paths"],
            lighting_config,
            self.profile.camera,
            self.scene.env_origins.cpu(),
            torch.tensor(cfg.hiwonder_metadata["support_height_m"]),
            frozen=room_control.frozen or not room_control.enabled,
        )
        self._update_lights(self.randomization.snapshot(groups=("room",))["room"])

    def _pre_physics_step(self, actions):
        # RSL-RL retains each returned log dictionary until its optimizer update.
        self.extras["log"] = dict(self.extras.get("log", {}))
        self.extras.pop("completed_env_ids", None)
        self.extras.pop("completed_successes", None)
        self.extras.pop("completed_family_ids", None)
        self.extras.pop("completed_config_ids", None)
        for key in list(self.extras.get("log", {})):
            if key.startswith("Metrics/success_rate") or key.startswith("Reward/"):
                self.extras["log"].pop(key)
        self.actions.copy_(actions.clamp(-1, 1))
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.joint_order]
        # Preserve a commanded grasp squeeze instead of replacing it with measured contact-limited q.
        # This increment references the delayed channel output; it is a discrete latency approximation.
        requested = (self.targets[:, self.joint_order] + self.actions * 0.12).clamp(limits[..., 0], limits[..., 1])
        sample = self.randomization.snapshot(groups=("actuator",))["actuator"]
        target = self.channel.step(
            requested, sample["delay_seconds"].to(self.device), sample["backlash_radians"].to(self.device)
        )
        self.targets[:, self.joint_order] = target
        wrist_view = self.profile.randomization.lighting.motion_mode == "wrist_view"
        snapshot = self.randomization.step(self.step_dt, groups=() if wrist_view else ("room",))
        if not wrist_view:
            self._update_lights(snapshot["room"])

    def _update_lights(self, room, env_ids=None):
        """Apply deterministic room-owned lights, using registered wrist poses for targeting."""
        camera_pose = None
        if self.profile.randomization.lighting.motion_mode == "wrist_view":
            link_pose = self.robot.data.body_link_pose_w.torch[:, self.tool_body_id].detach().cpu()
            camera_pose = camera_pose_from_link(
                link_pose, self._lighting_mount_translation, self._lighting_mount_rotation
            )
        self.lighting.update(room, camera_pose=camera_pose, env_ids=env_ids)

    def set_lighting_preview(self, controls: dict | None) -> dict:
        """Set inspection-only light overrides; never change a training sampler's distribution.

        None clears manual overrides. Saved profiles retain the sampled domains;
        the inspection tool can export a separate partial profile for deliberate training.
        """
        if self.cfg.hiwonder_split != "heldout":
            raise ValueError("Live lighting preview controls require a heldout inspection environment")
        room_control = self.profile.randomization.controls.room
        if controls is not None and (room_control.frozen or not room_control.enabled):
            raise ValueError("Live lighting changes require an enabled, unfrozen room control")
        overrides = self.lighting.set_preview(controls)
        self._update_lights(self.randomization.snapshot(groups=("room",))["room"])
        return {
            "overrides": overrides,
            "profile": asdict(self.profile.randomization.lighting),
            "motion_mode": self.profile.randomization.lighting.motion_mode,
            "preview_only": True,
        }

    def _apply_action(self):
        self.robot.actuators.target_command.set_position_index(value=self.targets)
        if self.native_binding is not None:
            from ..tasks.vbd_contact import NewtonHiWonderVBDManager

            NewtonHiWonderVBDManager.get_robot_motors().set_targets(self.targets[:, self.joint_order])
        self.runtime.apply_forces(self.physics_dt)

    def _evaluate(self):
        if self.result_step == self.common_step_counter:
            return
        states = self.runtime.states()
        poses = self.robot.data.body_link_pose_w.torch[:, self.tool_body_id]
        scales = self.randomization.snapshot(groups=("robot",))["robot"]["geometry_scale"].to(self.device)
        tool_world = poses[:, :3] + quat_apply(poses[:, 3:7], self.tool_offset[None] * scales[:, None])
        tool = (
            (tool_world - torch.as_tensor(self.runtime.offsets, device=self.device, dtype=torch.float32))
            .detach()
            .cpu()
            .numpy()
        )
        qd = self.robot.data.joint_vel.torch[:, self.joint_order]
        effort = self.robot.actuators.applied_effort.torch[:, self.joint_order]
        if self.native_binding is not None:
            import warp as wp

            from ..tasks.vbd_contact import NewtonHiWonderVBDManager

            effort = wp.to_torch(NewtonHiWonderVBDManager.get_robot_motors().applied_effort).reshape(self.num_envs, 6)
        # Preserve each original six-joint reduction while collecting the small
        # scalar results in one transfer instead of synchronizing per environment.
        costs = (
            torch.stack(
                [
                    torch.stack((self.actions[i].square().mean(), qd[i].square().mean(), effort[i].square().mean()))
                    for i in range(self.num_envs)
                ]
            )
            .detach()
            .cpu()
            .numpy()
        )
        for i, (episode, state) in enumerate(zip(self.episodes, states)):
            result = episode.evaluate(state, self.step_dt, action_cost=float(costs[i, 0]))
            reward, components = self.reward_shaper.evaluate(
                i,
                episode.task,
                result,
                state,
                tool[i],
                self.step_dt,
                action_cost=float(costs[i, 0]),
                velocity_cost=float(costs[i, 1]),
                effort_cost=float(costs[i, 2]),
            )
            self.rewards[i], self.success[i] = reward, result.success
            for name, value in components.items():
                if name not in self.episode_reward_components:
                    # Logging buffers also serve explicit resets outside rollout
                    # inference mode. Keep their persistent storage writable there.
                    with torch.inference_mode(False):
                        self.episode_reward_components[name] = torch.zeros(self.num_envs, device=self.device)
                self.episode_reward_components[name][i] += value
            self.failures[i], self.timeouts[i] = result.failure, result.truncated
            self.completed[i] = result.terminated or result.truncated
        self.last_states = states
        self.result_step = self.common_step_counter
        self.extras["episode_success"] = self.success.clone()

    def _get_dones(self):
        self._evaluate()
        # A task success/failure remains terminal even on the horizon boundary;
        # RSL-RL bootstraps truncated transitions, so the distinction is material.
        terminated = self.completed & ~self.timeouts
        self.timeouts.copy_((self.timeouts | (self.episode_length_buf >= self.max_episode_length)) & ~terminated)
        self.completed |= self.timeouts
        return terminated, self.timeouts

    def _get_rewards(self):
        self._evaluate()
        return self.rewards

    def _get_observations(self):
        pending = getattr(self, "_reset_camera_env_ids", set())
        if pending:
            from .reset_publication import publish_reset_body_poses

            ids = torch.tensor(sorted(pending), device=self.device, dtype=torch.long)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.sim.render()
            publish_reset_body_poses(self, ids)
            self._update_lights(self.randomization.snapshot(groups=("room",))["room"], env_ids=ids.cpu())
            self.scene["wrist_camera"].reset(ids)
            self._reset_camera_env_ids = set()
        q = self.robot.data.joint_pos.torch[:, self.joint_order]
        qd = self.robot.data.joint_vel.torch[:, self.joint_order]
        nuisance = self.randomization.snapshot()
        if not pending and self.profile.randomization.lighting.motion_mode == "wrist_view":
            # Lazy camera capture follows this call. Aim from the completed physics
            # interval rather than the pose preceding its eight physics ticks.
            self._update_lights(nuisance["room"])
        a = nuisance["actuator"]
        noise = torch.randn(q.shape, device=self.device, generator=self.observation_generator)
        observed = q + a["encoder_bias_radians"].to(self.device) + noise * a["encoder_noise_std"].to(self.device)
        proprio = torch.cat((observed, qd, self.actions), -1)
        if self.profile.actor_mode == "rgb":
            raw = self.scene["wrist_camera"].data.output["rgb"].torch[..., :3]
            raw = self.camera_channel.apply(raw, frame_ids=self.scene["wrist_camera"].frame.torch)
            offset = nuisance["smudge"]["offset"].to(self.device)
            pixels = self.image_randomizer.apply(raw, self.image_parameters, smudge_offset=offset)
            policy = self.frame_history.push(pixels.permute(0, 3, 1, 2).float() / 255.0)
        else:
            policy = proprio.clone()
        command_host = torch.zeros((self.num_envs, 40), device="cpu")
        # Gather small live device fields once. Their values are then packed
        # together with the host task summaries and nuisance records below.
        robot_host = (
            torch.cat((q, qd, self.robot.data.root_pose_w.torch, self.camera_channel.age_seconds[:, None]), dim=1)
            .detach()
            .cpu()
        )
        joint_count = q.shape[1]
        privileged_rows = []
        states = (
            self.last_states
            if self.result_step == self.common_step_counter and self.cfg.events is None
            else self.runtime.states()
        )
        for i, (definition, state) in enumerate(zip(self.definitions, states)):
            command_host[i, TASK_NAMES.index(definition.name)] = 1
            goal = torch.as_tensor(definition.goal, device="cpu")
            if goal.ndim != 1 or goal.numel() > 20 or not torch.isfinite(goal).all():
                raise ValueError(f"Public goal for {definition.name} must fit20 finite command values")
            command_host[i, 20 : 20 + goal.numel()] = goal

            parts = [
                ("robot/joint_position", robot_host[i, :joint_count]),
                ("robot/joint_velocity", robot_host[i, joint_count : 2 * joint_count]),
                ("robot/root_pose", robot_host[i, 2 * joint_count : -1]),
                ("command", command_host[i]),
            ]
            episode = self.episodes[i]
            parts.extend(
                [
                    ("episode/stage", episode.stage),
                    ("episode/held_seconds", episode.held),
                    ("episode/elapsed_seconds", episode.elapsed),
                    ("episode/stage_held_seconds", episode.stage_held),
                ]
            )
            for name in (
                "effort_limit",
                "stiffness",
                "damping",
                "friction",
                "armature",
                "velocity_limit",
                "delay_seconds",
                "backlash_radians",
                "encoder_bias_radians",
                "encoder_noise_std",
            ):
                parts.append(("actuator/" + name, nuisance["actuator"][name][i]))
            for name in ("mass_scale", "inertia_scale", "geometry_scale"):
                parts.append(("robot/" + name, nuisance["robot"][name][i]))
            for name in (
                "width",
                "depth",
                "height",
                "thickness",
                "edge_radius",
                "material_friction",
                "material_restitution",
            ):
                parts.append(("table/" + name, nuisance["table"][name][i]))
            parts.append(("camera/age_seconds", robot_host[i, -1]))
            for name in sorted(state.actors):
                actor = state.actors[name]
                parts.append(
                    (
                        "actor/" + name,
                        np.concatenate(
                            (actor.position, actor.orientation, actor.linear_velocity, actor.angular_velocity)
                        ),
                    )
                )
            for name in sorted(state.joint_positions):
                parts.append(("joint_position/" + name, state.joint_positions[name]))
            for name in sorted(state.joint_velocities):
                parts.append(("joint_velocity/" + name, state.joint_velocities[name]))
            for name in sorted(state.mechanisms):
                parts.append(("mechanism/" + name, state.mechanisms[name]))
            for name in sorted(state.deformable_positions):
                nodes = state.deformable_positions[name]
                # Fixed-size privileged geometry summary: both ends, centroid, spread and four landmarks.
                indices = np.linspace(0, len(nodes) - 1, 4).astype(int)
                parts.append(
                    ("deformable/" + name, np.concatenate((nodes[indices].ravel(), nodes.mean(0), nodes.std(0))))
                )
            parts.append(("contact/count", len(state.contacts)))
            for name, angles in sorted(state.crease_rest_angles.items()):
                parts.append(("crease/" + name, [float(np.mean(angles)), float(np.max(np.abs(angles)))]))
            privileged_rows.append(parts)
        critic, layouts = pack_privileged_rows(
            privileged_rows, capacity=256, device=self.device, dtype=command_host.dtype
        )
        command = command_host.to(device=self.device)
        self.cfg.hiwonder_metadata["critic_layout"] = layouts
        return {"policy_image": policy, "proprio": proprio, "command": command, "critic": critic}

    def _reset_idx(self, env_ids):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device)
        )
        ended = self.completed[ids]
        started = ids[self.has_started[ids] & ended]
        if len(started):
            outcomes = self.success[started].clone()
            if self.cfg.hiwonder_split == "train":
                self.sampler.update(self.config_ids[started], outcomes)
            self.extras["completed_env_ids"] = started.clone()
            self.extras["completed_successes"] = outcomes
            families = self.reset_family_ids[started].clone()
            self.extras["completed_family_ids"] = families
            self.extras["completed_config_ids"] = self.config_ids[started].clone()
            self.extras.setdefault("log", {})["Metrics/success_rate"] = outcomes.float().mean()
            for name, values in self.episode_reward_components.items():
                self.extras["log"]["Reward/" + name] = values[started].mean()
            for family_id, family in enumerate(self.reset_family_names):
                selected = families == family_id
                if selected.any():
                    self.extras["log"][f"Metrics/success_rate/{family}"] = outcomes[selected].float().mean()
        p = self.sampler.probabilities()
        actual = self.sampling_probabilities()
        self.extras.setdefault("log", {}).update(
            {
                "SGS/sampling_entropy": -(actual * actual.log()).sum(),
                "SGS/global_score_entropy": -(p * p.log()).sum(),
                "SGS/covered_fraction": (self.sampler.counts > 0).float().mean(),
            }
        )
        super()._reset_idx(ids)
        probabilities = self.sampler.probabilities()
        for i in ids.tolist():
            # Every fixed geometry cohort has its own immutable bank slice. IDs cannot
            # silently acquire a different geometry/goal when assigned to another room.
            compatible = torch.arange(i, self.sampler.num_configs, self.num_envs, device=self.device)
            weights = (
                probabilities[compatible]
                if self.cfg.hiwonder_sampling == "sgs"
                else torch.ones_like(compatible, dtype=torch.float32)
            )
            selected = torch.multinomial(weights, 1, generator=self.sampler.generator)
            self.config_ids[i] = compatible[selected[0]]
        self.randomization.reset(ids.cpu(), groups=["actuator", "base", "image", "smudge"], max_attempts=32)
        witnesses = reject_unreachable_bases(self.randomization, self.definitions, ids.cpu().numpy())
        self.cfg.hiwonder_metadata["position_ik_witnesses"].update(
            {i: value.tolist() for i, value in witnesses.items()}
        )
        # Family identity, coherent cohort and deterministic seed jointly define each bank entry.
        reset_definitions = list(self.definitions)
        records = []
        for i in ids.tolist():
            record = self._reset_record(i)
            self.cfg.hiwonder_metadata["position_ik_witnesses"][i] = record.ik_witnesses.tolist()
            reset_definitions[i] = record.definition
            self.episodes[i] = TaskEpisode(record.definition)
            self.reset_family_ids[i] = self.reset_family_names.index(record.family)
            records.append(record)
        self.runtime.reset(ids, reset_definitions)
        q = torch.as_tensor(np.stack([record.robot_q for record in records]), device=self.device, dtype=torch.float32)
        target_q = torch.as_tensor(
            np.stack([record.robot_target_q for record in records]), device=self.device, dtype=torch.float32
        )
        self.targets[ids[:, None], torch.tensor(self.joint_order, device=self.device)] = target_q
        allq = self.targets[ids].clone()
        allq[:, self.joint_order] = q
        self.robot.write_joint_position_to_sim_index(position=allq, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(allq), env_ids=ids)
        base = torch.as_tensor(
            self.cfg.hiwonder_metadata["robot_initial_pose"], device=self.device, dtype=torch.float32
        )[ids].clone()
        sample_base = self.randomization.snapshot()["base"]
        initial_sample = self.cfg.hiwonder_metadata.get("initial_base_positions")
        if initial_sample is not None:
            base[:, :3] += (
                sample_base["position"][ids.cpu()].to(self.device)
                - torch.as_tensor(initial_sample, device=self.device)[ids]
            )
        yaw_q = sample_base["quaternion_xyzw"][ids.cpu()].to(self.device)
        yaw = 2 * torch.atan2(yaw_q[:, 2], yaw_q[:, 3]) + np.pi / 2
        base[:, 3:5] = 0
        base[:, 5], base[:, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        base[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=base, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=ids
        )
        self._write_actuators(ids)
        self.channel.reset(ids, target_q)
        self.actions[ids] = 0
        self.reward_shaper.reset(ids.cpu().numpy())
        for values in self.episode_reward_components.values():
            values[ids] = 0
        self.camera_channel.reset(ids)
        if self.frame_history is not None:
            self.frame_history.reset(ids)
        self.has_started[ids] = True
        self.completed[ids] = False
        self.success[ids] = False
        self.result_step = -1
        # Keep retained environments' realized pixels unchanged on an asynchronous reset.
        parameters = self.image_randomizer.sample_from_snapshot(
            self.randomization.snapshot(ids.cpu()),
            self.profile.camera.height,
            self.profile.camera.width,
            device=self.device,
        )
        for key in vars(self.image_parameters):
            old, new = getattr(self.image_parameters, key), getattr(parameters, key)
            if isinstance(old, torch.Tensor) and old.shape[0] == self.num_envs:
                old[ids] = new
        if self.native_binding is None:
            # Publish after DirectRLEnv's forward/render, before its first observation.
            pending = getattr(self, "_reset_camera_env_ids", set())
            self._reset_camera_env_ids = pending | set(ids.cpu().tolist())

    def sampling_probabilities(self):
        """Return actual equal-cohort reset probabilities, distinct from global beta scores."""
        weights = (
            self.sampler.probabilities()
            if self.cfg.hiwonder_sampling == "sgs"
            else torch.ones(self.sampler.num_configs, device=self.device, dtype=torch.float64)
        )
        return cohort_probabilities(weights, self.num_envs)

    def _reset_record(self, env_id):
        """Qualify a fixed bank entry while resampling only its conditional base nuisance."""
        for _ in range(32):
            sample = self.randomization.snapshot([env_id])
            local_base = sample["base"]["position"][0].numpy().copy()
            local_base[2] = 0
            quaternion = sample["base"]["quaternion_xyzw"][0].numpy()
            try:
                record = make_reset(
                    self.definitions[env_id],
                    int(self.config_ids[env_id]),
                    self.profile.seed,
                    families=self.profile.reset_families,
                    grasp_library=self.cfg.hiwonder_grasp_library,
                    cohort_count=self.num_envs,
                    base_position=local_base,
                    base_yaw=2 * np.arctan2(quaternion[2], quaternion[3]) + np.pi / 2,
                    geometry_scale=float(sample["robot"]["geometry_scale"][0]),
                )
                overlaps = physical_nuisance_overlaps(
                    record.definition, sample, self.profile.randomization, include_reset_envelopes=False
                )
                if not overlaps:
                    return record
                if record.family != "stable_grasp":
                    raise ValueError(
                        f"Immutable configuration {int(self.config_ids[env_id])} intersects physical"
                        f" nuisance {sorted(overlaps)} despite construction qualification"
                    )
                # Stable-grasp object poses follow the sampled base. Preserve
                # the bank entry and nuisance geometry; only condition its base.
                raise UnreachableReset("Stable-grasp placement intersects physical nuisance")
            except UnreachableReset:
                control = self.randomization.config.controls.base
                if not control.enabled or control.frozen or control.cadence == "fixed":
                    raise ValueError(f"Fixed base cannot reach configuration {int(self.config_ids[env_id])}") from None
                self.randomization.reset([env_id], groups=["base"], max_attempts=32)
        raise ValueError(f"No reachable base for configuration {int(self.config_ids[env_id])} after32attempts")

    def _write_actuators(self, env_ids=None):
        """Apply the authoritative nuisance record through actual native property writers."""
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        actuator = self.randomization.snapshot(ids.cpu())["actuator"]
        for field, method, keyword in (
            ("stiffness", "write_joint_stiffness_to_sim_index", "stiffness"),
            ("damping", "write_joint_damping_to_sim_index", "damping"),
            ("effort_limit", "write_joint_effort_limit_to_sim_index", "limits"),
            ("velocity_limit", "write_joint_velocity_limit_to_sim_index", "limits"),
            ("armature", "write_joint_armature_to_sim_index", "armature"),
            ("friction", "write_joint_friction_coefficient_to_sim_index", "joint_friction_coeff"),
        ):
            getattr(self.robot, method)(
                **{keyword: actuator[field].to(self.device)}, joint_ids=self.joint_order, env_ids=ids
            )
        if self.native_binding is not None:
            from ..tasks.vbd_contact import NewtonHiWonderVBDManager

            values = np.stack(
                [
                    actuator[name].numpy()
                    for name in ("stiffness", "damping", "effort_limit", "velocity_limit", "friction", "armature")
                ],
                axis=-1,
            ).reshape(-1, 6)
            rows = (ids.cpu().numpy()[:, None] * 6 + np.arange(6)).reshape(-1)
            motors = NewtonHiWonderVBDManager.get_robot_motors()
            if motors is None:
                raise RuntimeError("Native HiWonder motor mapping was not initialized")
            motors.set_parameters(values, rows)

    def state_dict(self):
        state = {
            "bank_definition": copy.deepcopy(self.bank_definition),
            "bank_hash": self.bank_hash,
            "sampling": self.cfg.hiwonder_sampling,
            "split": self.cfg.hiwonder_split,
            "camera_channel": self.camera_channel.state_dict(),
            "sampler": self.sampler.state_dict(),
            "randomization": self.randomization.state_dict(),
            "config_ids": self.config_ids.clone(),
            "channel": self.channel.state_dict(),
            "observation_rng": self.observation_generator.get_state(),
            "image_parameters": self.image_parameters.state_dict(),
            "joint_positions": self.robot.data.joint_pos.torch.clone(),
            "joint_velocities": self.robot.data.joint_vel.torch.clone(),
            "root_pose": self.robot.data.root_pose_w.torch.clone(),
            "episode_length_buf": self.episode_length_buf.clone(),
            "root_velocity": self.robot.data.root_vel_w.torch.clone(),
            "episodes": copy.deepcopy(self.episodes),
            "runtime": self.runtime.state_dict(),
            "obs_buf": copy.deepcopy(self.obs_buf),
            "common_step_counter": self.common_step_counter,
            "frame_history": self.frame_history.state_dict() if self.frame_history is not None else None,
            "reward_shaper": self.reward_shaper.state_dict(),
            "episode_reward_components": {
                name: value.clone() for name, value in self.episode_reward_components.items()
            },
            "lighting": self.lighting.state_dict(),
        }
        for name in (
            "actions",
            "targets",
            "has_started",
            "completed",
            "success",
            "failures",
            "timeouts",
            "rewards",
            "reset_family_ids",
        ):
            state[name] = getattr(self, name).clone()
        return state

    def load_state_dict(self, state):
        from isaaclab_hiwonder.images import ImageParameters

        if state.get("sampling", self.cfg.hiwonder_sampling) != self.cfg.hiwonder_sampling:
            raise ValueError("Checkpoint sampling policy differs from constructed environment")
        if state.get("split", self.cfg.hiwonder_split) != self.cfg.hiwonder_split:
            raise ValueError("Checkpoint split differs from constructed environment")
        if state["bank_hash"] != self.bank_hash:
            raise ValueError("Checkpoint fixed SGS bank/profile differs from the constructed environment")
        self.sampler.load_state_dict(state["sampler"])
        self.randomization.load_state_dict(state["randomization"])
        self.reward_shaper.load_state_dict(state["reward_shaper"])
        self.episode_reward_components = {
            name: value.to(self.device).clone() for name, value in state["episode_reward_components"].items()
        }
        self._write_actuators()
        self.config_ids.copy_(state["config_ids"])
        self.channel.load_state_dict(state["channel"])
        self.camera_channel.load_state_dict(state["camera_channel"])
        if self.frame_history is not None:
            self.frame_history.load_state_dict(state["frame_history"])
        self.observation_generator.set_state(state["observation_rng"].cpu())
        self.image_parameters = ImageParameters.from_state_dict(state["image_parameters"], device=self.device)
        self.robot.write_joint_position_to_sim_index(position=state["joint_positions"].to(self.device))
        self.robot.write_joint_velocity_to_sim_index(velocity=state["joint_velocities"].to(self.device))
        self.robot.write_root_pose_to_sim_index(root_pose=state["root_pose"].to(self.device))
        self.robot.write_root_velocity_to_sim_index(root_velocity=state["root_velocity"].to(self.device))
        self.episode_length_buf.copy_(state["episode_length_buf"])
        self.episodes = copy.deepcopy(state["episodes"])
        self.runtime.load_state_dict(state["runtime"])
        self.obs_buf = copy.deepcopy(state["obs_buf"])
        self.common_step_counter = state["common_step_counter"]
        for name in (
            "actions",
            "targets",
            "has_started",
            "completed",
            "success",
            "failures",
            "timeouts",
            "rewards",
            "reset_family_ids",
        ):
            getattr(self, name).copy_(state[name])
        self.result_step = -1
        self.lighting.load_state_dict(state["lighting"])
