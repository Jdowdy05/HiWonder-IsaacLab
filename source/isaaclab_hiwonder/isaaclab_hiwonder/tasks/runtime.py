# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured SDK task state, indexed resets and physical mechanism effort writes."""

import copy
import re

import numpy as np
import torch

from ..assets.robot_models import get_robot_model
from ..sim.contact_filtering import body_contact_aliases
from .api import ActorState, SceneState, rotation_matrix
from .backend import TaskMechanics


def _numpy(value):
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "numpy"):
        return value.numpy().copy()
    return np.asarray(value).copy()


def _tensor(value):
    """View SDK, Torch or native Warp arrays before selecting host-bound data."""
    if hasattr(value, "warp"):
        value = value.warp
    if hasattr(value, "torch"):
        return value.torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device="cpu")
    # Warp's DLPack handoff orders the current Torch consumer after the current
    # Warp producer. CUDA-array-interface views alone do not carry that fence.
    return torch.from_dlpack(value)


def _array_view(value):
    """Keep NumPy layouts/adapters usable while viewing native device arrays safely."""
    if isinstance(value, np.ndarray):
        return value
    if (
        isinstance(value, torch.Tensor)
        or hasattr(value, "warp")
        or hasattr(value, "torch")
        or hasattr(value, "__dlpack__")
    ):
        return _tensor(value)
    return _numpy(value)


def _packed_numpy(*values):
    """Return owned host components without changing their dtype or arithmetic.

    Compatible floating tensors share one transfer. Mixed dtypes/devices and
    NumPy-only representations retain the original independent copy path.
    """
    arrays = tuple(_array_view(value) for value in values)
    first = arrays[0]
    if not isinstance(first, torch.Tensor) or first.dtype not in (torch.float16, torch.float32, torch.float64):
        return tuple(_numpy(value) for value in arrays)
    if not all(
        isinstance(value, torch.Tensor)
        and value.dtype == first.dtype
        and value.device == first.device
        and value.layout == torch.strided
        and value.shape[:-1] == first.shape[:-1]
        for value in arrays
    ):
        return tuple(_numpy(value) for value in arrays)
    packed = _numpy(torch.cat(arrays, dim=-1))
    components, offset = [], 0
    for value in arrays:
        end = offset + value.shape[-1]
        components.append(packed[..., offset:end])
        offset = end
    return tuple(components)


def cable_nodes_from_segments(poses, velocities, lengths):
    """Reconstruct measured rod centerline from native COM-origin segment poses.

    Internal junction nodes average the two measured incident endpoints. Angular
    velocity contributes to endpoint velocity; COM velocity alone is insufficient.
    """
    rotations = np.array([rotation_matrix(q) for q in poses[:, 3:]])
    offsets = rotations[:, :, 2] * (np.asarray(lengths)[:, None] / 2)
    starts, ends = poses[:, :3] - offsets, poses[:, :3] + offsets
    spin = np.cross(velocities[:, 3:], offsets)
    vstarts, vends = velocities[:, :3] - spin, velocities[:, :3] + spin
    nodes = np.concatenate([starts[:1], (ends[:-1] + starts[1:]) / 2, ends[-1:]])
    speeds = np.concatenate([vstarts[:1], (vends[:-1] + vstarts[1:]) / 2, vends[-1:]])
    return nodes, speeds


def link_state_from_com(pose, velocity, local_com):
    """Combine live base arrays, converting COM linear velocity to the actor origin."""
    pose = np.asarray(pose)
    speed = np.asarray(velocity).copy()
    rotations = np.stack([rotation_matrix(q) for q in pose[..., 3:7].reshape(-1, 4)]).reshape(*pose.shape[:-1], 3, 3)
    offset = np.einsum("...ij,...j->...i", rotations, local_com)
    speed[..., :3] -= np.cross(speed[..., 3:], offset)
    return np.concatenate([pose, speed], axis=-1)


class TaskRuntime:
    """Use scene-worker metadata to map task names onto actual initialized assets.

    NativeTaskBinding must be constructed by the environment before sim.reset and
    supplied as ``native_binding`` (or metadata['native_binding']). This class is
    constructed after SDK asset initialization. It does not invent contacts.
    """

    def __init__(self, scene, metadata, definitions, native_binding=None):
        self.scene = scene
        self.metadata = metadata
        self.robot_model = get_robot_model(metadata.get("robot_model", "stock"))
        self.definitions = list(definitions)
        self.count = len(definitions)
        self.device = scene.device
        self.origins = _numpy(scene.env_origins)
        self.offsets = self.origins + np.asarray(metadata["task_support_offsets_m"], dtype=float)
        self.native_binding = native_binding or metadata.get("native_binding")
        self.mechanics = [TaskMechanics(d) for d in definitions]
        self.articulation_maps = {}
        for root, entry in metadata["task_articulations"].items():
            asset = scene[entry["asset_key"]]
            body_names = list(asset.body_names)
            joint_names = list(asset.joint_names)
            body_ids = {name: body_names.index(name) for name in entry["body_names"]}
            joint_ids = {name: joint_names.index(name) for name in entry["joint_names"]}
            self.articulation_maps[root] = (asset, body_ids, joint_ids)
        self.default_cable_poses = {}
        for name, key in metadata["task_deformables"].items():
            if any(spec.name == name and spec.kind == "cable" for spec in definitions[0].deformables):
                self.default_cable_poses[name] = _numpy(scene[key].data.default_segment_pose_w)
        self.time = np.zeros(self.count)
        self._wrench_assets = set()
        self._native_model = None
        self._native_root_indices = {}
        self._native_root_batches = {}
        self._contact_topology = None
        self._definition_cache = None
        self._cable_lengths = {}

    def _invalidate_invariants(self):
        """Reset-time definition changes invalidate derived static data."""
        self._contact_topology = None
        self._definition_cache = None
        self._cable_lengths.clear()

    def _use_model(self, model):
        if model is not self._native_model:
            self._native_model = model
            self._native_root_indices.clear()
            self._native_root_batches.clear()
            self._invalidate_invariants()

    def _definitions_changed(self):
        definitions = tuple(id(definition) for definition in self.definitions)
        if definitions != self._definition_cache:
            self._cable_lengths.clear()
            self._contact_topology = None
            self._definition_cache = definitions

    def _root_state(self, asset):
        return self._root_states([asset])[id(asset)]

    def _root_states(self, assets):
        """Gather only requested native roots, once for each live model snapshot."""
        result = {}
        native = [asset for asset in assets if getattr(asset.data, "__backend_name__", None) == "newton"]
        if native:
            from isaaclab_newton.physics import NewtonManager

            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            self._use_model(model)
            paths = tuple(asset.cfg.prim_path for asset in native)
            for path in paths:
                if path not in self._native_root_indices:
                    pattern = re.compile(path.replace("{ENV_REGEX_NS}", r"/World/envs/env_[^/]+"))
                    matches = {}
                    for index, label in enumerate(model.body_label):
                        if pattern.fullmatch(label):
                            environment = re.search(r"/env_(\d+)/", label)
                            if environment is None:
                                raise RuntimeError(f"Task body has no environment index: {label}")
                            matches[int(environment.group(1))] = index
                    if set(matches) != set(range(self.count)):
                        raise RuntimeError(f"Native body mapping incomplete for {path}: {matches}")
                    self._native_root_indices[path] = np.array([matches[i] for i in range(self.count)])
            pose, velocity, local_com = (_array_view(value) for value in (state.body_q, state.body_qd, model.body_com))
            if paths not in self._native_root_batches:
                self._native_root_batches[paths] = torch.as_tensor(
                    np.concatenate([self._native_root_indices[path] for path in paths]),
                    device=pose.device if isinstance(pose, torch.Tensor) else "cpu",
                    dtype=torch.long,
                )
            indices = self._native_root_batches[paths]
            selected = []
            for value in (pose, velocity, local_com):
                if isinstance(value, torch.Tensor):
                    selected.append(value.index_select(0, indices.to(device=value.device)))
                else:
                    selected.append(value[_numpy(indices)])
            poses, velocities, centers = _packed_numpy(*selected)
            for index, asset in enumerate(native):
                rows = slice(index * self.count, (index + 1) * self.count)
                result[id(asset)] = link_state_from_com(poses[rows], velocities[rows], centers[rows])
        for asset in assets:
            if id(asset) not in result:
                pose, velocity, local_com = _packed_numpy(
                    asset.data.root_link_pose_w,
                    asset.data.root_com_vel_w,
                    _array_view(asset.data.body_com_pos_b)[:, 0],
                )
                result[id(asset)] = link_state_from_com(pose, velocity, local_com)
        return result

    def states(self):
        """Return owned, fresh measurements and complete task contact evidence."""
        return self._collect_states()

    def _collect_states(self, required=None):
        self._definitions_changed()
        states = [SceneState(actors={}, time=float(self.time[i])) for i in range(self.count)]

        def needed(category, name):
            return required is None or name in required[category]

        def fill(label, values):
            for i, row in enumerate(values):
                states[i].actors[label] = ActorState(row[:3] - self.offsets[i], row[3:7], row[7:10], row[10:13])

        roots = {
            name: self.scene[key] for name, key in self.metadata["task_body_assets"].items() if needed("actors", name)
        }
        root_states = self._root_states(list(roots.values()))
        for name, asset in roots.items():
            fill(name, root_states[id(asset)])
        for _, (asset, body_ids, joint_ids) in self.articulation_maps.items():
            selected_bodies = {name: index for name, index in body_ids.items() if needed("actors", name)}
            if selected_bodies:
                indices = list(selected_bodies.values())
                poses, velocities, centers = _packed_numpy(
                    _array_view(asset.data.body_link_pose_w)[:, indices],
                    _array_view(asset.data.body_com_vel_w)[:, indices],
                    _array_view(asset.data.body_com_pos_b)[:, indices],
                )
                bodies = link_state_from_com(poses, velocities, centers)
                for index, name in enumerate(selected_bodies):
                    fill(name, bodies[:, index])
            selected_joints = {name: index for name, index in joint_ids.items() if needed("joints", name)}
            if not selected_joints:
                continue
            indices = list(selected_joints.values())
            q, qd = _packed_numpy(
                _array_view(asset.data.joint_pos)[:, indices], _array_view(asset.data.joint_vel)[:, indices]
            )
            for index, name in enumerate(selected_joints):
                for i, state in enumerate(states):
                    state.joint_positions[name] = float(q[i, index])
                    state.joint_velocities[name] = float(qd[i, index])
        for name, key in self.metadata["task_deformables"].items():
            if not needed("deformables", name):
                continue
            asset = self.scene[key]
            spec = next(s for s in self.definitions[0].deformables if s.name == name)
            if spec.kind == "cable":
                poses = _numpy(asset.data.segment_pose_w)
                velocity = _numpy(asset.data.segment_velocity_w)
                for i, state in enumerate(states):
                    own = next(s for s in self.definitions[i].deformables if s.name == name)
                    vertices = np.asarray(own.vertices)
                    cached = self._cable_lengths.get((i, name))
                    if cached is None or vertices.dtype != cached[0].dtype or not np.array_equal(vertices, cached[0]):
                        cached = (vertices.copy(), np.linalg.norm(np.diff(vertices, axis=0), axis=1))
                        self._cable_lengths[i, name] = cached
                    lengths = cached[1]
                    nodes, speeds = cable_nodes_from_segments(poses[i], velocity[i], lengths)
                    state.deformable_positions[name] = nodes - self.offsets[i]
                    state.deformable_velocities[name] = speeds
                    state.actors[f"{name}/segment_0"] = ActorState(
                        poses[i, 0, :3] - self.offsets[i], poses[i, 0, 3:], velocity[i, 0, :3], velocity[i, 0, 3:]
                    )
            else:
                positions = _numpy(asset.data.nodal_pos_w)
                velocity = _numpy(asset.data.nodal_vel_w)
                for i, state in enumerate(states):
                    state.deformable_positions[name] = positions[i] - self.offsets[i]
                    state.deformable_velocities[name] = velocity[i]
                    if spec.kind == "plastic_shell":
                        if self.native_binding is None or self.native_binding.hook is None:
                            raise RuntimeError("Paper outcome requires initialized native plastic-state binding")
                        state.crease_rest_angles[name] = self.native_binding.hook.crease_angles(i, name)
        from ..sim.contact_filtering import actor_contact_presence

        for actor, entry in self.metadata["task_contacts"].items():
            actor_aliases = entry.get("actor_aliases", (actor,))
            filter_aliases = entry.get("filter_aliases", {})
            if required is not None and not any(
                frozenset((source, target)) in required["contacts"]
                for source in actor_aliases
                for label in entry["filter_labels"]
                for target in filter_aliases.get(label, (label,))
            ):
                continue
            sensor = self.scene[entry["sensor_key"]]
            value = sensor.data.force_matrix_w
            if value is None:
                raise RuntimeError(f"Filtered contact reporting unavailable for task actor {actor}")
            matrix = _numpy(value)
            touching = actor_contact_presence(matrix, entry["filter_labels"])
            for i, state in enumerate(states):
                for label, contact in touching.items():
                    if contact[i]:
                        state.contacts.update(
                            (source, target)
                            for source in actor_aliases
                            for target in filter_aliases.get(label, (label,))
                        )
        if self.definitions[0].deformables and required is None:
            self._native_contacts(states)
        for mechanics, state in zip(self.mechanics, states):
            if mechanics.cap is not None:
                mechanics.cap.observe(state)
        return states

    def _native_contact_mapping(self, model, entries, ranges):
        """Resolve immutable shape aliases and particle spans for this topology."""
        topology = (
            tuple(
                (
                    env_id,
                    tuple(definition.initial),
                    tuple((spec.name, spec.kind) for spec in definition.deformables),
                    task_path,
                    tuple(paths.items()),
                )
                for env_id, definition, task_path, paths in entries
            ),
            tuple((path, value[0], value[1]) for path, value in ranges.items()),
            tuple(model.body_label),
            tuple(model.shape_label),
            id(model.shape_body),
        )
        if self._contact_topology is None or self._contact_topology[0] != topology:
            shape_body = _numpy(model.shape_body)
            body_labels = list(model.body_label)
            shape_labels = list(model.shape_label)
            body_alias = {}
            particle_ranges = []
            for env_id, definition, task_path, paths in entries:
                room = task_path.rsplit("/", 1)[0]
                for index, label in enumerate(body_labels):
                    if label.startswith(task_path + "/"):
                        tail = label[len(task_path) + 1 :]
                        if tail in definition.initial:
                            body_alias[index] = ((env_id, tail),)
                    elif label.startswith(room + "/Robot/"):
                        body = label[len(room + "/Robot/") :]
                        body_alias[index] = tuple(
                            (env_id, alias) for alias in body_contact_aliases(self.robot_model, body)
                        )
                    elif label == room + "/Tabletop":
                        body_alias[index] = ((env_id, "table"),)
                    for spec in definition.deformables:
                        if spec.kind == "cable" and label.startswith(paths[spec.name]):
                            body_alias[index] = ((env_id, spec.name),)
                for spec in definition.deformables:
                    if spec.kind != "cable":
                        for path, (offset, count, _) in ranges.items():
                            if path == paths[spec.name] or path.startswith(paths[spec.name] + "/"):
                                particle_ranges.append((offset, offset + count, env_id, spec.name))
            aliases = []
            for body, label in zip(shape_body, shape_labels):
                alias = body_alias.get(int(body), ())
                if not alias:
                    for env_id, _, task_path, _ in entries:
                        if label.startswith(task_path.rsplit("/", 1)[0] + "/Tabletop"):
                            alias = ((env_id, "table"),)
                            break
                aliases.append(alias)
            self._contact_topology = (topology, aliases, particle_ranges)
        _, aliases, particle_ranges = self._contact_topology
        return aliases, particle_ranges

    def _native_contacts(self, states):
        """Read native rigid and full-surface contacts, never bounding-box guesses.

        Surface contact candidates conservatively block release until separated.
        They are not used to establish insertion or support success. Rod contacts
        use solved rigid-contact forces, not mere broad-phase overlap.
        """
        from isaaclab_newton.physics import NewtonManager

        manager = NewtonManager
        model = manager.get_model()
        contacts = manager.get_contacts()
        if model is None or contacts is None:
            raise RuntimeError("Native deformable tasks require readable Newton contact buffers")
        self._use_model(model)
        aliases, particle_ranges = self._native_contact_mapping(
            model, self.metadata["native_task_entries"], manager.backend.deformable_ranges
        )

        def shape_alias(index):
            return aliases[index]

        count = int(_numpy(contacts.rigid_contact_count)[0])
        if count:
            a = _numpy(contacts.rigid_contact_shape0)[:count]
            b = _numpy(contacts.rigid_contact_shape1)[:count]
            forces = _numpy(contacts.rigid_contact_force)[:count]
            for first, second, force in zip(a, b, forces):
                if first < 0 or second < 0 or np.linalg.norm(force) < 1e-5:
                    continue
                for left in shape_alias(first):
                    for right in shape_alias(second):
                        if left[0] == right[0] and left[1] != right[1]:
                            states[left[0]].contacts.add((left[1], right[1]))
        count = int(_numpy(contacts.soft_contact_count)[0])
        if count:
            indices = _numpy(contacts.soft_contact_indices)[:count]
            shapes = _numpy(contacts.soft_contact_shape)[:count]
            for ids, shape in zip(indices, shapes):
                if shape < 0:
                    continue
                for rigid in shape_alias(shape):
                    for lo, hi, env_id, name in particle_ranges:
                        if env_id == rigid[0] and np.any((ids >= lo) & (ids < hi)):
                            states[env_id].contacts.add((name, rigid[1]))

    def reset(self, env_ids, definitions=None):
        ids = np.asarray(_numpy(env_ids), dtype=int).reshape(-1)
        if definitions is not None:
            if len(definitions) == self.count:
                definitions = [definitions[i] for i in ids]
            if len(definitions) != len(ids):
                raise ValueError("Reset definitions must correspond exactly to selected environment IDs")
            for index, definition in zip(ids, definitions):
                current = self.definitions[index]
                if (definition.name, definition.variant) != (current.name, current.variant):
                    raise ValueError("Changing task topology requires rebuilding the scene")
                for before, after in zip(current.deformables, definition.deformables):
                    same_mesh = before.vertices.shape == after.vertices.shape and np.array_equal(
                        before.edges, after.edges
                    )
                    if not same_mesh:
                        raise ValueError("Deformable topology change requires rebuilding the scene")
                    a = np.linalg.norm(
                        before.vertices[before.edges[:, 0]] - before.vertices[before.edges[:, 1]], axis=1
                    )
                    b = np.linalg.norm(after.vertices[after.edges[:, 0]] - after.vertices[after.edges[:, 1]], axis=1)
                    if not np.allclose(a, b, rtol=1e-5, atol=1e-7):
                        raise ValueError("Deformable rest geometry change requires rebuilding the scene")
                self.definitions[index] = definition
                self.mechanics[index] = TaskMechanics(definition)
        selected = torch.as_tensor(ids, device=self.device, dtype=torch.long)

        def tensor(value):
            return torch.as_tensor(np.asarray(value), device=self.device, dtype=torch.float32)

        for name, key in self.metadata["task_body_assets"].items():
            values = [self.definitions[i].initial[name] for i in ids]
            poses = [np.r_[s.position + self.offsets[i], s.orientation] for i, s in zip(ids, values)]
            velocities = [np.r_[s.linear_velocity, s.angular_velocity] for s in values]
            asset = self.scene[key]
            asset.write_root_pose_to_sim_index(root_pose=tensor(poses), env_ids=selected)
            asset.write_root_link_velocity_to_sim_index(root_velocity=tensor(velocities), env_ids=selected)
        for category in ("distractor", "confetti"):
            assets = self.metadata.get(f"{category}_assets", {})
            if assets:
                initial = np.asarray(self.metadata[f"{category}_initial_poses"], dtype=float)
                for index, key in assets.items():
                    poses = initial[ids, int(index)].copy()
                    poses[:, :3] += self.origins[ids]
                    asset = self.scene[key]
                    asset.write_root_pose_to_sim_index(root_pose=tensor(poses), env_ids=selected)
                    asset.write_root_link_velocity_to_sim_index(
                        root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=selected
                    )
        for root, (asset, body_ids, joint_ids) in self.articulation_maps.items():
            values = [self.definitions[i].initial[root] for i in ids]
            poses = [np.r_[s.position + self.offsets[i], s.orientation] for i, s in zip(ids, values)]
            asset.write_root_pose_to_sim_index(root_pose=tensor(poses), env_ids=selected)
            asset.write_root_link_velocity_to_sim_index(
                root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=selected
            )
            joint_positions = [
                [self.definitions[i].parameters.get("initial_joints", {}).get(name, 0) for name in joint_ids]
                for i in ids
            ]
            asset.write_joint_position_to_sim_index(
                position=tensor(joint_positions), joint_ids=list(joint_ids.values()), env_ids=selected
            )
            asset.write_joint_velocity_to_sim_index(
                velocity=torch.zeros((len(ids), len(joint_ids)), device=self.device),
                joint_ids=list(joint_ids.values()),
                env_ids=selected,
            )
        for name, key in self.metadata["task_deformables"].items():
            asset = self.scene[key]
            specifications = [next(d for d in self.definitions[i].deformables if d.name == name) for i in ids]
            if specifications[0].kind == "cable":
                poses = []
                for i, spec in zip(ids, specifications):
                    value = self.default_cable_poses[name][i].copy()
                    value[:, :3] = (spec.vertices[:-1] + spec.vertices[1:]) / 2 + self.offsets[i]
                    for j, direction in enumerate(np.diff(spec.vertices, axis=0)):
                        target = direction / np.linalg.norm(direction)
                        source = rotation_matrix(value[j, 3:])[:, 2]
                        delta = np.r_[np.cross(source, target), 1 + np.dot(source, target)]
                        if np.linalg.norm(delta) < 1e-8:
                            axis = np.cross(source, [1, 0, 0] if abs(source[0]) < 0.9 else [0, 1, 0])
                            delta = np.r_[axis / np.linalg.norm(axis), 0]
                        delta /= np.linalg.norm(delta)
                        a, b = delta, value[j, 3:].copy()
                        value[j, 3:6] = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
                        value[j, 6] = a[3] * b[3] - np.dot(a[:3], b[:3])
                    poses.append(value)
                asset.write_segment_pose_to_sim_index(segment_pose=tensor(poses), env_ids=selected)
                asset.write_segment_velocity_to_sim_index(
                    segment_velocity=torch.zeros((len(ids), len(poses[0]), 6), device=self.device), env_ids=selected
                )
            else:
                vertices = np.stack([spec.vertices + self.offsets[i] for i, spec in zip(ids, specifications)])
                asset.write_nodal_state_to_sim_index(
                    tensor(np.concatenate([vertices, np.zeros_like(vertices)], axis=-1)), env_ids=selected
                )
        if self.native_binding is not None:
            mask = np.zeros(self.count, dtype=bool)
            mask[ids] = True
            self.native_binding.reset(mask)
        self.time[ids] = 0
        for i in ids:
            self.mechanics[i] = TaskMechanics(self.definitions[i])
        self._invalidate_invariants()

    def _mechanics_inputs(self):
        """Dependencies of the current rigid TaskMechanics equations.

        Unrecognized mechanics and deformable combinations retain the complete
        measurement path rather than assuming that their force laws need less.
        """
        from .catalog import TASK_NAMES

        required = {"actors": set(), "joints": set(), "deformables": set(), "contacts": set()}
        wrench_bodies = set()
        for mechanics in self.mechanics:
            definition = mechanics.definition
            if type(mechanics) is not TaskMechanics or definition.name not in TASK_NAMES or definition.deformables:
                return None
            required["joints"].update(joint.name for joint in definition.joints if "detent_states" in joint.parameters)
            if definition.name == "bottle_cap":
                required["actors"].update(("cap", "bottle"))
                wrench_bodies.update(("cap", "bottle"))
                required["contacts"].update((frozenset(("cap", "left_finger")), frozenset(("cap", "right_finger"))))
            elif definition.name == "key_lock":
                required["actors"].add("key")
                required["joints"].update(("lock_angle", "bolt_travel"))
                required["contacts"].add(frozenset(("key", "cylinder")))
        # Wrench composer positions are required for every body in an active group.
        for root, (_, body_ids, _) in self.articulation_maps.items():
            if wrench_bodies.intersection(body_ids) or root in self._wrench_assets:
                required["actors"].update(body_ids)
        for name, key in self.metadata["task_body_assets"].items():
            if name in wrench_bodies or key in self._wrench_assets:
                required["actors"].add(name)
        return required

    def apply_forces(self, dt=0.0):
        """Write mechanism efforts every physics step; native soft forces run per substep."""
        self.time += dt
        if not self.articulation_maps and not any(m.cap is not None for m in self.mechanics):
            return
        measured = self._collect_states(self._mechanics_inputs())
        forces = [mechanics.forces(state, dt) for mechanics, state in zip(self.mechanics, measured)]
        for _, (asset, _, joint_ids) in self.articulation_maps.items():
            effort = np.array([[f.joint_efforts.get(name, 0.0) for name in joint_ids] for f in forces], np.float32)
            asset.actuators.target_command.set_effort_index(
                value=torch.as_tensor(effort, device=self.device), joint_ids=list(joint_ids.values())
            )
            asset.write_data_to_sim()
        body_groups = [(self.scene[key], {name: 0}, key) for name, key in self.metadata["task_body_assets"].items()]
        body_groups.extend((asset, body_ids, root) for root, (asset, body_ids, _) in self.articulation_maps.items())
        for asset, body_ids, key in body_groups:
            active = any(name in f.body_forces or name in f.body_torques for f in forces for name in body_ids)
            if not active and key not in self._wrench_assets:
                continue
            external = np.asarray(
                [[f.body_forces.get(name, np.zeros(3)) for name in body_ids] for f in forces], np.float32
            )
            torque = np.asarray(
                [[f.body_torques.get(name, np.zeros(3)) for name in body_ids] for f in forces], np.float32
            )
            composer = asset.permanent_wrench_composer
            composer.reset()
            positions = np.asarray(
                [[measured[i].actors[name].position + self.offsets[i] for name in body_ids] for i in range(self.count)],
                np.float32,
            )
            composer.add_forces_and_torques(
                torch.as_tensor(external, device=self.device),
                torch.as_tensor(torque, device=self.device),
                positions=torch.as_tensor(positions, device=self.device),
                body_ids=list(body_ids.values()),
                is_global=True,
            )
            self._wrench_assets.add(key)

    def state_dict(self):
        """Capture measured task state and persistent constitutive state for resume."""
        result = {
            "version": 1,
            "definitions": copy.deepcopy(self.definitions),
            "time": self.time.copy(),
            "bodies": {},
            "articulations": {},
            "deformables": {},
            "plastic": None,
            "mechanisms": [m.cap.state_dict() if m.cap is not None else None for m in self.mechanics],
            "nuisance_bodies": {},
        }
        for name, key in self.metadata["task_body_assets"].items():
            result["bodies"][name] = self._root_state(self.scene[key])
        for category in ("distractor", "confetti"):
            for key in self.metadata.get(f"{category}_assets", {}).values():
                result["nuisance_bodies"][key] = self._root_state(self.scene[key])
        for root, (asset, _, _) in self.articulation_maps.items():
            result["articulations"][root] = {
                "root": self._root_state(asset),
                "q": _numpy(asset.data.joint_pos),
                "qd": _numpy(asset.data.joint_vel),
            }
        for name, key in self.metadata["task_deformables"].items():
            asset = self.scene[key]
            if name in self.default_cable_poses:
                result["deformables"][name] = {
                    "poses": _numpy(asset.data.segment_pose_w),
                    "velocities": _numpy(asset.data.segment_velocity_w),
                }
            else:
                result["deformables"][name] = {
                    "positions": _numpy(asset.data.nodal_pos_w),
                    "velocities": _numpy(asset.data.nodal_vel_w),
                }
        if self.native_binding is not None and self.native_binding.hook is not None:
            hook = self.native_binding.hook
            result["plastic"] = {"rest": hook.rest.numpy().copy(), "accumulated": hook.accumulated.numpy().copy()}
        return result

    def load_state_dict(self, state):
        """Restore physical arrays before the next solver step; no fabricated outcomes."""
        if state.get("version") != 1 or len(state["definitions"]) != self.count:
            raise ValueError("Task checkpoint version or environment count mismatch")
        self.reset(np.arange(self.count), state["definitions"])
        self.definitions = copy.deepcopy(state["definitions"])
        self._invalidate_invariants()

        def tensor(value):
            return torch.as_tensor(value, device=self.device, dtype=torch.float32)

        for name, values in state["bodies"].items():
            asset = self.scene[self.metadata["task_body_assets"][name]]
            asset.write_root_pose_to_sim_index(root_pose=tensor(values[:, :7]))
            asset.write_root_link_velocity_to_sim_index(root_velocity=tensor(values[:, 7:]))
        for key, values in state.get("nuisance_bodies", {}).items():
            asset = self.scene[key]
            asset.write_root_pose_to_sim_index(root_pose=tensor(values[:, :7]))
            asset.write_root_link_velocity_to_sim_index(root_velocity=tensor(values[:, 7:]))
        for root, values in state["articulations"].items():
            asset = self.articulation_maps[root][0]
            asset.write_root_pose_to_sim_index(root_pose=tensor(values["root"][:, :7]))
            asset.write_root_link_velocity_to_sim_index(root_velocity=tensor(values["root"][:, 7:]))
            asset.write_joint_position_to_sim_index(position=tensor(values["q"]))
            asset.write_joint_velocity_to_sim_index(velocity=tensor(values["qd"]))
        for name, values in state["deformables"].items():
            asset = self.scene[self.metadata["task_deformables"][name]]
            if "poses" in values:
                asset.write_segment_pose_to_sim_index(segment_pose=tensor(values["poses"]))
                asset.write_segment_velocity_to_sim_index(segment_velocity=tensor(values["velocities"]))
            else:
                nodal = np.concatenate([values["positions"], values["velocities"]], axis=-1)
                asset.write_nodal_state_to_sim_index(tensor(nodal))
        if state["plastic"] is not None:
            import warp as wp

            if self.native_binding is None or self.native_binding.hook is None:
                raise RuntimeError("Checkpoint has plastic state but native hook is unavailable")
            hook = self.native_binding.hook
            for field in ("rest", "accumulated"):
                value = np.asarray(state["plastic"][field], np.float32)
                target = getattr(hook, field)
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError("Plastic checkpoint topology mismatch")
                wp.copy(target, wp.array(value, dtype=wp.float32, device=hook.device))
        self.time = np.asarray(state["time"], dtype=float).copy()
        for mechanics, value in zip(self.mechanics, state.get("mechanisms", [None] * self.count)):
            if value is not None:
                if mechanics.cap is None:
                    raise ValueError("Cap checkpoint cannot load into a different task")
                mechanics.cap.load_state_dict(value)
