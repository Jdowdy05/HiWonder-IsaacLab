# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete Isaac Lab 3.0 task backend presets and lifecycle binding."""

import numpy as np


def filter_cable_material_overlap(builder, spec, path):
    """Remove capsule end-cap overlap within one contiguous material neighborhood.

    Newton filters directly jointed rods. Finer rods also have overlapping caps
    across intervening segments shorter than the cable diameter. Those contacts
    repel the same material from itself. Use intrinsic rest arclength, never
    spatial proximity, so distant portions retain the self-contact knots need.
    """
    edges = np.asarray(spec.edges)
    expected = np.column_stack((np.arange(len(edges)), np.arange(1, len(edges) + 1)))
    if not np.array_equal(edges, expected):
        raise ValueError("Native cable overlap filtering requires an ordered open chain")
    bodies = []
    for index in range(len(edges)):
        matches = [
            body
            for body, label in enumerate(builder.body_label)
            if label.startswith(path + "/") and label.endswith(f"_edge_body_{index}")
        ]
        if len(matches) != 1:
            raise ValueError(f"Cable segment mapping is not unique: {path}, {index}")
        bodies.append(matches[0])
    shapes = {body: [i for i, owner in enumerate(builder.shape_body) if owner == body] for body in bodies}
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(spec.vertices, axis=0), axis=1))))
    pairs = []
    for first in range(len(edges)):
        for second in range(first + 2, len(edges)):
            if arc[second] - arc[first + 1] >= 2 * spec.radius - 1e-9:
                break
            for a in shapes[bodies[first]]:
                for b in shapes[bodies[second]]:
                    builder.add_shape_collision_filter_pair(a, b)
            pairs.append((first, second))
    return pairs


def physics_config(definition):
    """Return a real SDK physics config, keeping coupled bodies in one solver.

    Monolithic VBD integrates the robot, task rigid bodies, rods and surfaces.
    This preserves reaction forces from the plug/cable and garment contacts.
    It remains a qualification candidate, not a measured hardware model.
    """
    if definition.physics_family in ("rigid", "articulated"):
        from isaaclab_physx.physics import PhysxCfg

        return PhysxCfg(gpu_found_lost_pairs_capacity=2**22)
    from isaaclab_newton.physics import (
        NewtonCfg,
        NewtonCollisionPipelineCfg,
        NewtonShapeCfg,
        NewtonSoftContactCfg,
        VBDSolverCfg,
    )

    radius = min(s.radius for s in definition.deformables)
    has_surface = any(s.kind in ("surface", "plastic_shell") for s in definition.deformables)
    return NewtonCfg(
        solver_cfg=VBDSolverCfg(
            class_type="isaaclab_hiwonder.tasks.vbd_contact:NewtonHiWonderVBDManager",
            iterations=20,
            rigid_compliant_alm=True,
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=radius,
            particle_self_contact_margin=2 * radius,
            particle_collision_detection_interval=1,
            particle_rest_shape_contact_exclusion_radius=0.0,
            rigid_body_contact_buffer_size=2048,
            rigid_body_particle_contact_buffer_size=2048,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            enable_rigid_soft_full_surface_contact=has_surface,
            soft_contact_margin=max(2 * radius, 0.001),
        ),
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=8000, soft_contact_kd=0.05, soft_contact_mu=0.7),
        default_shape_cfg=NewtonShapeCfg(ke=2e4, kd=100, mu=0.6, gap=0.001),
        num_substeps=8,
        collision_decimation=1,
    )


def articulation_groups(definition):
    """Describe each connected mechanism for ArticulationCfg(spawn=None).

    Root path is relative to /Room/Task. Connected child bodies must not also be
    wrapped as RigidObjects. Unconnected static objects need no state wrapper.
    """
    children = {joint.child for joint in definition.joints}
    roots = {joint.parent for joint in definition.joints} - children
    groups = []
    for root in sorted(roots):
        members = {root}
        joints = []
        pending = list(definition.joints)
        while pending:
            found = [j for j in pending if j.parent in members]
            if not found:
                break
            for joint in found:
                joints.append(joint.name)
                members.add(joint.child)
                pending.remove(joint)
        groups.append(
            {
                "root_body": root,
                "root_path": root,
                "body_names": sorted(members),
                "joint_names": joints,
                "initial_joints": {j: definition.parameters.get("initial_joints", {}).get(j, 0.0) for j in joints},
            }
        )
    return groups


def author_native_frame_bonds(stage, entries):
    """Author full-rank rigid bonds through native noncollinear cable sites.

    Newton imports these three hard ball constraints inside each source world.
    Two axial sites constrain bending; the radial site also constrains twist.
    This avoids both a positional-only pin and a separately integrated spring.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom

    from .api import rotation_matrix

    result = []
    for env_id, definition, task_path, paths in entries:
        for spec in definition.deformables:
            for attachment in spec.attachments:
                if attachment["kind"] != "rigid_frame":
                    raise ValueError("Only complete cable frame attachments are supported")
                curves = [
                    prim
                    for prim in Usd.PrimRange(stage.GetPrimAtPath(paths[spec.name]))
                    if prim.IsA(UsdGeom.BasisCurves)
                ]
                if len(curves) != 1:
                    raise ValueError(f"Cannot uniquely find cable geometry below {paths[spec.name]}")
                length = float(np.linalg.norm(spec.vertices[1] - spec.vertices[0]))
                points = np.array([[0, 0, 0], [0, 0, length], [0, spec.radius, length / 2]])
                frame = np.asarray(attachment["parent_frame"])
                targets = points @ rotation_matrix(frame[3:]).T + frame[:3]
                path = f"{task_path}/CableBond_{spec.name}"
                prim = stage.DefinePrim(path, "PhysicsAttachment")
                prim.CreateRelationship("physics:src0").SetTargets([curves[0].GetPath()])
                prim.CreateRelationship("physics:src1").SetTargets([f"{task_path}/{attachment['body']}"])
                attributes = {
                    "type0": (Sdf.ValueTypeNames.Token, "segment"),
                    "type1": (Sdf.ValueTypeNames.Token, "xform"),
                    "indices0": (Sdf.ValueTypeNames.IntArray, [0, 0, 0]),
                    "coords0": (
                        Sdf.ValueTypeNames.Float3Array,
                        [Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, spec.radius, 0)],
                    ),
                    "coords1": (Sdf.ValueTypeNames.Float3Array, [Gf.Vec3f(*point.tolist()) for point in targets]),
                    "stiffness": (Sdf.ValueTypeNames.Float, float("inf")),
                    "damping": (Sdf.ValueTypeNames.Float, 0.0),
                    "attachmentEnabled": (Sdf.ValueTypeNames.Bool, True),
                }
                for name, (kind, value) in attributes.items():
                    prim.CreateAttribute("physics:" + name, kind).Set(value)
                result.append((env_id, spec.name, path))
    return result


class NativeTaskBinding:
    """Install native cable frame constraints and per-substep plastic forces.

    ``entries`` are (environment_id, TaskDefinition, task_prim_path, deformable_paths)
    with exact prim paths, e.g. /World/envs/env_0/Room/Task and
    {"sheet": "/World/envs/env_0/Sheet"}. Keep this object alive for the scene.
    """

    def __init__(self, entries, manager=None):
        from isaaclab_newton.physics import NewtonManager

        from isaaclab.physics import PhysicsEvent

        self.entries = list(entries)
        self.manager = manager or NewtonManager
        self.hook = None
        self.segment_body_ids = {}
        self.bond_joint_ids = {}
        self.authored_rod_inertias = {}
        self.rod_inertia_receipt = {"body_count": 0, "bodies": []}
        from isaaclab.sim import SimulationContext

        stage = SimulationContext.instance().stage
        self.frame_bonds = []
        for env_id, definition, task_path, _ in self.entries:
            for spec in definition.deformables:
                if spec.attachments:
                    path = f"{task_path}/CableBond_{spec.name}"
                    if not stage.GetPrimAtPath(path).IsValid():
                        raise RuntimeError(f"Author cable frame bonds in room sources before cloning: {path}")
                    self.frame_bonds.append((env_id, spec.name, path))
        self.filter_handle = self.manager.register_callback(
            self._filter_bonded_contacts, PhysicsEvent.MODEL_INIT, order=50, name="hiwonder_bonded_cable_contacts"
        )
        self.model_handle = self.manager.register_callback(
            self._provision_surface_collision, PhysicsEvent.MODEL_INIT, order=100, name="hiwonder_surface_collision_sdf"
        )
        self.handle = self.manager.register_callback(
            self._ready, PhysicsEvent.PHYSICS_READY, order=100, name="hiwonder_task_constitutive_forces"
        )

    def _ready(self, _):
        from .native_forces import NewtonTaskForceHook
        from .native_inertia import restore_cable_inertias

        model = self.manager.get_model()
        # PHYSICS_READY precedes solver construction in this SDK. The local VBD
        # manager subsequently notifies BODY_PROPERTIES in its initial FK hook,
        # before graph capture, after constructing caches from these tensors.
        self.rod_inertia_receipt = restore_cable_inertias(model, self.authored_rod_inertias)
        labels = list(model.body_label)
        surfaces = []
        ranges = self.manager.backend.deformable_ranges
        for env_id, definition, task_path, paths in self.entries:
            for spec in definition.deformables:
                path = paths[spec.name]
                if spec.kind == "plastic_shell":
                    matches = [value for key, value in ranges.items() if key == path or key.startswith(path + "/")]
                    if len(matches) != 1 or matches[0][1] != len(spec.vertices):
                        raise RuntimeError(f"Exact paper particle mapping unavailable for {path}: {matches}")
                    surfaces.append((env_id, spec, matches[0][0]))
                for attachment in spec.attachments:
                    if attachment["kind"] != "rigid_frame":
                        raise ValueError("Native task binding requires explicit rigid-frame cable attachments")
                    parent_path = f"{task_path}/{attachment['body']}"
                    parent = [i for i, label in enumerate(labels) if label == parent_path]
                    child = [
                        i for i, label in enumerate(labels) if label.startswith(path) and label.endswith("_edge_body_0")
                    ]
                    if len(parent) != 1 or len(child) != 1:
                        raise RuntimeError(
                            f"Cannot uniquely bind plug/cable frame: {parent_path}, {path};"
                            f" parent={parent}, child={child}"
                        )
                    bond_path = f"{task_path}/CableBond_{spec.name}"
                    joints = [i for i, label in enumerate(model.joint_label) if label.startswith(bond_path + "_site")]
                    if len(joints) != 3:
                        raise RuntimeError(f"Full cable frame bond was not imported: {bond_path}: {joints}")
                    self.bond_joint_ids[env_id, spec.name] = tuple(joints)
                    self.segment_body_ids[env_id, spec.name] = child[0]
        self.hook = NewtonTaskForceHook(model, surfaces)
        self.hook.register(self.manager)

    def _filter_bonded_contacts(self, _):
        """Exclude only capsule portions physically embedded in the plug strain relief."""
        from isaaclab_newton.physics.newton_manager_cfg import NewtonBuilderCfg

        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        from isaaclab_hiwonder.sim.collision_cache import capture_robot_hulls

        self.collision_cache_receipts = []
        for _, _, task_path, _ in self.entries:
            robot_path = task_path.rsplit("/", 1)[0] + "/Robot"
            self.collision_cache_receipts.append(capture_robot_hulls(builder, sim.stage, robot_path))
        for _, definition, task_path, paths in self.entries:
            for spec in definition.deformables:
                if spec.kind == "cable":
                    filter_cable_material_overlap(builder, spec, paths[spec.name])
                for attachment in spec.attachments:
                    parent_path = f"{task_path}/{attachment['body']}"
                    parent = [i for i, label in enumerate(builder.body_label) if label == parent_path]
                    segments = attachment.get("bonded_segments", (attachment["segment"],))
                    children = [
                        i
                        for i, label in enumerate(builder.body_label)
                        if label.startswith(paths[spec.name])
                        and any(label.endswith(f"_edge_body_{s}") for s in segments)
                    ]
                    if len(parent) != 1 or len(children) != len(segments):
                        raise RuntimeError(f"Bonded cable body mapping incomplete for {parent_path}")
                    parent_shapes = [i for i, body in enumerate(builder.shape_body) if body == parent[0]]
                    child_shapes = [i for i, body in enumerate(builder.shape_body) if body in children]
                    for first in parent_shapes:
                        for second in child_shapes:
                            builder.add_shape_collision_filter_pair(first, second)
        from .native_inertia import capture_cable_inertias

        expected = {
            paths[spec.name]: len(spec.edges)
            for _, definition, _, paths in self.entries
            for spec in definition.deformables
            if spec.kind == "cable"
        }
        self.authored_rod_inertias = capture_cable_inertias(builder, expected)

    def _provision_surface_collision(self, _):
        """Add SDF contact representations to existing collision meshes, preserving hulls.

        Newton requires these for edge/face contacts. Rod-only scenes contain no
        surface particles and use the native rigid collision pipeline directly.
        """
        if not any(s.kind in ("surface", "plastic_shell") for _, d, _, _ in self.entries for s in d.deformables):
            return
        import tempfile
        from pathlib import Path

        import newton
        from isaaclab_newton.physics.newton_manager_cfg import NewtonBuilderCfg

        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        cache = Path(tempfile.gettempdir()) / "isaaclab_hiwonder" / "newton_sdf"
        cache.mkdir(parents=True, exist_ok=True)
        selected = {}
        for index, mesh in enumerate(builder.shape_source):
            if not isinstance(mesh, newton.Mesh):
                continue
            if not int(builder.shape_flags[index]) & int(newton.ShapeFlags.COLLIDE_PARTICLES):
                continue
            scale = float(np.max(np.abs(builder.shape_scale[index])))
            previous = selected.get(id(mesh), (mesh, 0.0))
            selected[id(mesh)] = (mesh, max(previous[1], scale))
        for mesh, scale in selected.values():
            if mesh.sdf is None:
                # Small convex pieces retain ~64 samples across their longest
                # dimension; large tables/walls are capped at 2mm world voxels.
                extent = float(np.max(np.ptp(np.asarray(mesh.vertices), axis=0))) * scale
                voxel = min(0.002, max(0.00015, extent / 64)) / max(scale, 1e-9)
                mesh.build_sdf(
                    device=sim.device,
                    target_voxel_size=voxel,
                    margin=2 * voxel,
                    narrow_band_range=(-4 * voxel, 4 * voxel),
                    cache_dir=str(cache),
                )

    def reset(self, environment_mask):
        if self.hook is not None:
            self.hook.reset(np.asarray(environment_mask, dtype=bool))

    def close(self):
        self.handle.deregister()
        self.model_handle.deregister()
        self.filter_handle.deregister()
        # NewtonManager clears force callbacks on STOP; the public API currently
        # has no state-force unregistration method. Close with the owning scene.
