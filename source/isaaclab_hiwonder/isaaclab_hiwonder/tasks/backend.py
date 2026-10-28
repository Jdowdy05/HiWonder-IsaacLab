# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD scene authoring and per-substep task constitutive-force integration.

Simulator imports are lazy. Authoring is not contact validation. The caller must
bind measured state and apply every returned force before solver integration.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .api import SceneState, rotation_matrix
from .mechanics import PaperCreaseModel, cable_attachment_wrench, detent_torque
from .screw_cap import ScrewCap


@dataclass
class TaskForces:
    """Body wrenches are referenced at the actor origin; particle forces at nodes."""

    joint_efforts: dict[str, float] = field(default_factory=dict)
    particle_forces: dict[str, np.ndarray] = field(default_factory=dict)
    body_forces: dict[str, np.ndarray] = field(default_factory=dict)
    body_torques: dict[str, np.ndarray] = field(default_factory=dict)


class TaskMechanics:
    """One instance per environment; state persists only within an episode."""

    def __init__(self, definition):
        self.definition = definition
        self.cap = ScrewCap(definition.parameters) if definition.name == "bottle_cap" else None
        self.paper = {
            s.name: PaperCreaseModel(
                s.parameters["hinges"], s.parameters["hinge_stiffness"], s.parameters["yield_angle"]
            )
            for s in definition.deformables
            if s.kind == "plastic_shell"
        }

    def forces(self, state: SceneState, dt=0.0) -> TaskForces:
        result = TaskForces()
        p = self.definition.parameters
        for joint in self.definition.joints:
            if "detent_states" in joint.parameters:
                result.joint_efforts[joint.name] = detent_torque(
                    state.joint_positions[joint.name],
                    state.joint_velocities[joint.name],
                    joint.parameters["detent_states"],
                )
        if self.definition.name == "bottle_cap":
            force, torque, reaction, moment = self.cap.forces(state, dt)
            result.body_forces.update(cap=force, bottle=reaction)
            result.body_torques.update(cap=torque, bottle=moment)
        elif self.definition.name == "key_lock":
            theta = state.joint_positions["lock_angle"]
            key = state.actors["key"]
            inserted = np.linalg.norm(np.asarray(key.position) - p["seat"]) < 0.0015
            aligned = rotation_matrix(key.orientation)[2, 2] > 0.99
            if not inserted or not aligned or not state.touching("key", "cylinder"):
                # Tumblers resist cylinder motion until the key physically engages.
                result.joint_efforts["lock_angle"] = float(
                    np.clip(-2 * theta - 0.03 * state.joint_velocities["lock_angle"], -0.1, 0.1)
                )
            # Cam energy U=k*(x + r*sin(theta))^2/2 gives reciprocal bolt/cylinder reactions.
            x = state.joint_positions["bolt_travel"]
            error = x + p["cam_radius"] * np.sin(theta)
            bolt_force = -1500 * error - 2 * (
                state.joint_velocities["bolt_travel"]
                + p["cam_radius"] * np.cos(theta) * state.joint_velocities["lock_angle"]
            )
            result.joint_efforts["bolt_travel"] = float(np.clip(bolt_force, -4, 4))
            result.joint_efforts["lock_angle"] = float(
                np.clip(
                    result.joint_efforts.get("lock_angle", 0) + bolt_force * p["cam_radius"] * np.cos(theta), -0.1, 0.1
                )
            )
        for spec in self.definition.deformables:
            if spec.name in self.paper:
                result.particle_forces[spec.name] = self.paper[spec.name].forces(state.deformable_positions[spec.name])
            for attachment in spec.attachments:
                if attachment.get("kind") == "rigid_frame":
                    # Native noncollinear-site constraints own full pose coupling.
                    # No particle-pin substitute is applied by the CPU reference.
                    continue
                ids = np.asarray(attachment["nodes"])
                forces, torque = cable_attachment_wrench(
                    state.deformable_positions[spec.name][ids],
                    state.deformable_velocities[spec.name][ids],
                    state.actors[attachment["body"]],
                    attachment["anchors"],
                    attachment["stiffness"],
                    attachment["damping"],
                )
                all_forces = result.particle_forces.setdefault(
                    spec.name, np.zeros_like(state.deformable_positions[spec.name])
                )
                np.add.at(all_forces, ids, forces)
                body = attachment["body"]
                result.body_forces[body] = result.body_forces.get(body, np.zeros(3)) - forces.sum(axis=0)
                result.body_torques[body] = result.body_torques.get(body, np.zeros(3)) + torque
        return result


def author_rigid_usd(task, path):
    """Write actual convex-compound colliders, masses, materials, joints and limits.

    Root ``/Task`` has support-local transforms. Mesh vertices are meters; angular
    USD joint limits are converted to degrees here only. Returns named prim paths.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Task")
    stage.SetDefaultPrim(root.GetPrim())
    result = {}
    connected = {j.parent for j in task.joints} | {j.child for j in task.joints}
    roots = {j.parent for j in task.joints} - {j.child for j in task.joints}

    def orient(xform, position, orientation):
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
        x, y, z, w = orientation
        xform.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

    for body in task.bodies:
        body_path = f"/Task/{body.name}"
        result[body.name] = body_path
        xform = UsdGeom.Xform.Define(stage, body_path)
        state = task.initial[body.name]
        orient(xform, state.position, state.orientation)
        if not body.static or body.name in connected:
            rigid = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
            rigid.CreateRigidBodyEnabledAttr(True)
            # Fixed bases use world joints, avoiding unsupported kinematic articulation roots.
            mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
            mass.CreateMassAttr(body.mass)
            if not body.shapes:
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7, 1e-7, 1e-7))
        if body.name in roots:
            UsdPhysics.ArticulationRootAPI.Apply(xform.GetPrim())
            fixed = UsdPhysics.FixedJoint.Define(stage, f"/Task/fixed_{body.name}")
            fixed.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in state.position]))
            x, y, z, w = state.orientation
            fixed.CreateLocalRot0Attr(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
        material = UsdShade.Material.Define(stage, f"/Task/material_{body.name}")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(body.friction)
        physics_material.CreateDynamicFrictionAttr(body.friction)
        physics_material.CreateRestitutionAttr(body.restitution)
        for index, shape in enumerate(body.shapes):
            shape_path = f"{body_path}/collision_{body.name}_{index}"
            if shape.kind == "box":
                geom = UsdGeom.Cube.Define(stage, shape_path)
                geom.CreateSizeAttr(1)
                geom.AddScaleOp().Set(Gf.Vec3f(*shape.dimensions))
            elif shape.kind == "cylinder":
                geom = UsdGeom.Cylinder.Define(stage, shape_path)
                geom.CreateRadiusAttr(shape.dimensions[0])
                geom.CreateHeightAttr(shape.dimensions[1])
                geom.CreateAxisAttr("Z")
            elif shape.kind == "sphere":
                geom = UsdGeom.Sphere.Define(stage, shape_path)
                geom.CreateRadiusAttr(shape.dimensions[0])
            elif shape.kind == "mesh":
                geom = UsdGeom.Mesh.Define(stage, shape_path)
                geom.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in shape.vertices])
                geom.CreateFaceVertexCountsAttr([3] * len(shape.triangles))
                geom.CreateFaceVertexIndicesAttr(shape.triangles.ravel().tolist())
                geom.CreateSubdivisionSchemeAttr("none")
                UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
            else:
                raise ValueError(f"Unsupported task collider {shape.kind}")
            orient(geom, shape.position, shape.orientation)
            # Explicit T,R,S ordering: dimensions must never scale collider offsets.
            ops = geom.GetOrderedXformOps()
            geom.SetXformOpOrder(
                sorted(
                    ops,
                    key=lambda op: {"xformOp:translate": 0, "xformOp:orient": 1, "xformOp:scale": 2}[op.GetOpName()],
                )
            )
            geom.CreateDisplayColorAttr([Gf.Vec3f(*body.color)])
            if shape.collision:
                UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material, materialPurpose="physics")
    for spec in task.joints:
        cls = UsdPhysics.RevoluteJoint if spec.kind == "revolute" else UsdPhysics.PrismaticJoint
        joint = cls.Define(stage, f"/Task/{spec.name}")
        result[spec.name] = str(joint.GetPath())
        joint.CreateBody0Rel().SetTargets([Sdf.Path(result[spec.parent])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(result[spec.child])])
        joint.CreateAxisAttr("XYZ"[int(np.argmax(np.abs(spec.axis)))])
        for index, label in enumerate((spec.parent, spec.child)):
            state = task.initial[label]
            position = rotation_matrix(state.orientation).T @ (np.asarray(spec.position) - state.position)
            getattr(joint, f"CreateLocalPos{index}Attr")(Gf.Vec3f(*map(float, position)))
            x, y, z, w = state.orientation
            getattr(joint, f"CreateLocalRot{index}Attr")(Gf.Quatf(float(w), Gf.Vec3f(float(-x), float(-y), float(-z))))
        scale = 180 / np.pi if spec.kind == "revolute" else 1
        joint.CreateLowerLimitAttr(spec.lower * scale)
        joint.CreateUpperLimitAttr(spec.upper * scale)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if spec.kind == "revolute" else "linear")
        # Angular USD drive gains are per degree; task mechanics use SI radians.
        drive.CreateStiffnessAttr(spec.stiffness / scale)
        drive.CreateDampingAttr(spec.damping / scale)
        drive.CreateMaxForceAttr(spec.effort_limit)
        drive.CreateTargetPositionAttr(spec.rest * scale)
    stage.GetRootLayer().Save()
    return result


def author_surface_usd(spec, path):
    """Write indexed garment/paper surface without replacing topology by a rectangle."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Surface")
    mesh = UsdGeom.Mesh.Define(stage, "/Surface/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in spec.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(spec.triangles))
    mesh.CreateFaceVertexIndicesAttr(spec.triangles.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def native_deformable_config(spec, prim_path, asset_directory):
    """Build actual Isaac Lab 3.0 Newton/VBD asset configuration.

    Rigid-frame attachments require NewtonTaskForceHook on the native first rod
    segment and plug body; a positional particle pin is insufficient.
    Plastic shell needs TaskMechanics particle forces in addition to this config.
    """
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import CableObjectCfg, DeformableObjectCfg

    if spec.kind == "cable":
        if spec.cable_material is None:
            raise ValueError("Native cable requires explicit continuum moduli in CableMaterialSpec")
        material = spec.cable_material
        return CableObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.CableCfg(
                positions=[tuple(p) for p in spec.vertices],
                physics_material=sim_utils.CableMaterialCfg(
                    thickness=2 * spec.radius,
                    density=spec.density,
                    stretch_stiffness=material.stretch_modulus_pa,
                    bend_stiffness=material.bend_modulus_pa,
                    shear_stiffness=material.shear_modulus_pa,
                    twist_stiffness=material.twist_modulus_pa,
                ),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            ),
        )
    path = Path(asset_directory) / f"{spec.name}.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    author_surface_usd(spec, path)
    return DeformableObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(path),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=spec.density * 2 * spec.radius,
                particle_radius=spec.radius,
                tri_ke=spec.stretch_stiffness,
                tri_ka=spec.stretch_stiffness,
                tri_kd=spec.damping,
                edge_ke=spec.bend_stiffness,
                edge_kd=spec.damping,
            ),
        ),
    )
