# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Bake initial structural randomization into independently owned room USDs.

One task topology is selected per launch. Structural draws (including robot
geometry, table topology and clutter) require a scene rebuild. Indexed material,
light, actuator and root-state updates are owned by the environment adapter.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from ..assets.robot_models import get_robot_model
from ..config import HiWonderEnvConfig
from ..tasks.backend import author_rigid_usd, native_deformable_config
from ..tasks.native_backend import articulation_groups
from .asset_cache import texture_asset
from .rooms import MaterialSample, RoomParameters, author_mesh, author_room, bind_material, update_material


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _texture(identifier: str, directory: Path, seed: int) -> str | None:
    return texture_asset(identifier, directory, seed)


def _material(fields: dict, prefix: str, index, config, directory: Path, seed: int) -> MaterialSample:
    def get(key):
        value = np.asarray(fields[prefix + key])
        return value if index is None else value[index]

    uv = float(get("uv_scale"))
    return MaterialSample(
        color=tuple(float(v) for v in get("color")),
        roughness=float(get("roughness")),
        metallic=float(get("metallic")),
        specular=float(get("specular")),
        uv_scale=(uv, uv),
        texture_path=_texture(config.texture_ids[int(get("texture_id"))], directory, seed),
    )


def _pose(xform, position, xyzw=(0, 0, 0, 1)):
    ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
    translate = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
    orient = ops.get(UsdGeom.XformOp.TypeOrient) or xform.AddOrientOp()
    translate.Set(Gf.Vec3d(*map(float, position)))
    orient.Set(Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*map(float, xyzw[:3]))))


def _physics_material(stage, path, friction, restitution):
    material = UsdShade.Material(stage.GetPrimAtPath(path))
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(float(restitution))
    return material


def _camera(stage, profile, scale, model):
    camera_path = "/Room/Robot/" + model.camera_path
    mount = UsdGeom.Xform(stage.GetPrimAtPath(camera_path.rsplit("/", 1)[0]))
    optical = Gf.Quatd(profile.optical_quaternion_xyzw[3], Gf.Vec3d(*profile.optical_quaternion_xyzw[:3]))
    renderer = optical * Gf.Quatd(0, Gf.Vec3d(1, 0, 0))
    q = (*renderer.GetImaginary(), renderer.GetReal())
    _pose(mount, np.asarray(profile.optical_translation_m) * scale, q)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    f = profile.focal_length_mm * 0.01
    camera.CreateFocalLengthAttr(f)
    # Match calibrated fx/fy; physical filmback remains independently recorded.
    camera.CreateHorizontalApertureAttr(profile.width * f / profile.intrinsics[0])
    camera.CreateVerticalApertureAttr(profile.height * f / profile.intrinsics[4])
    camera.CreateFStopAttr(profile.f_stop or 0.0)
    camera.CreateFocusDistanceAttr(profile.focus_distance_m)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*profile.clipping_range_m))
    if profile.distortion_model != "brown_conrady":
        raise ValueError("Scene authoring currently supports Brown-Conrady camera profiles")
    prim = camera.GetPrim()
    prim.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prefix = "omni:lensdistortion:opencvPinhole:"
    prim.CreateAttribute(prefix + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(profile.width, profile.height))
    for name, value in zip(
        ("fx", "fy", "cx", "cy"),
        (profile.intrinsics[0], profile.intrinsics[4], profile.intrinsics[2], profile.intrinsics[5]),
    ):
        prim.CreateAttribute(prefix + name, Sdf.ValueTypeNames.Float).Set(float(value))
    coefficients = dict(
        zip(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"), profile.distortion_coefficients)
    )
    for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"):
        prim.CreateAttribute(prefix + name, Sdf.ValueTypeNames.Float).Set(float(coefficients.get(name, 0)))


def _confetti(stage, sample, config, support_height):
    path = "/Room/Confetti"
    stage.RemovePrim(path)
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    if config.physical_budget > 8:
        raise ValueError("Physical confetti is bounded to eight rigid cards per environment")
    active = sorted(set(np.flatnonzero(sample["active"]).tolist()) | set(range(config.physical_budget)))
    prototypes, positions, scales, rotations = [], [], [], []
    for i in active:
        shape = config.shapes[int(sample["shape_id"][i])]
        if shape == "disc":
            theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            points = np.column_stack((0.5 * np.cos(theta), 0.5 * np.sin(theta), np.zeros_like(theta)))
        elif shape == "triangle":
            points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0, 0.5, 0)]
        elif shape == "rectangle":
            points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]
        else:
            raise ValueError(f"Unsupported confetti shape {shape}")
        mat_path = f"{path}/Looks/Color{i}"
        update_material(
            stage,
            mat_path,
            MaterialSample(
                color=tuple(map(float, sample["color"][i])),
                opacity=float(sample["opacity"][i]),
                metallic=float(sample["reflectivity"][i]),
                roughness=0.4,
            ),
        )
        position = np.asarray(sample["position"][i]).copy()
        position[2] = support_height + config.elevation_m
        size = float(sample["size"][i])
        aspect = float(sample["aspect_ratio"][i])
        yaw = float(sample["yaw"][i])
        if i < config.physical_budget:
            body = UsdGeom.Xform.Define(stage, f"/Room/PhysicalConfetti/Piece{i}")
            _pose(body, position, (0, 0, math.sin(yaw / 2), math.cos(yaw / 2)))
            enabled = bool(sample["active"][i])
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(not enabled)
            # Paper-like estimate: 80 g/m², 0.1mm thickness, at least 1mg.
            mass = max(size * size * aspect * 0.08, 1e-6)
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
            outline = np.asarray(points) * [size, size * aspect, 1]
            count = len(outline)
            vertices = np.concatenate((outline + [0, 0, -0.00005], outline + [0, 0, 0.00005]))
            faces = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
            faces.extend((j, (j + 1) % count, (j + 1) % count + count, j + count) for j in range(count))
            mesh = author_mesh(
                stage, str(body.GetPath()) + "/Shape", vertices, faces, material=mat_path, collision=True, convex=True
            )
            UsdPhysics.CollisionAPI(mesh.GetPrim()).CreateCollisionEnabledAttr(enabled)
            mesh.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
            mesh.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0002)
            mesh.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0)
            if not enabled:
                body.CreateVisibilityAttr("invisible")
            continue
        proto = f"{path}/Prototypes/Piece{i}"
        author_mesh(stage, proto, points, [tuple(range(len(points)))], material=mat_path, collision=False)
        prototypes.append(proto)
        positions.append(Gf.Vec3f(*map(float, position)))
        scales.append(Gf.Vec3f(size, size * aspect, 1))
        rotations.append(Gf.Quath(math.cos(yaw / 2), Gf.Vec3h(0, 0, math.sin(yaw / 2))))
    instancer.CreatePrototypesRel().SetTargets(prototypes)
    instancer.CreateProtoIndicesAttr(list(range(len(prototypes))))
    instancer.CreatePositionsAttr(positions)
    instancer.CreateScalesAttr(scales)
    instancer.CreateOrientationsAttr(rotations)


def _distractors(stage, sample, config, directory, seed, support_height):
    paths = []
    for i in range(len(sample["active"])):
        path = f"/Room/Distractors/Object{i}"
        shape = config.shapes[int(sample["shape_id"][i])]
        body = UsdGeom.Xform.Define(stage, path)
        size = sample["size"][i]
        position = np.asarray(sample["position"][i]).copy()
        position[2] = support_height + size[2] / 2 + 0.001
        _pose(body, position, sample["quaternion_xyzw"][i])
        if shape == "box":
            geom = UsdGeom.Cube.Define(stage, path + "/Shape")
            geom.CreateSizeAttr(1)
            geom.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        elif shape == "sphere":
            # Bake the sampled ellipsoid rather than nonuniformly scaling a
            # sphere primitive, whose PhysX collider can disagree with visuals.
            sectors, rings = 20, 10
            points = [(0, 0, 0.5), (0, 0, -0.5)]
            for ring in range(1, rings):
                polar = math.pi * ring / rings
                points.extend(
                    (
                        0.5 * math.sin(polar) * math.cos(2 * math.pi * j / sectors),
                        0.5 * math.sin(polar) * math.sin(2 * math.pi * j / sectors),
                        0.5 * math.cos(polar),
                    )
                    for j in range(sectors)
                )
            faces = [(0, 2 + j, 2 + (j + 1) % sectors) for j in range(sectors)]
            for ring in range(rings - 2):
                start, end = 2 + ring * sectors, 2 + (ring + 1) * sectors
                faces.extend(
                    (start + j, end + j, end + (j + 1) % sectors, start + (j + 1) % sectors) for j in range(sectors)
                )
            last = 2 + (rings - 2) * sectors
            faces.extend((1, last + (j + 1) % sectors, last + j) for j in range(sectors))
            geom = author_mesh(stage, path + "/Shape", np.asarray(points) * size, faces, collision=False)
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
        elif shape == "cylinder":
            count = 32
            points = [
                (math.cos(2 * math.pi * j / count) / 2, math.sin(2 * math.pi * j / count) / 2, z)
                for z in (-0.5, 0.5)
                for j in range(count)
            ]
            faces = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
            faces.extend((j, (j + 1) % count, (j + 1) % count + count, j + count) for j in range(count))
            geom = author_mesh(stage, path + "/Shape", np.asarray(points) * size, faces, collision=False)
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
        else:
            raise ValueError(f"Unsupported distractor shape {shape}")
        material_path = f"/Room/Looks/Distractor{i}"
        update_material(stage, material_path, _material(sample, "material_", i, config.material, directory, seed + i))
        bind_material(stage, geom.GetPrim(), material_path)
        physical = bool(sample["active"][i] and sample["physical"][i])
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(not physical)
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(float(sample["mass"][i]))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(physical)
        if not sample["active"][i]:
            body.CreateVisibilityAttr("invisible")
        if physical:
            physics_mat = _physics_material(
                stage, material_path, sample["material_friction"][i], sample["material_restitution"][i]
            )
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(physics_mat, materialPurpose="physics")
        paths.append(path)
    return paths


def build_scene(profile: HiWonderEnvConfig, definitions: list, snapshot: dict, output_directory: Path):  # noqa: C901
    """Return an SDK scene configuration and exact batched state/contact bindings.

    All definitions must share a body/joint/deformable topology. Per-environment
    CAD variants, rooms, materials, clutter, table and task geometry are baked.
    ``robot_initial_pose`` and ``task_support_offsets_m`` are environment-local;
    add ``scene.env_origins`` when writing simulation world poses.
    """
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, VisualMaterialCfg
    from isaaclab.cloner import CloneCfg
    from isaaclab.cloner.cloner_strategies import sequential
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg

    if len(definitions) != profile.num_envs:
        raise ValueError("One task definition per environment is required")
    definition = definitions[0]
    model = get_robot_model(profile.robot_model)

    def topology(d):
        return (
            [b.name for b in d.bodies],
            [(j.name, j.parent, j.child) for j in d.joints],
            [(s.name, s.kind) for s in d.deformables],
        )

    if any(topology(d) != topology(definition) or d.physics_family != definition.physics_family for d in definitions):
        raise ValueError(
            "Select one task topology/backend per launch; heterogeneous task switching requires a scene rebuild"
        )
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    samples = {group: {key: _numpy(value) for key, value in fields.items()} for group, fields in snapshot.items()}
    groups = articulation_groups(definition)
    connected = {name for group in groups for name in group["body_names"]}
    metadata = dict(
        robot_model=model.name,
        robot_joint_names=model.joint_names,
        task_body_assets={},
        task_articulations={},
        task_deformables={},
        task_contacts={},
        distractor_assets={},
        distractor_initial_poses=[],
        confetti_assets={},
        confetti_initial_poses=[],
        room_paths=[],
        fill_light_paths=[],
        room_files=[],
        robot_material_names={},
        robot_collision_cache=[],
        robot_collision_profiles=[],
        support_height_m=[],
        task_support_offsets_m=[],
        robot_initial_pose=[],
        native_task_entries=[],
        actual_room_parameters=[],
        initial_base_positions=samples["base"]["position"].tolist(),
        structural_randomization="baked_initial_snapshot_requires_scene_rebuild",
        clone_strategy="sequential",
        camera_profile=asdict(profile.camera),
        light_photometry={"normalize": True, "intensity_units": "USD normalized intensity", "ambient_intensity": 0.0},
        task_name=definition.name,
    )
    from .robot_collisions import apply_robot_collision_profile, load_collision_profile

    collision_profile = (
        load_collision_profile(model.collision_profile_path, model=model)
        if profile.robot_collision_profile == "manipulation" and any(task.deformables for task in definitions)
        else None
    )
    for env_id, task in enumerate(definitions):
        sampled = {group: {key: value[env_id] for key, value in fields.items()} for group, fields in samples.items()}
        table, room, robot = sampled["table"], sampled["room"], sampled["robot"]
        height = float(task.parameters.get("support_height_m", table["height"]))
        metadata["support_height_m"].append(height)
        metadata["task_support_offsets_m"].append([0, 0, height])
        texture_dir = output_directory / "textures"
        seed = profile.seed + env_id * 100003
        table_material = _material(table, "material_", None, profile.randomization.table.material, texture_dir, seed)
        walls = tuple(
            _material(room, "wall_", i, profile.randomization.lighting.wall_material, texture_dir, seed + i + 10)
            for i in range(4)
        )
        parameters = RoomParameters(
            width=profile.room.width_m,
            depth=profile.room.depth_m,
            height=profile.room.height_m,
            wall_thickness=profile.room.wall_thickness_m,
            table_width=float(table["width"]),
            table_depth=float(table["depth"]),
            table_height=height,
            table_thickness=min(float(table["thickness"]), 0.015 if height <= 0.05 else height * 0.5),
            table_shape=profile.randomization.table.shapes[int(table["shape_id"])],
            table_edge_radius=float(table["edge_radius"]),
            table_material=table_material,
            wall_materials=walls,
            light_positions=tuple(tuple(map(float, row)) for row in room["light_position"]),
            light_colors=tuple(tuple(map(float, row)) for row in room["light_color"]),
            light_intensities=tuple(map(float, room["light_intensity"])),
            light_beam_angles=tuple(map(float, room["light_beam_angle_degrees"])),
            confetti_count=0,
            seed=seed,
        )
        path = output_directory / f"room_{env_id}.usda"
        if path.exists():
            path.unlink()
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        paths = author_room(stage, "/Room", parameters)
        metadata["actual_room_parameters"].append(asdict(parameters))
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Room"))
        table_prim = stage.GetPrimAtPath(paths.table)
        UsdPhysics.RigidBodyAPI.Apply(table_prim).CreateKinematicEnabledAttr(True)
        table_physics = _physics_material(
            stage, paths.materials["table"], table["material_friction"], table["material_restitution"]
        )
        UsdShade.MaterialBindingAPI.Apply(table_prim).Bind(table_physics, materialPurpose="physics")
        background = profile.randomization.lighting.backgrounds[int(room["background_id"])]
        if background != "none":
            from dataclasses import replace

            texture = _texture(background, texture_dir, seed)
            for i in range(4):
                update_material(stage, paths.materials[f"wall_{i}"], replace(walls[i], texture_path=texture))
        for i, light_path in enumerate(paths.lights):
            prim = stage.GetPrimAtPath(light_path)
            light = UsdLux.LightAPI(prim)
            # Keep sampled source power independent of the 4cm emitter radius.
            # Unnormalized small area lights were overwhelmed by default ambient.
            light.CreateNormalizeAttr(True)
            kind = profile.randomization.lighting.types[int(room["light_type_id"][i])]
            if profile.randomization.lighting.profile in ("white_disco", "rgb_disco") or (
                profile.randomization.lighting.motion_mode == "wrist_view"
            ):
                kind = "spot"
            if kind not in ("sphere", "spot"):
                raise ValueError(f"Unsupported room light type {kind}")
            if kind == "sphere":
                prim.RemoveAPI(UsdLux.ShapingAPI)
            direction = Gf.Vec3d(*map(float, room["light_direction"][i]))
            q = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction).GetQuat()
            _pose(UsdGeom.Xformable(prim), room["light_position"][i], (*q.GetImaginary(), q.GetReal()))
            light.CreateIntensityAttr(float(room["light_intensity"][i]) if room["light_active"][i] else 0.0)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(float(room["light_temperature_kelvin"][i]))
            UsdLux.ShadowAPI.Apply(prim).CreateShadowEnableAttr(bool(room["light_shadows"][i]))
        from .lighting import author_fill_light

        fill_path = author_fill_light(
            stage,
            "/Room",
            (0.0, 0.0, min(height + 1.3, profile.room.height_m - 0.2)),
            float(room["white_fill_intensity"]),
        )
        variant = profile.randomization.robot.geometry_variants[int(robot["geometry_variant_id"])]
        robot_file = model.asset_path(variant)
        if not robot_file.is_file():
            raise ValueError(f"No coherent robot asset for variant {variant}")
        robot_prim = stage.DefinePrim("/Room/Robot", "Xform")
        robot_prim.GetReferences().AddReference(str(robot_file.resolve()))
        robot_prim.SetCustomDataByKey("hiwonder:sourceAsset", str(robot_file.resolve()))
        base = sampled["base"]
        position = base["position"].copy()
        position[2] = height
        qbase = Gf.Quatd(float(base["quaternion_xyzw"][3]), Gf.Vec3d(*map(float, base["quaternion_xyzw"][:3])))
        yaw = Gf.Quatd(math.cos(math.pi / 4), Gf.Vec3d(0, 0, math.sin(math.pi / 4)))
        qbase = qbase * yaw
        quaternion = (*qbase.GetImaginary(), qbase.GetReal())
        _pose(UsdGeom.Xformable(robot_prim), position, quaternion)
        metadata["robot_initial_pose"].append([*position.tolist(), *quaternion])
        scale = float(robot["geometry_scale"])
        _camera(stage, profile.camera, scale, model)
        for link_id, link_name in enumerate(model.links):
            material_path = f"/Room/Looks/Robot_{link_name}"
            update_material(
                stage,
                material_path,
                _material(
                    robot, "material_", link_id, profile.randomization.robot.material, texture_dir, seed + 30 + link_id
                ),
            )
            body = stage.GetPrimAtPath("/Room/Robot/" + link_name)
            UsdShade.MaterialBindingAPI.Apply(body).Bind(
                UsdShade.Material(stage.GetPrimAtPath(material_path)),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            physics_material = _physics_material(
                stage, material_path, robot["material_friction"][link_id], robot["material_restitution"][link_id]
            )
            UsdShade.MaterialBindingAPI.Apply(body).Bind(
                physics_material, materialPurpose="physics", bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            for descendant in Usd.PrimRange(body):
                if descendant.IsA(UsdGeom.Mesh) and "/visuals/" in str(descendant.GetPath()):
                    vertices = np.asarray(UsdGeom.Mesh(descendant).GetPointsAttr().Get())
                    axes = np.argsort(np.ptp(vertices, axis=0))[-2:]
                    coords = vertices[:, axes]
                    coords = (coords - coords.min(0)) / np.maximum(np.ptp(coords, axis=0), 1e-8)
                    UsdGeom.PrimvarsAPI(descendant).CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
                    ).Set([Gf.Vec2f(*map(float, p)) for p in coords])
            mass = UsdPhysics.MassAPI(body)
            mass.CreateMassAttr(mass.GetMassAttr().Get() * float(robot["mass_scale"][link_id]) / scale**3)
            inertia = mass.GetDiagonalInertiaAttr().Get()
            mass.CreateDiagonalInertiaAttr(inertia * (float(robot["inertia_scale"][link_id]) / scale**5))
        for i, joint_name in enumerate(model.joint_names):
            joint = stage.GetPrimAtPath("/Room/Robot/joints/" + joint_name)
            actuator = sampled["actuator"]
            drive = UsdPhysics.DriveAPI(joint, "angular")
            drive.CreateStiffnessAttr(float(actuator["stiffness"][i]) * math.pi / 180)
            drive.CreateDampingAttr(float(actuator["damping"][i]) * math.pi / 180)
            drive.CreateMaxForceAttr(float(actuator["effort_limit"][i]))
            joint.AddAppliedSchema("PhysxJointAPI")
            joint.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(
                float(actuator["velocity_limit"][i]) * 180 / math.pi
            )
            joint.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(
                float(actuator["friction"][i])
            )
            joint.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(actuator["armature"][i]))
        metadata["robot_collision_profiles"].append(
            apply_robot_collision_profile(
                stage,
                "/Room/Robot",
                mode=profile.robot_collision_profile,
                self_collisions=profile.robot_self_collisions,
                native=bool(task.deformables),
                profile=collision_profile,
                model=model,
            )
        )
        if task.deformables and profile.robot_collision_profile == "cad":
            from .collision_cache import apply_cached_robot_collisions

            metadata["robot_collision_cache"].append(apply_cached_robot_collisions(stage, "/Room/Robot"))
        task_file = output_directory / f"task_{env_id}.usda"
        if task_file.exists():
            task_file.unlink()
        author_rigid_usd(task, task_file)
        task_root = stage.DefinePrim("/Room/Task", "Xform")
        task_root.GetReferences().AddReference(str(task_file))
        _pose(UsdGeom.Xformable(task_root), (0, 0, height))
        if task.joints:
            from .task_anchors import bind_task_world_anchors

            bind_task_world_anchors(stage, "/Room/Task")
        for body in task.bodies:
            if body.static and body.name not in connected:
                UsdPhysics.RigidBodyAPI.Apply(
                    stage.GetPrimAtPath(f"/Room/Task/{body.name}")
                ).CreateKinematicEnabledAttr(True)
        # Each environment's native shape is part of its room prototype; a
        # spawnless view avoids a second clone axis mixing unrelated draws.
        for spec in task.deformables:
            deformable_path = f"/Room/Task/Deformables/{spec.name}"
            cfg = native_deformable_config(spec, deformable_path, output_directory / f"deformables_{env_id}")
            UsdGeom.Xform.Define(stage, "/Room/Task/Deformables")
            with sim_utils.use_stage(stage):
                cfg.spawn.func(deformable_path, cfg.spawn)
        if task.deformables:
            from ..tasks.native_backend import author_native_frame_bonds

            # Newton parses source worlds during scene cloning, before the
            # environment's later NativeTaskBinding lifecycle registration.
            author_native_frame_bonds(
                stage,
                [
                    (
                        env_id,
                        task,
                        "/Room/Task",
                        {spec.name: f"/Room/Task/Deformables/{spec.name}" for spec in task.deformables},
                    )
                ],
            )
        _confetti(stage, sampled["confetti"], profile.randomization.confetti, height)
        distractors = _distractors(
            stage, sampled["distractor"], profile.randomization.distractor, texture_dir, seed + 100, height
        )
        initial = sampled["distractor"]["position"].copy()
        initial[:, 2] = height + sampled["distractor"]["size"][:, 2] / 2 + 0.001
        metadata["distractor_initial_poses"].append(
            np.concatenate((initial, sampled["distractor"]["quaternion_xyzw"]), axis=-1).tolist()
        )
        budget = profile.randomization.confetti.physical_budget
        initial = sampled["confetti"]["position"][:budget].copy()
        initial[:, 2] = height + profile.randomization.confetti.elevation_m
        yaw = sampled["confetti"]["yaw"][:budget]
        orientation = np.column_stack((np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)))
        metadata["confetti_initial_poses"].append(np.concatenate((initial, orientation), axis=-1).tolist())
        absolute_root = f"/World/envs/env_{env_id}"
        concrete_paths = asdict(paths)

        def concrete(value):
            if isinstance(value, str):
                return absolute_root + value
            if isinstance(value, (list, tuple)):
                return [concrete(v) for v in value]
            return {k: concrete(v) for k, v in value.items()}

        concrete_paths = concrete(concrete_paths)
        concrete_paths["distractors"] = [absolute_root + p for p in distractors]
        concrete_paths["robot_materials"] = {name: absolute_root + f"/Room/Looks/Robot_{name}" for name in model.links}
        metadata["room_paths"].append(concrete_paths)
        metadata["fill_light_paths"].append(absolute_root + fill_path)
        metadata["room_files"].append(str(path))
        if task.deformables:
            metadata["native_task_entries"].append(
                (
                    env_id,
                    task,
                    absolute_root + "/Room/Task",
                    {s.name: absolute_root + f"/Room/Task/Deformables/{s.name}" for s in task.deformables},
                )
            )
        stage.GetRootLayer().Save()
    scene = InteractiveSceneCfg(
        num_envs=profile.num_envs,
        env_spacing=max(profile.room.env_spacing_m, profile.room.width_m + 2 * profile.room.wall_thickness_m + 0.1),
        replicate_physics=bool(definition.deformables),
        filter_collisions=True,
        clone_cfg=CloneCfg(clone_strategy=sequential),
    )
    scene.room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.MultiUsdFileCfg(usd_path=metadata["room_files"], activate_contact_sensors=True),
    )
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Room/Robot",
        spawn=None,
        joint_ordering=model.joint_names,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(zip(model.joint_names, model.nominal_joint_positions_rad))
        ),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    scene.table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Room/Tabletop", spawn=None)
    for index in range(profile.randomization.distractor.max_count):
        key = f"distractor_{index}"
        setattr(scene, key, RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Room/Distractors/Object{index}", spawn=None))
        metadata["distractor_assets"][index] = key
    for index in range(profile.randomization.confetti.physical_budget):
        key = f"confetti_{index}"
        setattr(
            scene, key, RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Room/PhysicalConfetti/Piece{index}", spawn=None)
        )
        metadata["confetti_assets"][index] = key
    for body in definition.bodies:
        if body.name not in connected:
            key = "task_body_" + body.name
            setattr(scene, key, RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/Room/Task/{body.name}", spawn=None))
            metadata["task_body_assets"][body.name] = key
    for group in groups:
        key = "task_articulation_" + group["root_body"]
        setattr(
            scene,
            key,
            ArticulationCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Room/Task/{group['root_path']}",
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(joint_pos=group["initial_joints"]),
                actuators={"passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
            ),
        )
        metadata["task_articulations"][group["root_body"]] = dict(asset_key=key, **group)
    for spec in definition.deformables:
        key = "task_deformable_" + spec.name
        cfg = native_deformable_config(
            spec, f"{{ENV_REGEX_NS}}/Room/Task/Deformables/{spec.name}", output_directory / "deformables"
        )
        setattr(scene, key, cfg)
        cfg.spawn = None
        metadata["task_deformables"][spec.name] = key
    actors = {body.name: f"{{ENV_REGEX_NS}}/Room/Task/{body.name}" for body in definition.bodies}
    from .contact_filtering import robot_contact_actors

    hand_actors, contact_aliases = robot_contact_actors(model, "{ENV_REGEX_NS}/Room/Robot")
    actors.update(hand_actors)
    actors["table"] = "{ENV_REGEX_NS}/Room/Tabletop"
    filter_layout = {name: [path] for name, path in actors.items()}
    if definition.physics_family in ("rigid", "articulated"):
        from .contact_filtering import contact_filter_layout

        # Inspect the authored source rather than guessing primitive counts.
        # Compound actor filters work with unequal collider counts, provided
        # collider leaf names do not alias sensing/filtering body names.
        contact_stage = Usd.Stage.Open(metadata["room_files"][0])
        colliders = {}
        for actor, path in actors.items():
            source_path = path.replace("{ENV_REGEX_NS}", "")
            colliders[actor] = [
                "{ENV_REGEX_NS}" + str(prim.GetPath())
                for prim in Usd.PrimRange(contact_stage.GetPrimAtPath(source_path))
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            ]
        filter_layout = contact_filter_layout(actors, colliders)
    for actor, path in actors.items():
        if actor == "table":
            continue
        labels = [label for label in actors if label != actor for _ in filter_layout[label]]
        key = "contact_" + actor
        filters = [target for label in actors if label != actor for target in filter_layout[label]]
        setattr(scene, key, ContactSensorCfg(prim_path=path, update_period=0, filter_prim_paths_expr=filters))
        metadata["task_contacts"][actor] = dict(
            actor_aliases=contact_aliases.get(actor, (actor,)),
            filter_aliases=contact_aliases,
            sensor_key=key,
            filter_labels=labels,
            filter_paths=filters,
            actor_filter_columns={
                label: [i for i, value in enumerate(labels) if value == label] for label in dict.fromkeys(labels)
            },
        )
    for name, path in {
        "table": "Table",
        **{f"wall_{i}": f"Wall{i}" for i in range(4)},
        **{f"robot_{name}": f"Robot_{name}" for name in model.links},
    }.items():
        key = "material_" + name
        setattr(
            scene,
            key,
            VisualMaterialCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Room/Looks/{path}",
                spawn=None,
                channels=("color", "roughness", "metallic", "specular", "uv_scale"),
            ),
        )
        if name.startswith("robot_"):
            metadata["robot_material_names"][name.removeprefix("robot_")] = key
    scene.wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Room/Robot/" + model.camera_path,
        spawn=None,
        width=profile.camera.width,
        height=profile.camera.height,
        update_period=1 / profile.camera.fps,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        renderer_cfg=IsaacRtxRendererCfg(
            enable_scene_partitioning=True,
            global_settings=IsaacRtxRendererGlobalSettingsCfg(
                ambient_light_intensity=0.0,
                enable_direct_lighting=True,
                enable_shadows=True,
            ),
        ),
    )
    metadata["asset_hashes"] = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in metadata["room_files"]
    }
    metadata["native_entries"] = metadata["native_task_entries"]
    return scene, metadata
