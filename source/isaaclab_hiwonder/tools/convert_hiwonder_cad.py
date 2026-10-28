# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Convert supplied follower Rhino cached meshes into a six-actuator USD.

Run in an isolated conversion environment with rhino3dm, numpy, trimesh,
fast-simplification and usd-core. The simulator environment must use its bundled
USD, not pip usd-core. Source CAD is read only. Redistribution rights of the
user-supplied CAD remain unverified; generated geometry is a local artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import rhino3dm
import trimesh

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

REGISTRATION = Path(__file__).parents[1] / "isaaclab_hiwonder" / "assets" / "registration.py"
spec = importlib.util.spec_from_file_location("hiwonder_registration", REGISTRATION)
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )


def extract_mesh(geometry) -> trimesh.Trimesh | None:
    """Read saved render tessellation; skip construction curves and empty extrusions."""
    if isinstance(geometry, rhino3dm.Extrusion):
        geometry = geometry.ToBrep(True)
    if not isinstance(geometry, rhino3dm.Brep):
        return None
    vertices, triangles = [], []
    for face in geometry.Faces:
        mesh = face.GetMesh(rhino3dm.MeshType.Render)
        if mesh is None:
            continue
        offset = len(vertices)
        vertices.extend((p.X, p.Y, p.Z) for p in mesh.Vertices)
        for a, b, c, d in mesh.Faces:
            triangles.append((a + offset, b + offset, c + offset))
            if c != d:
                triangles.append((a + offset, c + offset, d + offset))
    if not triangles:
        return None
    return trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)


def usd_mesh(stage, path, mesh, vertices):
    result = UsdGeom.Mesh.Define(stage, path)
    result.CreatePointsAttr(vertices.astype(np.float32).tolist())
    result.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    result.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    result.CreateSubdivisionSchemeAttr("none")
    result.CreateDoubleSidedAttr(True)
    result.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
    return result


def write_usd(output: Path, meshes, scale: float) -> dict:
    """Author local rigid-body frames and joint frames, including coherent scaling."""
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/HiWonder").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("hiwonder:geometryScale", scale)
    root.SetCustomDataByKey("hiwonder:qualification", "CAD_registered_simulation_estimate")
    origin = np.array(reg.LINKS["base"]["origin_mm"])
    colors = {
        0: (0.48, 0.31, 0.10),
        1: (0.035, 0.035, 0.04),
        2: (0.07, 0.07, 0.075),
        3: (0.48, 0.49, 0.5),
        4: (0.025, 0.025, 0.025),
        5: (0.025, 0.025, 0.03),
        6: (0.35, 0.35, 0.36),
        7: (0.05, 0.15, 0.65),
        8: (0.035, 0.09, 0.13),
    }
    materials = {}
    for layer, color in colors.items():
        mat = UsdShade.Material.Define(stage, f"/HiWonder/materials/layer_{layer}")
        shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.18 if layer == 8 else 0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7 if layer in (0, 3, 6) else 0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[layer] = mat
    report = {}
    for name, link in reg.LINKS.items():
        path = f"/HiWonder/{name}"
        body = UsdGeom.Xform.Define(stage, path)
        local_origin = np.array(link["origin_mm"])
        body.AddTranslateOp().Set(Gf.Vec3d(*((local_origin - origin) * 0.001 * scale)))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(link["mass_kg"] * scale**3)
        bounds = []
        visual_triangles = 0
        for index in link["objects"]:
            if index not in meshes:
                continue
            mesh, layer = meshes[index]
            points = (mesh.vertices - local_origin) * 0.001 * scale
            bounds.extend([points.min(0), points.max(0)])
            visual = usd_mesh(stage, path + f"/visuals/part_{index}", mesh, points)
            UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(materials[layer])
            visual_triangles += len(mesh.faces)
            if index in reg.STRUCTURAL_COLLIDERS | reg.SERVO_COLLIDERS:
                collision = usd_mesh(stage, path + f"/collisions/part_{index}", mesh, points)
                collision.CreateVisibilityAttr("invisible")
                UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
                approximation = "convexDecomposition" if index in reg.STRUCTURAL_COLLIDERS else "convexHull"
                UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(approximation)
        bounds = np.asarray(bounds)
        minimum, maximum = bounds.min(0), bounds.max(0)
        size = maximum - minimum
        center = (minimum + maximum) / 2
        inertia = (
            link["mass_kg"]
            * scale**3
            / 12
            * np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2])
        )
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        report[name] = dict(
            mass_kg=link["mass_kg"] * scale**3,
            center_of_mass_m=center.tolist(),
            diagonal_inertia_kg_m2=inertia.tolist(),
            bounds_local_m=[minimum.tolist(), maximum.tolist()],
            visual_triangles=visual_triangles,
        )
    fixed = UsdPhysics.FixedJoint.Define(stage, "/HiWonder/joints/fixed_base")
    fixed.CreateBody1Rel().SetTargets(["/HiWonder/base"])
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    for data in reg.JOINTS:
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/HiWonder/joints/" + data["name"])
        joint.CreateBody0Rel().SetTargets(["/HiWonder/" + data["parent"]])
        joint.CreateBody1Rel().SetTargets(["/HiWonder/" + data["child"]])
        parent = np.array(reg.LINKS[data["parent"]]["origin_mm"])
        child = np.array(reg.LINKS[data["child"]]["origin_mm"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*((child - parent) * 0.001 * scale)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*data["axis"])).GetQuat()
        rotation = Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary()))
        joint.CreateLocalRot0Attr(rotation)
        joint.CreateLocalRot1Attr(rotation)
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(data["limits_deg"][0])
        joint.CreateUpperLimitAttr(data["limits_deg"][1])
        joint.CreateCollisionEnabledAttr(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        # USD angular drives use degrees, unlike Isaac Lab actuator SI gains.
        drive.CreateStiffnessAttr(8.0 * np.pi / 180)
        drive.CreateDampingAttr(0.35 * np.pi / 180)
        drive.CreateMaxForceAttr(1.0 * scale**4)
        drive.CreateTargetPositionAttr(0)
    mount = UsdGeom.Xform.Define(stage, "/HiWonder/gripper_body/camera_mount")
    mount.AddTranslateOp().Set(Gf.Vec3d(*(np.array(reg.CAMERA["optical_translation_m"]) * scale)))
    q = reg.CAMERA["renderer_quaternion_xyzw"]
    mount.AddOrientOp().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    camera = UsdGeom.Camera.Define(stage, "/HiWonder/" + reg.CAMERA_PATH)
    # USD stores lens lengths in tenths of a stage unit; this stage uses meters.
    camera.CreateFocalLengthAttr(reg.CAMERA["focal_length_mm"] * 0.01)
    camera.CreateHorizontalApertureAttr(reg.CAMERA["sensor_aperture_mm"][0] * 0.01)
    camera.CreateVerticalApertureAttr(reg.CAMERA["sensor_aperture_mm"][1] * 0.01)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*reg.CAMERA["clipping_range_m"]))
    camera.CreateFStopAttr(0.0)  # Disable depth of field; physical f-stop is unknown.
    camera.CreateFocusDistanceAttr(reg.CAMERA["focus_distance_m"])
    tool = UsdGeom.Xform.Define(stage, "/HiWonder/gripper_body/tool_center")
    tool.AddTranslateOp().Set(
        Gf.Vec3d(
            *((np.array([-2.0, -181.0, 207.0]) - np.array(reg.LINKS["gripper_body"]["origin_mm"])) * 0.001 * scale)
        )
    )
    stage.GetRootLayer().Save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=reg.ASSET_DIR / "generated")
    parser.add_argument("--variants", action="store_true", help="Also prebuild uniform ±1 percent geometry variants.")
    args = parser.parse_args()
    cad = rhino3dm.File3dm.Read(str(args.source))
    if str(cad.Settings.ModelUnitSystem) != "UnitSystem.Millimeters":
        raise ValueError("Registration requires source CAD units Millimeters")
    assigned = [i for link in reg.LINKS.values() for i in link["objects"]]
    if sorted(assigned) != list(range(len(cad.Objects))):
        raise ValueError("Every CAD object must belong to exactly one registered rigid link")
    args.output.mkdir(parents=True, exist_ok=True)
    meshes, object_report = {}, []
    for index, obj in enumerate(cad.Objects):
        mesh = extract_mesh(obj.Geometry)
        entry = dict(index=index, source_id=str(obj.Attributes.Id), layer=obj.Attributes.LayerIndex)
        if mesh is None:
            entry["status"] = "construction_curve_or_empty_extrusion"
        else:
            entry["source_triangles"] = len(mesh.faces)
            entry["source_bounds_mm"] = mesh.bounds.tolist()
            target = (
                1800
                if index in reg.STRUCTURAL_COLLIDERS
                else (1000 if index == 253 else (180 if obj.Attributes.LayerIndex in (4, 6) else 600))
            )
            if len(mesh.faces) > target:
                mesh = mesh.simplify_quadric_decimation(face_count=target)
            entry["triangles"] = len(mesh.faces)
            entry["status"] = "cached_render_mesh"
            meshes[index] = (mesh, obj.Attributes.LayerIndex)
        object_report.append(entry)
    camera_path = args.output / "camera_preview.json"
    camera_path.write_text(json.dumps(reg.CAMERA, indent=2) + "\n")
    variants = [("hiwonder_soarm101.usdc", 1.0)]
    if args.variants:
        variants += [("hiwonder_soarm101_scale099.usdc", 0.99), ("hiwonder_soarm101_scale101.usdc", 1.01)]
    artifacts = []
    for filename, scale in variants:
        output = args.output / filename
        links = write_usd(output, meshes, scale)
        artifacts.append(dict(file=filename, sha256=sha256(output), scale=scale, links=links))
    manifest = dict(
        source=dict(
            filename=args.source.name,
            sha256=sha256(args.source),
            units="mm",
            object_count=len(cad.Objects),
            source_kind="user_supplied_HiWonder_assembled_follower_CAD",
        ),
        registration_sha256=sha256(REGISTRATION),
        converter_sha256=sha256(Path(__file__)),
        registration=dict(links=reg.LINKS, joints=reg.JOINTS),
        objects=object_report,
        artifacts=artifacts,
        camera=reg.CAMERA,
        zero_pose="CAD assembled pose; all six q=0 rad",
        nominal_joint_positions_rad=list(reg.NOMINAL_JOINT_POSITIONS_RAD),
        manufacturer=dict(
            model="HX-30HM",
            gear_ratio=345,
            voltage_v=11.1,
            stall_torque_kg_cm=30,
            stall_torque_nm=2.941995,
            no_load_seconds_per_60deg=0.19,
            no_load_rad_s=5.5115660589,
            source="supplied kit manual; not continuous ratings",
        ),
        estimates=dict(
            mass="per-link engineering estimates, not weighed",
            inertia="solid bounding-box diagonal about box center, not CAD material integration",
            drive=(
                "simulation-only 8 Nm/rad stiffness, .35 Nm.s/rad damping, 1 Nm effort; not identified physical gains"
            ),
            joint_limits="conservative provisional symmetric travel relative to CAD pose; not hardware encoder limits",
            collision=(
                "per-CAD-component convex decomposition for structures, convex hull for servo housings; "
                "contact qualification pending"
            ),
            camera="mechanical lens plane and lens center; optical center offset and image roll uncalibrated",
            tool_center="approximate finger-tip midpoint, requires grasp validation",
            geometry_variants=(
                "uniform ±1 percent scaling about base support, joint/camera/collision positions scaled; "
                "mass scales cubically and inertia to fifth power"
            ),
        ),
        qualification="CAD_REGISTRATION_ESTIMATE_NOT_PHYSICALLY_CALIBRATED",
        redistribution=(
            "Source license not established; generated meshes are local artifacts, do not publish without rights review"
        ),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(artifacts=artifacts, source=manifest["source"]), indent=2))


if __name__ == "__main__":
    main()
