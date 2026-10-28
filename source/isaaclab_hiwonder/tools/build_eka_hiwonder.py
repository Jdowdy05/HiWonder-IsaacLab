# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Assemble the supplied Eka hand onto the unchanged HiWonder wrist.

Consumes the checked FreeCAD mesh archive shipped with the source STEP. Building
USD does not access motors. Convex collision cooking and contact qualification
are separate gates; source BRep defects remain disclosed in the manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_hiwonder.assets.robot_models import get_robot_model


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mesh(stage, path, vertices, faces):
    result = UsdGeom.Mesh.Define(stage, path)
    result.CreatePointsAttr(np.asarray(vertices, np.float32).tolist())
    result.CreateFaceVertexCountsAttr([3] * len(faces))
    result.CreateFaceVertexIndicesAttr(np.asarray(faces, np.int32).ravel().tolist())
    result.CreateSubdivisionSchemeAttr("none")
    result.CreateDoubleSidedAttr(True)
    return result


def material(stage, name, color, roughness=0.5, metallic=0.0):
    result = UsdShade.Material.Define(stage, "/HiWonder/materials/eka_" + name)
    shader = UsdShade.Shader.Define(stage, str(result.GetPath()) + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def build_variant(variant, output, archive, metadata, precision):
    model, stock = get_robot_model("eka_v6"), get_robot_model("stock")
    scale = {"nominal": 1.0, "scale099": 0.99, "scale101": 1.01}[variant]
    source = Usd.Stage.Open(str(stock.asset_path(variant)))
    original = source.Flatten()
    target = output / model.asset_path(variant).name
    if target.exists():
        raise FileExistsError("Use a fresh output directory: " + str(target))
    stage = Usd.Stage.CreateNew(str(target))
    stage.GetRootLayer().TransferContent(original)
    stage.RemovePrim("/HiWonder/moving_jaw")
    stage.RemovePrim("/HiWonder/gripper_body")
    stage.RemovePrim("/HiWonder/joints/gripper")
    base_origin = np.asarray(model.links["base"]["origin_mm"])
    rotation = np.asarray(model.mount["rotation"])
    translation = np.asarray(model.mount["translation_cad_mm"])
    bounds = {name: [] for name in model.protected_links}
    for name in model.protected_links:
        body = UsdGeom.Xform.Define(stage, "/HiWonder/" + name)
        body.AddTranslateOp().Set(
            Gf.Vec3d(*((np.asarray(model.links[name]["origin_mm"]) - base_origin) * 0.001 * scale))
        )
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    plastic = material(stage, "plastic", (0.14, 0.20, 0.25), 0.55)
    liner = material(stage, "liner", (0.04, 0.04, 0.045), 0.85)
    case = material(stage, "case", (0.05, 0.07, 0.09), 0.4)
    metal = material(stage, "metal", (0.56, 0.58, 0.62), 0.23, 0.8)
    source_parts = []
    for index, part in enumerate(metadata["parts"]):
        name = part["body"]
        points = np.asarray(archive[f"v{index}"], float)
        faces = np.asarray(archive[f"f{index}"], np.int32)
        local = (points @ rotation.T + translation - np.asarray(model.links[name]["origin_mm"])) * (0.001 * scale)
        precision_key = f"{variant}_v{index}"
        if precision_key in precision:
            local = np.asarray(precision[precision_key], np.float32)
            faces = np.asarray(precision[f"{variant}_f{index}"], np.int32)
        bounds[name].append(local)
        visual = mesh(stage, f"/HiWonder/{name}/visuals/eka_{index}", local, faces)
        is_hardware = part.get("is_hardware", False)
        internal_spacer = "Spacer" in part["label"]
        shade = (
            case
            if "Motor" in part["label"]
            else (metal if is_hardware else (liner if "Liner" in part["label"] else plastic))
        )
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(shade)
        collision = mesh(stage, f"/HiWonder/{name}/collisions/eka_{index}", local, faces)
        collision.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(
            "convexHull" if is_hardware or internal_spacer else "convexDecomposition"
        )
        if not is_hardware and not internal_spacer and name.endswith("_tip"):
            collision.GetPrim().AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
            collision.GetPrim().CreateAttribute(
                "physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int
            ).Set(64)
        source_parts.append(
            dict(
                label=part["label"],
                body=name,
                visual=str(visual.GetPath()),
                collision=str(collision.GetPath()),
                triangles=len(faces),
                source_brep_valid=part.get("source_brep_valid"),
                source_mesh_closed=part.get("mesh_closed"),
                source_kind="reference_hardware" if is_hardware else "supplied_printable_STEP",
                mesh_repair=part.get("repair"),
                internal_spacer_proxy=internal_spacer,
            )
        )
    # Retain the original wrist horn and camera assembly. The camera optical
    # pose is unchanged; its attachment to the new palm is not a measured mount.
    for branch, ids in (
        ("visuals", (200, 252, 253, 254, 255, 256, 257, 258, 261)),
        ("collisions", (200, 252, 254, 261)),
    ):
        for index in ids:
            path = f"/HiWonder/gripper_body/{branch}/part_{index}"
            if source.GetPrimAtPath(path):
                UsdGeom.Xform.Define(stage, path.rsplit("/", 1)[0])
                Sdf.CopySpec(original, path, stage.GetRootLayer(), path)
                inherited = UsdGeom.Mesh(stage.GetPrimAtPath(path))
                bounds["gripper_body"].append(np.asarray(inherited.GetPointsAttr().Get()))
    camera = "/HiWonder/gripper_body/camera_mount"
    Sdf.CopySpec(original, camera, stage.GetRootLayer(), camera)
    report = {}
    for name, chunks in bounds.items():
        points = np.concatenate(chunks)
        lower, upper = points.min(0), points.max(0)
        size, center = upper - lower, (upper + lower) / 2
        mass_value = model.links[name]["mass_kg"] * scale**3
        inertia = (
            mass_value
            / 12
            * np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2])
        )
        api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/HiWonder/" + name))
        api.CreateMassAttr(mass_value)
        api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        report[name] = dict(
            mass_kg=mass_value,
            center_of_mass_m=center.tolist(),
            diagonal_inertia_kg_m2=inertia.tolist(),
            bounds_local_m=[lower.tolist(), upper.tolist()],
            qualification="estimated_mass_and_box_inertia",
        )
    for data in model.joints[5:]:
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/HiWonder/joints/" + data["name"])
        joint.CreateBody0Rel().SetTargets(["/HiWonder/" + data["parent"]])
        joint.CreateBody1Rel().SetTargets(["/HiWonder/" + data["child"]])
        parent, child = [np.asarray(model.links[data[k]]["origin_mm"]) for k in ("parent", "child")]
        joint.CreateLocalPos0Attr(Gf.Vec3f(*((child - parent) * (0.001 * scale))))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        quat = Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*data["axis"])).GetQuat()
        quat = Gf.Quatf(quat.GetReal(), Gf.Vec3f(*quat.GetImaginary()))
        joint.CreateLocalRot0Attr(quat)
        joint.CreateLocalRot1Attr(quat)
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(data["limits_deg"][0])
        joint.CreateUpperLimitAttr(data["limits_deg"][1])
        joint.CreateCollisionEnabledAttr(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.5 * np.pi / 180)
        drive.CreateDampingAttr(0.03 * np.pi / 180)
        drive.CreateMaxForceAttr(0.12)
        drive.CreateTargetPositionAttr(0)
    tool = UsdGeom.Xform.Define(stage, "/HiWonder/gripper_body/tool_center")
    tool.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(model.tool_center_m) * scale)))
    tool.GetPrim().SetCustomDataByKey(
        "hiwonder:frameMeaning", "zero_pose_reference; runtime_TCP_is_actual_tip_pad_midpoint"
    )
    for name, offset in model.finger_pad_offsets_m.items():
        pad = UsdGeom.Xform.Define(stage, f"/HiWonder/{name}/contact_pad")
        pad.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(offset) * scale)))
    root = stage.GetPrimAtPath("/HiWonder")
    root.SetCustomDataByKey("hiwonder:robotModel", "eka_v6")
    root.SetCustomDataByKey("hiwonder:sourceStepSha256", metadata["source_sha256"])
    stage.GetRootLayer().Save()
    return dict(file=target.name, scale=scale, sha256=digest(target), links=report, parts=source_parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=get_robot_model("eka_v6").asset_directory)
    parser.add_argument(
        "--geometry", type=Path, default=get_robot_model("eka_v6").asset_directory / "source_geometry.npz"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a fresh output directory; existing manifest will not be replaced")
    precision_file = args.geometry.with_name("precision_caps.npz")
    if not precision_file.is_file():
        raise FileNotFoundError("Repaired float-precision cap mesh archive is required: " + str(precision_file))
    with (
        np.load(args.geometry, allow_pickle=False) as archive,
        np.load(precision_file, allow_pickle=False) as precision,
    ):
        metadata = json.loads(archive["metadata"].tobytes())
        artifacts = [
            build_variant(name, args.output, archive, metadata, precision)
            for name in ("nominal", "scale099", "scale101")
        ]
    manifest = dict(
        name="HiWonder_SOARM101_Eka_XL330_V6",
        robot_model="eka_v6",
        units="m",
        source=dict(
            filename="Eka_XL330_SO100_V6_Distal_Assembly_Printable_Parts.step",
            sha256=metadata["source_sha256"],
            printable_solids=15,
            raw_invalid_brep_parts=[
                p["label"] for p in metadata["parts"] if not p.get("is_hardware") and not p["source_brep_valid"]
            ],
            unclosed_printable_mesh_parts=[
                p["label"] for p in metadata["parts"] if not p.get("is_hardware") and not p["mesh_closed"]
            ],
        ),
        artifacts=artifacts,
        joint_order=list(get_robot_model("eka_v6").joint_names),
        mount=get_robot_model("eka_v6").mount,
        mesh_repairs=[dict(label=p["label"], **p["repair"]) for p in metadata["parts"] if p.get("repair")],
        float_precision_repair_sha256=digest(precision_file),
        qualification=(
            "raw_source_caps_have_BRep_defects; user_authorized_derived_mesh_patch; "
            "collision_contact_qualification_is_separate"
        ),
        camera_qualification="retained_original_wrist_relative_optics_and_hardware; new_attachment_not_measured",
        dynamics_qualification=(
            "estimated_mass_inertia_joint_travel_and_XL330_continuous_effort; no_hardware_calibration"
        ),
        generator_sha256=digest(Path(__file__)),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="SAVED_EKA_USD_CANDIDATES",
                files=[a["file"] for a in artifacts],
                raw_invalid_brep_parts=manifest["source"]["raw_invalid_brep_parts"],
            )
        )
    )


if __name__ == "__main__":
    main()
