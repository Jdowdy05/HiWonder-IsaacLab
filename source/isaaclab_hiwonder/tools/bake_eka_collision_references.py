# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CPU bake: reuse exact arm/camera hulls, cook new printed hand components."""

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import newton
import numpy as np
import warp as wp

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_hiwonder.assets.robot_models import get_robot_model
from isaaclab_hiwonder.sim.collision_cache import _body_inertials, _identity, _read
from isaaclab_hiwonder.sim.robot_collisions import apply_robot_collision_profile, restore_profile_hulls
from isaaclab_hiwonder.tasks.api import rotation_matrix

OUT = None
MODEL = get_robot_model("eka_v6")
STOCK = get_robot_model("stock")


def read_stock(variant):
    path = STOCK.asset_directory / f"collision_reference_{variant}.json.gz"
    payload = _read(path, _identity(STOCK.asset_path(variant)))
    return [
        g
        for g in payload["groups"]
        if g["original"].split("/")[1] in ("base", "shoulder", "upper_arm", "forearm", "wrist")
        or g["original"].endswith("/part_261")
    ]


def preflight(asset_directory):
    """Validate every candidate and inherited cache before cooking or writing anything."""
    registered = get_robot_model("eka_v6")
    directory = Path(asset_directory).resolve()
    model = replace(
        registered,
        asset_directory=directory,
        manifest_path=directory / "manifest.json",
        collision_profile_path=directory / "manipulation_collisions.json",
    )
    outputs = [directory / f"collision_reference_{v}.json.gz" for v in model.geometry_variants]
    outputs.append(directory / "reference-bake-receipt.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refuse to replace an existing reference or bake receipt")
    pinned = json.loads(registered.manifest_path.read_text())
    manifest = json.loads(model.manifest_path.read_text())
    source_hash = pinned["source"]["sha256"]
    if (
        manifest.get("robot_model") != registered.name
        or manifest.get("source", {}).get("sha256") != source_hash
        or hashlib.sha256((directory / "source.step").read_bytes()).hexdigest() != source_hash
    ):
        raise ValueError("Candidate model/source STEP differs from the pinned registered source")
    for filename in ("source_geometry.npz", "precision_caps.npz"):
        if (
            hashlib.sha256((directory / filename).read_bytes()).digest()
            != hashlib.sha256((registered.asset_directory / filename).read_bytes()).digest()
        ):
            raise ValueError("Candidate source mesh archive differs from registered source: " + filename)
    artifacts = {row["file"]: row for row in manifest["artifacts"]}
    if len(manifest["artifacts"]) != 3 or set(artifacts) != {model.asset_path(v).name for v in model.geometry_variants}:
        raise ValueError("Candidate manifest must identify exactly the three registered USD variants")
    pinned_artifacts = {row["file"]: row for row in pinned["artifacts"]}
    validator_spec = importlib.util.spec_from_file_location(
        "eka_bake_validator", Path(__file__).with_name("validate_hiwonder_asset.py")
    )
    validator = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator)
    expected, inherited = {}, {}
    for variant in model.geometry_variants:
        source = model.asset_path(variant)
        stage = Usd.Stage.Open(str(source))
        root = stage.GetDefaultPrim() if stage else None
        if (
            not root
            or root.GetCustomDataByKey("hiwonder:robotModel") != model.name
            or root.GetCustomDataByKey("hiwonder:sourceStepSha256") != source_hash
        ):
            raise ValueError("Candidate USD root model/source identity is incorrect: " + variant)
        validator.validate_asset(source, model.manifest_path, robot_model=model)
        inherited[variant] = read_stock(variant)
        paths = {
            part["collision"].removeprefix("/HiWonder")
            for part in artifacts[source.name]["parts"]
            if part["source_kind"] != "reference_hardware" and not part["internal_spacer_proxy"]
        }
        pinned_paths = {
            part["collision"].removeprefix("/HiWonder")
            for part in pinned_artifacts[source.name]["parts"]
            if part["source_kind"] != "reference_hardware" and not part["internal_spacer_proxy"]
        }
        if not paths or paths != pinned_paths or {path.split("/")[1] for path in paths} != set(model.protected_links):
            raise ValueError("Candidate decomposition manifest omits registered hand components")
        actual = {
            str(prim.GetPath()).removeprefix("/HiWonder")
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexDecomposition"
        }
        actual -= {group["original"] for group in inherited[variant]}
        if actual != paths:
            raise ValueError("Enabled hand decomposition colliders differ from the complete manifest: " + variant)
        expected[variant] = paths
    return model, expected, inherited


def validate_cooked_groups(groups, expected):
    """Reject empty, duplicate or partial hand cook results before the first output."""
    paths = [group["original"] for group in groups]
    if not groups or len(paths) != len(expected) or set(paths) != set(expected):
        raise RuntimeError("Cooked hand component groups do not match the preflight inventory")
    for group in groups:
        body = "/" + group["original"].split("/")[1]
        if not group["parts"] or any(part["body"] != body for part in group["parts"]):
            raise RuntimeError("Cooked hand group has empty hulls or the wrong body owner")


def make_stage(variant):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = stage.DefinePrim("/Robot", "Xform")
    root.GetReferences().AddReference(str(MODEL.asset_path(variant)))
    root.SetCustomDataByKey("hiwonder:sourceAsset", str(MODEL.asset_path(variant)))
    return stage


def write_reference(variant, groups, stage, builder):
    path = MODEL.asset_directory / f"collision_reference_{variant}.json.gz"
    if path.exists():
        raise FileExistsError(path)
    authored = _body_inertials(stage, "/Robot")
    physical = {}
    for relative in authored:
        prim = stage.GetPrimAtPath("/Robot" + relative)
        mass = UsdPhysics.MassAPI(prim)
        physical[relative] = dict(
            mass=mass.GetMassAttr().Get(),
            inertia=np.diag(mass.GetDiagonalInertiaAttr().Get()).tolist(),
            center_of_mass=list(mass.GetCenterOfMassAttr().Get()),
        )
    payload = dict(
        identity=_identity(MODEL.asset_path(variant), builder),
        groups=groups,
        authored_inertials=authored,
        builder_inertials=physical,
        source_kind="matched_stock_arm_cache_plus_nominal_hand_convex_cook",
        hand_variant_rule="uniform_affine_scale_of_nominal_reference",
        source_step_sha256=hashlib.sha256((MODEL.asset_directory / "source.step").read_bytes()).hexdigest(),
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    envelope = dict(payload=payload, payload_sha256=hashlib.sha256(encoded).hexdigest())
    with gzip.open(path, "wb") as stream:
        stream.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode())
    print("SAVED_REFERENCE", variant, sum(len(g["parts"]) for g in groups), flush=True)


def main():
    global MODEL, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-directory",
        type=Path,
        default=MODEL.asset_directory,
        help="Fresh Eka candidate with manifest, three USD variants, source.step and registered mesh archives",
    )
    args = parser.parse_args()
    MODEL, expected, stock_groups = preflight(args.asset_directory)
    OUT = MODEL.asset_directory
    wp.set_device("cpu")
    stage = make_stage("nominal")
    inherited = copy.deepcopy(stock_groups["nominal"])
    shapes = []
    for g, group in enumerate(inherited):
        for i, part in enumerate(group["parts"]):
            link = part["body"].strip("/")
            shapes.append(
                dict(
                    link=link,
                    path=f"{link}/manipulation_collisions/reference_{g}/shape_{i}",
                    **{key: part[key] for key in ("vertices", "triangles", "transform", "scale")},
                )
            )
    partial = dict(
        robot_model="eka_v6",
        sha256="reference-arm-injection",
        maximum_body_surface_loss_m=0.002,
        variants={
            "nominal": dict(
                asset_sha256=hashlib.sha256(MODEL.asset_path().read_bytes()).hexdigest(),
                structural_paths=[g["original"] for g in inherited],
                shapes=shapes,
                reference_hulls=len(shapes),
            )
        },
    )
    apply_robot_collision_profile(stage, "/Robot", native=True, profile=partial, model=MODEL)
    print("IMPORT_BEGIN", "exact inherited hulls", len(shapes), flush=True)
    builder = newton.ModelBuilder()
    builder.add_usd(stage)
    print("IMPORT_COMPLETE", builder.shape_count, flush=True)
    receipt = restore_profile_hulls(builder, stage, "/Robot")
    print("INHERITED_REIMPORT", receipt["mesh_sources_restored_before_sdf"], flush=True)
    new_groups = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot")):
        path = str(prim.GetPath())
        if "/collisions/eka_" not in path or not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "convexDecomposition":
            continue
        labels = [
            i for i, label in enumerate(builder.shape_label) if label == path or label.startswith(path + "_convex_")
        ]
        if not labels:
            raise RuntimeError("Missing newly cooked component: " + path)
        parts = []
        for i in labels:
            mesh = builder.shape_source[i]
            if builder.shape_type[i] != newton.GeoType.CONVEX_MESH:
                raise RuntimeError("New component did not produce convex collision shapes")
            vertices = np.asarray(mesh.vertices, np.float32)
            transform, scale = np.asarray(builder.shape_transform[i]), np.asarray(builder.shape_scale[i])
            points = (vertices * scale) @ rotation_matrix(transform[3:]).T + transform[:3]
            parts.append(
                dict(
                    vertices=vertices.tolist(),
                    triangles=np.asarray(mesh.indices).reshape(-1, 3).tolist(),
                    transform=transform.tolist(),
                    scale=scale.tolist(),
                    flags=int(builder.shape_flags[i]),
                    body=builder.body_label[builder.shape_body[i]][len("/Robot") :],
                    bounds_body_local=[points.min(0).tolist(), points.max(0).tolist()],
                )
            )
        new_groups.append(dict(original=path[len("/Robot") :], parts=parts))
        print("HAND_COMPONENT", path, len(parts), flush=True)
    validate_cooked_groups(new_groups, expected["nominal"])
    write_reference("nominal", inherited + new_groups, make_stage("nominal"), builder)
    for variant, factor in (("scale099", 0.99), ("scale101", 1.01)):
        scaled = copy.deepcopy(new_groups)
        for group in scaled:
            for part in group["parts"]:
                part["vertices"] = (np.asarray(part["vertices"]) * factor).tolist()
                part["transform"][:3] = (np.asarray(part["transform"][:3]) * factor).tolist()
                part["bounds_body_local"] = (np.asarray(part["bounds_body_local"]) * factor).tolist()
        validate_cooked_groups(scaled, expected[variant])
        write_reference(variant, copy.deepcopy(stock_groups[variant]) + scaled, make_stage(variant), builder)
    (OUT / "reference-bake-receipt.json").write_text(
        json.dumps(
            dict(
                status="BAKED_NATIVE_REFERENCES",
                hand_components=len(new_groups),
                hand_hulls=sum(len(g["parts"]) for g in new_groups),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
