# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Build an inscribed body profile from immutable Newton reference hulls.

Run on CPU after preparing collision_reference_{variant}.json.gz alongside the
original three USD assets. Gripper links retain every original hull. Body pieces
are selected/reduced with complete-convex-solid subset and coverage checks.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from isaaclab_hiwonder.assets.collision_geometry import certify_hull_cover, select_collision_hulls
from isaaclab_hiwonder.assets.registration import ASSET_DIR
from isaaclab_hiwonder.tasks.api import rotation_matrix

PROTECTED = ("gripper_body", "moving_jaw")
VARIANTS = ("nominal", "scale099", "scale101")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def body_vertices(part):
    transform = np.asarray(part["transform"])
    return (np.asarray(part["vertices"]) * part["scale"]) @ rotation_matrix(transform[3:]).T + transform[:3]


def build_variant(variant, assets, loss):
    source = assets / ("hiwonder_soarm101" + ("" if variant == "nominal" else "_" + variant) + ".usdc")
    reference = assets / ("collision_reference_" + variant + ".json.gz")
    with gzip.open(reference, "rb") as stream:
        envelope = json.load(stream)
    payload = envelope["payload"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != envelope["payload_sha256"]:
        raise ValueError("Reference payload checksum differs")
    if payload["identity"]["source_robot_sha256"] != digest(source):
        raise ValueError("Reference belongs to another source asset")
    result = dict(
        asset_file=source.name,
        asset_sha256=digest(source),
        reference_file=reference.name,
        reference_sha256=digest(reference),
        structural_paths=[group["original"] for group in payload["groups"]],
        shapes=[],
        groups=[],
    )
    grouped = {}
    for source_group, group in enumerate(payload["groups"]):
        for source_index, part in enumerate(group["parts"]):
            link = part["body"].strip("/")
            grouped.setdefault(link, []).append((source_group, source_index, part))
    for group_index, (link, records) in enumerate(grouped.items()):
        original = [record[2] for record in records]
        proof = dict(
            source_group=group_index,
            link=link,
            references=[[record[0], record[1]] for record in records],
            reference_hulls=len(original),
            protected=link in PROTECTED,
        )
        if link in PROTECTED:
            selected = [dict(reference_index=i, unchanged=True) for i in range(len(original))]
            proof.update(witnesses=list(range(len(original))), maximum_body_loss_m=0.0)
        else:
            vertices = [body_vertices(part) for part in original]
            cover = select_collision_hulls(vertices, max_surface_loss=loss)
            checked = certify_hull_cover(vertices, cover, max_surface_loss=loss)
            selected = cover["parts"]
            proof.update(witnesses=cover["witnesses"].tolist(), **checked)
        proof["shape_indices"] = []
        for index, selected_part in enumerate(selected):
            ref = selected_part["reference_index"]
            if selected_part["unchanged"]:
                entry = {key: original[ref][key] for key in ("vertices", "triangles", "transform", "scale")}
                entry["kind"] = "reference"
            else:
                entry = dict(
                    vertices=selected_part["vertices"].tolist(),
                    triangles=selected_part["triangles"].tolist(),
                    transform=[0, 0, 0, 0, 0, 0, 1],
                    scale=[1, 1, 1],
                    kind="inscribed",
                )
            entry.update(
                link=link,
                group_index=group_index,
                reference_index=ref,
                path=f"{link}/manipulation_collisions/group_{group_index}/shape_{index}",
            )
            proof["shape_indices"].append(len(result["shapes"]))
            result["shapes"].append(entry)
        result["groups"].append(proof)
        print(
            json.dumps(
                dict(
                    variant=variant,
                    link=link,
                    before=len(original),
                    after=len(selected),
                    protected=proof["protected"],
                    bound_m=proof["maximum_body_loss_m"],
                )
            ),
            flush=True,
        )
    result["reference_hulls"] = sum(g["reference_hulls"] for g in result["groups"])
    result["profile_hulls"] = len(result["shapes"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--assets", type=Path, default=ASSET_DIR / "generated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-surface-loss-m", type=float, default=0.002)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing profile artifact")
    if not 0 < args.body_surface_loss_m <= 0.002:
        raise ValueError("Body loss must be positive and at most 2mm")
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    result = dict(
        schema_version=1,
        name="manipulation",
        protected_links=list(PROTECTED),
        maximum_body_surface_loss_m=args.body_surface_loss_m,
        variants={},
    )
    for variant in variants:
        result["variants"][variant] = build_variant(variant, args.assets, args.body_surface_loss_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    manifest = dict(
        schema_version=1,
        file=args.output.name,
        sha256=digest(args.output),
        generator_sha256=digest(Path(__file__)),
        qualification="geometric_subset_and_body_loss_verified_backend_checks_required",
    )
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
