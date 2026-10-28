# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reproduce the derived cap boundary repair without overwriting supplied assets.

Requires the optional ``pymeshfix==0.18.1`` tool dependency. The original STEP,
unpatched cap tessellation, and exported source geometry remain unchanged. Output
is a new geometry archive and body-local float32 cap archive for the USD builder.
This is a faceted repair, not reconstructed analytic or parametric CAD.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from isaaclab_hiwonder.assets.robot_models import get_robot_model


def boundary_fill(vertices, faces, *, refine):
    import pymeshfix

    repair = pymeshfix.PyTMesh()
    repair.load_array(np.asarray(vertices, np.float64), np.asarray(faces, np.int32))
    repair.fill_small_boundaries(0, refine=refine)
    return repair.return_arrays()


def assert_contact_preserved(original, repaired, *, sign, transform):
    centers = original.triangles_center
    contact = (
        (original.face_normals[:, 0] * sign > 0.5) & (centers[:, 1] < -58) & (centers[:, 2] > 0) & (centers[:, 2] < 21)
    )
    if not contact.any():
        raise ValueError("Source has no expected contact surface")
    expected = transform(original.vertices)
    distance, mapping = cKDTree(repaired.vertices).query(expected)
    if distance[np.unique(original.faces[contact])].max() > 1e-8:
        raise ValueError("Repair moved a gripping-surface vertex")
    actual = {tuple(sorted(map(int, face))) for face in repaired.faces}
    if any(tuple(sorted(map(int, mapping[face]))) not in actual for face in original.faces[contact]):
        raise ValueError("Repair replaced an original gripping-surface triangle")


def main():
    model = get_robot_model("eka_v6")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=model.asset_directory / "source_geometry.npz")
    parser.add_argument("--original-caps", type=Path, default=model.asset_directory / "original_cap_meshes.npz")
    parser.add_argument("--output", type=Path, required=True, help="New directory; must not already exist")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rotation = np.asarray(model.mount["rotation"])
    translation = np.asarray(model.mount["translation_cad_mm"])
    with np.load(args.geometry, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(arrays["metadata"].tobytes())
    precision, report = {}, []
    with np.load(args.original_caps, allow_pickle=False) as archive:
        for index, body, sign in ((6, "eka_right_tip", -1), (13, "eka_left_tip", 1)):
            raw = trimesh.Trimesh(archive[f"v{index}"], archive[f"f{index}"], process=False)
            vertices, faces = boundary_fill(raw.vertices, raw.faces, refine=True)
            fixed = trimesh.Trimesh(vertices, faces, process=False)
            if not (fixed.is_watertight and fixed.is_winding_consistent and fixed.volume > 0):
                raise ValueError("Source boundary fill failed")
            # Strictly preserve every source triangle before float conversion.
            distance, _ = cKDTree(vertices).query(raw.vertices)

            # MeshFix reorders indices and the STEP tessellation has coincident
            # seam vertices. Compare triangle geometry rather than ambiguous IDs.
            def triangle_key(points):
                return tuple(sorted(tuple(point) for point in np.round(points, 10)))

            actual = {triangle_key(vertices[face]) for face in faces}
            if distance.max() > 1e-10 or any(triangle_key(raw.vertices[face]) not in actual for face in raw.faces):
                raise ValueError("Boundary fill modified original source geometry")
            arrays[f"v{index}"], arrays[f"f{index}"] = vertices, faces.astype(np.int32)
            origin = np.asarray(model.links[body]["origin_mm"])
            for variant, scale in (("nominal", 1.0), ("scale099", 0.99), ("scale101", 1.01)):

                def local(points):
                    return ((points @ rotation.T + translation - origin) * (0.001 * scale)).astype(np.float32)

                candidate = trimesh.Trimesh(local(vertices), faces, process=True)
                candidate.update_faces(candidate.nondegenerate_faces())
                candidate.update_faces(candidate.unique_faces())
                candidate.remove_unreferenced_vertices()
                v, f = boundary_fill(candidate.vertices * 1000, candidate.faces, refine=False)
                candidate = trimesh.Trimesh((v / 1000).astype(np.float32), f, process=True)
                components = candidate.split(only_watertight=False)
                keep = [part for part in components if abs(part.volume) > 1e-14]
                if len(keep) != 1:
                    raise ValueError("Repair produced multiple nontrivial components")
                candidate = keep[0]
                if not (candidate.is_watertight and candidate.is_winding_consistent and candidate.volume > 0):
                    raise ValueError("Cap is not closed at float32 simulation precision")
                assert_contact_preserved(raw, candidate, sign=sign, transform=local)
                precision[f"{variant}_v{index}"] = candidate.vertices.astype(np.float32)
                precision[f"{variant}_f{index}"] = candidate.faces.astype(np.int32)
                report.append(
                    dict(
                        variant=variant,
                        index=index,
                        vertices=len(candidate.vertices),
                        faces=len(candidate.faces),
                        volume_m3=candidate.volume,
                        contact_triangles_preserved=True,
                        closed=True,
                        discarded_sliver_volume_m3=[p.volume for p in components if abs(p.volume) <= 1e-14],
                    )
                )
    arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode(), np.uint8)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "source_geometry.npz", **arrays)
    np.savez_compressed(args.output / "precision_caps.npz", **precision)
    (args.output / "repair-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="PATCHED_CAPS", verified_variants=len(report), output=str(args.output))))


if __name__ == "__main__":
    main()
