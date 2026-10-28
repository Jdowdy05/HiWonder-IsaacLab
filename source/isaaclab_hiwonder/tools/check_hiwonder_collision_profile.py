# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Check immutable hull certificates and sample actual Newton self contacts.

This is an isolated collision query, without a room, cameras, SDFs or solver.
The original reference hulls and the manipulation profile use identical query
settings. Run inside an external process memory limit; GPU runs are serial.
"""

import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp

from pxr import Usd, UsdGeom

from isaaclab_hiwonder.assets.collision_geometry import certify_hull_cover
from isaaclab_hiwonder.assets.robot_models import get_robot_model
from isaaclab_hiwonder.sim.robot_collisions import (
    apply_robot_collision_profile,
    load_collision_profile,
    restore_profile_hulls,
)
from isaaclab_hiwonder.tasks.api import rotation_matrix


def points(part):
    transform = np.asarray(part["transform"])
    return (np.asarray(part["vertices"]) * part["scale"]) @ rotation_matrix(transform[3:]).T + transform[:3]


def reference_and_certificate(data, variant, model="stock"):
    selected_model = get_robot_model(model)
    if data.get("robot_model", "stock") != selected_model.name:
        raise ValueError("Collision profile belongs to a different robot model")
    if data.get("protected_links") != list(selected_model.protected_links):
        raise ValueError("Protected hand links differ from the selected robot model")
    row = data["variants"][variant]
    if row["asset_file"] != selected_model.asset_path(variant).name:
        raise ValueError("Collision source asset differs from the selected robot variant")
    with gzip.open(selected_model.asset_directory / row["reference_file"], "rb") as stream:
        reference = json.load(stream)["payload"]
    protected = set(selected_model.protected_links)
    groups = {group["link"]: group for group in row["groups"]}
    if len(groups) != len(row["groups"]) or not protected <= groups.keys():
        raise ValueError("Protected hand groups are missing or duplicated")
    for link, group in groups.items():
        if group["protected"] != (link in protected):
            raise ValueError("Protected hand group is incorrectly classified")
        if link in protected:
            complete = {
                (g, i)
                for g, source in enumerate(reference["groups"])
                for i, part in enumerate(source["parts"])
                if part["body"].strip("/") == link
            }
            refs = [tuple(pair) for pair in group["references"]]
            if not complete or len(refs) != len(complete) or set(refs) != complete:
                raise ValueError("Protected hand reference hulls are missing or duplicated")
    original = copy.deepcopy(data)
    original_row = original["variants"][variant]
    original_row["shapes"] = []
    checks = []
    for group in row["groups"]:
        sources = [reference["groups"][g]["parts"][i] for g, i in group["references"]]
        selected = [row["shapes"][i] for i in group["shape_indices"]]
        cover = dict(
            witnesses=group["witnesses"],
            parts=[
                dict(reference_index=p["reference_index"], unchanged=p["kind"] == "reference", vertices=points(p))
                for p in selected
            ],
        )
        checks.append(
            dict(
                link=group["link"],
                protected=group["protected"],
                **certify_hull_cover(
                    [points(p) for p in sources], cover, max_surface_loss=data["maximum_body_surface_loss_m"]
                ),
            )
        )
        if group["protected"]:
            for part, source in zip(selected, sources, strict=True):
                for name in ("vertices", "triangles", "transform", "scale"):
                    if part[name] != source[name]:
                        raise ValueError("Protected gripper data differs from immutable reference")
        for i, source in enumerate(sources):
            original_row["shapes"].append(
                dict(
                    **{key: source[key] for key in ("vertices", "triangles", "transform", "scale")},
                    kind="reference",
                    link=group["link"],
                    path=f"{group['link']}/manipulation_collisions/reference/hull_{i}",
                )
            )
    return original, checks


def query(data, variant, poses, device, robot_model="stock", contact_capacity=2097152):
    selected_model = get_robot_model(robot_model)
    row = data["variants"][variant]
    source = selected_model.asset_path(variant)
    if row["asset_file"] != source.name or data.get("robot_model", "stock") != selected_model.name:
        raise ValueError("Collision query source disagrees with the selected robot model")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = stage.DefinePrim("/Robot", "Xform")
    root.GetReferences().AddReference(str(source))
    root.SetCustomDataByKey("hiwonder:sourceAsset", str(source))
    started = time.perf_counter()
    apply_robot_collision_profile(
        stage, "/Robot", native=True, self_collisions=True, profile=data, model=selected_model
    )
    builder = newton.ModelBuilder()
    builder.add_usd(stage)
    restore_profile_hulls(builder, stage, "/Robot")
    mesh_bytes = sum(
        np.asarray(s.vertices).nbytes + np.asarray(s.indices).nbytes for s in builder.shape_source if s is not None
    )
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=contact_capacity, broad_phase="sap")
    contacts = pipeline.contacts()
    distance = wp.empty(contacts.rigid_contact_max, dtype=wp.float32, device=device)
    shape_bodies = model.shape_body.numpy()
    names = [label.rsplit("/", 1)[-1] for label in builder.body_label]
    starts = {label.rsplit("/", 1)[-1]: builder.joint_q_start[i] for i, label in enumerate(builder.joint_label)}
    setup_seconds = time.perf_counter() - started
    records, elapsed, maximum_contacts = [], [], 0
    for pose_index, pose in enumerate(poses):
        q = np.asarray(builder.joint_q, dtype=np.float32).copy()
        for name, value in zip(selected_model.joint_names, pose, strict=True):
            q[starts[name]] = value
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.synchronize_device(device)
        started = time.perf_counter()
        pipeline.collide(state, contacts)
        wp.synchronize_device(device)
        elapsed.append(time.perf_counter() - started)
        count = int(contacts.rigid_contact_count.numpy()[0])
        maximum_contacts = max(maximum_contacts, count)
        if count >= contacts.rigid_contact_max:
            raise RuntimeError(
                f"Collision query contact buffer saturated at pose {pose_index}: "
                f"count={count}, capacity={contacts.rigid_contact_max}, q={np.asarray(pose).tolist()}"
            )
        newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
        d = distance[:count].numpy()
        if not np.isfinite(d).all() or not np.isfinite(state.body_q.numpy()).all():
            raise RuntimeError("Collision query produced nonfinite geometry")
        a, b = contacts.rigid_contact_shape0[:count].numpy(), contacts.rigid_contact_shape1[:count].numpy()
        pairs = set()
        for i in np.flatnonzero(d < -0.000002):
            pair = tuple(sorted((names[shape_bodies[a[i]]], names[shape_bodies[b[i]]])))
            pairs.add(pair)
        records.append(pairs)
    return records, dict(
        shapes=builder.shape_count,
        mesh_source_bytes=mesh_bytes,
        setup_seconds=setup_seconds,
        maximum_contacts=maximum_contacts,
        median_query_ms=float(np.median(elapsed[1:]) * 1000),
        p95_query_ms=float(np.percentile(elapsed[1:], 95) * 1000),
    )


def sample_joint_poses(model="stock", random_poses=128):
    """Preserve the zero/single-joint sweep protocol with the selected joint dimension."""
    selected = get_robot_model(model)
    if random_poses < 0:
        raise ValueError("Need a nonnegative random pose count")
    limits = np.deg2rad([joint["limits_deg"] for joint in selected.joints])
    poses = [np.zeros(selected.num_joints)]
    for joint in range(selected.num_joints):
        for angle in np.linspace(*limits[joint], 21):
            pose = np.zeros(selected.num_joints)
            pose[joint] = angle
            poses.append(pose)
    poses.extend(np.random.default_rng(7529).uniform(limits[:, 0], limits[:, 1], (random_poses, selected.num_joints)))
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-poses", type=int, default=128)
    parser.add_argument(
        "--contact-capacity",
        type=int,
        default=2097152,
        help="Diagnostic query buffer only; deep teleported self-intersections may need more capacity",
    )
    parser.add_argument("--robot-model", choices=("stock", "eka_v6"), default="stock")
    parser.add_argument("--variant", choices=("nominal", "scale099", "scale101", "all"), default="all")
    args = parser.parse_args()
    if args.random_poses < 0 or args.contact_capacity < 1 or args.output.exists():
        raise ValueError("Need nonnegative sample count and a fresh output path")
    wp.set_device(args.device)
    model = get_robot_model(args.robot_model)
    poses = sample_joint_poses(model, args.random_poses)
    data = load_collision_profile(model.collision_profile_path, model=model)
    result = dict(
        status="PASS_SAMPLED_COLLISION_PROFILE",
        device=args.device,
        robot_model=model.name,
        joint_names=model.joint_names,
        protected_links=model.protected_links,
        profile_sha256=data["sha256"],
        poses_per_variant=len(poses),
        contact_capacity=args.contact_capacity,
        penetration_threshold_m=0.000002,
        variants={},
        qualification="geometric subsets plus sampled rigid contacts; no solver or SDF qualification",
    )
    variants = data["variants"] if args.variant == "all" else (args.variant,)
    for variant in variants:
        original, proof = reference_and_certificate(data, variant, model=model)
        before, before_metrics = query(
            original, variant, poses, args.device, robot_model=model, contact_capacity=args.contact_capacity
        )
        after, after_metrics = query(
            data, variant, poses, args.device, robot_model=model, contact_capacity=args.contact_capacity
        )
        additions = [
            dict(pose=i, joint_positions_rad=poses[i].tolist(), pairs=sorted(a - b))
            for i, (a, b) in enumerate(zip(after, before, strict=True))
            if a - b
        ]
        result["variants"][variant] = dict(
            certificate=proof,
            before=before_metrics,
            after=after_metrics,
            new_penetrating_body_pairs=additions,
            baseline_poses_with_penetration=sum(bool(p) for p in before),
        )
        print(
            json.dumps(dict(variant=variant, before=before_metrics, after=after_metrics, additions=len(additions))),
            flush=True,
        )
        if additions:
            result["status"] = "FAIL_NEW_SAMPLED_CONTACT_PAIRS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS_SAMPLED_COLLISION_PROFILE":
        raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()
