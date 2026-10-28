# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit stock/Eka model ownership, independent of scene or physics imports.

Stock registration remains unchanged. Eka joint zeros reproduce the supplied
printable STEP, whose mounting holes match the original wrist horn. Travel,
motor gains, continuous effort and inertials are simulation estimates; this is
not a hardware calibration. Only the CAD mounting pattern is measured here.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import registration as stock


@dataclass(frozen=True)
class RobotModel:
    name: str
    joint_names: tuple[str, ...]
    joints: tuple[dict, ...]
    links: dict[str, dict]
    nominal_joint_positions_rad: tuple[float, ...]
    gripper_joint_indices: tuple[int, ...]
    gripper_open_positions_rad: tuple[float, ...]
    gripper_closed_positions_rad: tuple[float, ...]
    camera_path: str
    camera: dict
    tool_center_m: tuple[float, float, float]
    finger_pad_offsets_m: dict[str, tuple[float, float, float]]
    contact_bodies: dict[str, tuple[str, ...]]
    protected_links: tuple[str, ...]
    actuator_scales: dict[str, tuple[float, ...]]
    critic_width: int
    asset_directory: Path
    asset_prefix: str
    manifest_path: Path
    collision_profile_path: Path
    mount: dict

    @property
    def num_joints(self):
        return len(self.joint_names)

    @property
    def num_links(self):
        return len(self.links)

    @property
    def arm_joint_indices(self):
        return (0, 1, 2, 3, 4)

    @property
    def gripper_joint_names(self):
        return tuple(self.joint_names[i] for i in self.gripper_joint_indices)

    @property
    def geometry_variants(self):
        return ("nominal", "scale099", "scale101")

    def asset_path(self, variant="nominal"):
        if variant not in self.geometry_variants:
            raise ValueError(f"Unknown geometry variant {variant!r}")
        suffix = "" if variant == "nominal" else "_" + variant
        return self.asset_directory / (self.asset_prefix + suffix + ".usdc")


def _stock():
    tool = (np.array([-2, -181, 207]) - np.asarray(stock.LINKS["gripper_body"]["origin_mm"])) * 0.001
    return RobotModel(
        name="stock",
        joint_names=stock.JOINT_NAMES,
        joints=tuple(copy.deepcopy(stock.JOINTS)),
        links=copy.deepcopy(stock.LINKS),
        nominal_joint_positions_rad=stock.NOMINAL_JOINT_POSITIONS_RAD,
        gripper_joint_indices=(5,),
        gripper_open_positions_rad=(stock.GRIPPER_OPEN_RAD,),
        gripper_closed_positions_rad=(stock.GRIPPER_CLOSED_RAD,),
        camera_path=stock.CAMERA_PATH,
        camera=copy.deepcopy(stock.CAMERA),
        tool_center_m=tuple(tool),
        finger_pad_offsets_m={},
        contact_bodies={"left_finger": ("gripper_body",), "right_finger": ("moving_jaw",)},
        protected_links=("gripper_body", "moving_jaw"),
        actuator_scales={
            key: (1.0,) * 6
            for key in ("stiffness", "damping", "effort_limit", "velocity_limit", "friction", "armature")
        },
        critic_width=256,
        asset_directory=stock.ASSET_DIR / "generated",
        asset_prefix="hiwonder_soarm101",
        manifest_path=stock.MANIFEST_PATH,
        collision_profile_path=stock.ASSET_DIR / "generated/manipulation_collisions.json",
        mount={},
    )


def _eka():
    root = stock.ASSET_DIR / "eka_v6"
    c = math.sqrt(0.5)
    rotation = np.array([[1, 0, 0], [0, c, -c], [0, c, c]])
    # Exact circular edges measured in the supplied STEP and original 3DM horn.
    adapter_center = np.array([0.0, 42.195, 2.681])
    horn_center = np.array([0.4998776259297, -112.6950614604658, 274.5839957004861])
    translation = horn_center - rotation @ adapter_center
    a = 7 / math.sqrt(2)
    adapter_holes = np.array([[x, 42.195, 2.681 + z] for z in (-a, a) for x in (-a, a)])
    horn_holes = np.array(
        [
            [-4.449871166857875, -109.19506052391577, 271.08399476393606],
            [5.449626418717215, -109.19506052391588, 271.083994763936],
            [-4.449871166857847, -116.19506239701568, 278.0839966370361],
            [5.449626418717242, -116.19506239701579, 278.08399663703597],
        ]
    )
    base = np.asarray(stock.LINKS["base"]["origin_mm"])
    links = copy.deepcopy({k: v for k, v in stock.LINKS.items() if k not in ("gripper_body", "moving_jaw")})
    links["gripper_body"] = dict(origin_mm=list(stock.LINKS["gripper_body"]["origin_mm"]), mass_kg=0.0826, objects=[])
    source_pivots = {
        "eka_left_proximal": (-20, 0, 0),
        "eka_left_tip": (-20, -45, 0),
        "eka_right_proximal": (20, 0, 0),
        "eka_right_tip": (20, -45, 0),
    }
    for name, pivot in source_pivots.items():
        links[name] = dict(
            origin_mm=(rotation @ pivot + translation).tolist(),
            mass_kg=0.0266 if "proximal" in name else (0.0136 if "left" in name else 0.0143),
            objects=[],
        )
    joints = copy.deepcopy(stock.JOINTS[:5])
    axis = rotation[:, 2]
    names = list(stock.JOINT_NAMES[:5])
    for side, sign in (("left", 1), ("right", -1)):
        parent = "gripper_body"
        for segment, child, limits in (
            ("root", f"eka_{side}_proximal", [-25, 20]),
            ("tip", f"eka_{side}_tip", [0, 60]),
        ):
            name = f"{side}_finger_{segment}"
            names.append(name)
            joints.append(dict(name=name, parent=parent, child=child, axis=(axis * sign).tolist(), limits_deg=limits))
            parent = child
    # First intersections of the supplied nominal hand with gap-centered +/-X
    # rays at source Y=-68.5/Z=9 mm. These are CAD contact references, not tactile
    # calibration. The supplied file contains a right liner but no left liner.
    pads = {"eka_left_tip": (12.079368043906028, -23.5, 9), "eka_right_tip": (-12.282176744620114, -23.5, 9)}
    pads = {name: tuple(rotation @ value * 0.001) for name, value in pads.items()}
    palm_origin = np.asarray(links["gripper_body"]["origin_mm"])
    nominal_tcp = np.mean(
        [np.asarray(links[name]["origin_mm"]) + np.asarray(offset) * 1000 for name, offset in pads.items()], axis=0
    )
    hand = ("gripper_body", "eka_left_proximal", "eka_left_tip", "eka_right_proximal", "eka_right_tip")
    camera = copy.deepcopy(stock.CAMERA)
    camera["optical_transform_status"] = "retained_wrist_pose_pending_eka_mount_measurement"
    return RobotModel(
        name="eka_v6",
        joint_names=tuple(names),
        joints=tuple(joints),
        links=links,
        nominal_joint_positions_rad=(0, 0, 0, 0, 0, math.radians(-20), 0, math.radians(-20), 0),
        gripper_joint_indices=(5, 6, 7, 8),
        gripper_open_positions_rad=(math.radians(-20), 0, math.radians(-20), 0),
        gripper_closed_positions_rad=(0, math.radians(18), 0, math.radians(18)),
        camera_path=stock.CAMERA_PATH,
        camera=camera,
        tool_center_m=tuple((nominal_tcp - palm_origin) * 0.001),
        finger_pad_offsets_m=pads,
        contact_bodies={"left_finger": ("eka_left_tip",), "right_finger": ("eka_right_tip",), "gripper": hand},
        protected_links=hand,
        actuator_scales={
            key: (1.0,) * 5 + (scale,) * 4
            for key, scale in {
                "stiffness": 0.02,
                "damping": 0.02,
                "effort_limit": 0.12,
                "velocity_limit": 1.5,
                "friction": 0.02,
                "armature": 0.01,
            }.items()
        },
        critic_width=320,
        asset_directory=root,
        asset_prefix="hiwonder_eka_v6",
        manifest_path=root / "manifest.json",
        collision_profile_path=root / "manipulation_collisions.json",
        mount=dict(
            rotation=rotation.tolist(),
            translation_cad_mm=translation.tolist(),
            translation_support_mm=(translation - base).tolist(),
            adapter_holes_mm=adapter_holes.tolist(),
            horn_holes_support_mm=(horn_holes - base).tolist(),
            adapter_center_mm=adapter_center.tolist(),
            horn_center_cad_mm=horn_center.tolist(),
            source_step="source.step",
            qualification="CAD_hole_pattern_fit; physical_threads_clearances_and_camera_mount_unqualified",
        ),
    )


def get_robot_model(name="stock") -> RobotModel:
    """Return a fresh descriptor so one environment cannot mutate another model."""
    if isinstance(name, RobotModel):
        return name
    if name == "stock":
        return _stock()
    if name == "eka_v6":
        return _eka()
    raise ValueError(f"Unknown robot model {name!r}; choose stock or eka_v6")
