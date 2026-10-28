# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral task scene and measured-state contracts (SI, xyzw)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActorState:
    position: tuple | np.ndarray
    orientation: tuple | np.ndarray = (0.0, 0.0, 0.0, 1.0)
    linear_velocity: tuple | np.ndarray = (0.0, 0.0, 0.0)
    angular_velocity: tuple | np.ndarray = (0.0, 0.0, 0.0)


@dataclass
class ShapeSpec:
    """Primitive dimensions are full lengths; cylinder=(radius,height), sphere=(radius,)."""

    kind: str
    dimensions: tuple = ()
    position: tuple = (0.0, 0.0, 0.0)
    orientation: tuple = (0.0, 0.0, 0.0, 1.0)
    vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    collision: bool = True


@dataclass
class BodySpec:
    name: str
    shapes: list[ShapeSpec]
    mass: float = 0.03
    static: bool = False
    friction: float = 0.6
    restitution: float = 0.05
    color: tuple = (0.4, 0.6, 0.8)


@dataclass
class JointSpec:
    name: str
    kind: str
    parent: str
    child: str
    axis: tuple
    lower: float
    upper: float
    effort_limit: float
    position: tuple = (0.0, 0.0, 0.0)
    stiffness: float = 0.0
    damping: float = 0.01
    rest: float = 0.0
    parameters: dict = field(default_factory=dict)


@dataclass
class CableMaterialSpec:
    """Estimated continuum moduli, not per-joint spring stiffnesses.

    Native CableMaterialCfg consumes Pa; Newton discretizes EA/L and EI/L.
    Surface membrane/edge coefficients have different units.
    """

    stretch_modulus_pa: float = 1.0e8
    bend_modulus_pa: float = 1.0e6
    shear_modulus_pa: float = 1.0e8
    twist_modulus_pa: float = 1.0e6
    provenance: str = "estimated_not_measured"


@dataclass
class DeformableSpec:
    name: str
    kind: str
    vertices: np.ndarray
    edges: np.ndarray
    triangles: np.ndarray
    radius: float
    density: float
    stretch_stiffness: float
    bend_stiffness: float
    damping: float
    self_contact: bool = True
    attachments: list[dict] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    cable_material: CableMaterialSpec | None = None


@dataclass
class SceneState:
    """Measured simulator state, never commanded targets or synthetic success flags.

    Contacts are unordered body-name pairs generated from nonzero rigid contact
    forces. Native soft-contact candidates conservatively block gripper release;
    they are never used to establish rigid seating or support success.
    Deformable positions/velocities follow the exact DeformableSpec vertex ordering.
    joint_positions contains unwrapped positions in radians/meters. Adapters must not
    populate contact/constraint evidence from desired poses or commanded actions.
    """

    actors: dict[str, ActorState]
    joint_positions: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    contacts: set[tuple[str, str]] = field(default_factory=set)
    deformable_positions: dict[str, np.ndarray] = field(default_factory=dict)
    deformable_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    crease_rest_angles: dict[str, np.ndarray] = field(default_factory=dict)
    mechanisms: dict[str, float] = field(default_factory=dict)
    time: float = 0.0

    def touching(self, first: str, second: str) -> bool:
        return (first, second) in self.contacts or (second, first) in self.contacts


@dataclass
class TaskDefinition:
    name: str
    variant: str
    physics_family: str
    bodies: list[BodySpec]
    joints: list[JointSpec]
    deformables: list[DeformableSpec]
    initial: dict[str, ActorState]
    goal: np.ndarray
    parameters: dict
    horizon_seconds: float = 30.0
    hold_seconds: float = 0.5
    qualification: str = "geometry_and_algorithm_only"
    required_validation: tuple[str, ...] = ("contact stability", "robot reachability", "backend execution")


@dataclass
class TaskResult:
    reward: float
    success: bool
    failure: bool
    terminated: bool
    truncated: bool
    components: dict[str, float]
    metrics: dict[str, float]


def rotation_matrix(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Actor quaternion must be finite and nonzero")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
