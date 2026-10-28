# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent profiles. Distances are meters; angles radians unless named degrees.

Randomization ranges are engineering starting estimates, not measured HX-30HM or camera calibration.
"""

import json
import math
from dataclasses import dataclass, field, replace
from typing import Literal

from .assets.registration import MANIFEST_PATH
from .assets.robot_models import get_robot_model
from .rewards import RewardConfig


def resolve_backend(requested: str, physics_family: str) -> str:
    """Choose the task's implemented solver and reject an incompatible explicit request."""
    required = "physx" if physics_family in ("rigid", "articulated") else "newton_vbd"
    if requested not in ("auto", required):
        raise ValueError(f"Task physics family {physics_family} requires {required}; got {requested}")
    return required


@dataclass(frozen=True)
class CameraProfile:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    calibration_status: Literal["estimated", "calibrated"] = "estimated"
    provenance: str = "USB capture dimensions verified; optical parameters and mount require calibration"
    usb_identity: str | None = None
    optical_parent_link: str = "gripper_body"
    optical_transform_status: str = "CAD_mechanical_estimate"
    optical_translation_m: tuple[float, float, float] = (-0.00250012, -0.05209936, 0.04465591)
    optical_quaternion_xyzw: tuple[float, float, float, float] = (0.984807753, 0.0, 0.0, 0.173648178)
    # OpenCV optical frame: +x right, +y down, +z forward. Renderer conversion is explicit.
    optical_frame_convention: str = "opencv"
    renderer_frame_convention: str = "opengl"
    intrinsics: tuple[float, ...] = (554.2666667, 0.0, 320.0, 0.0, 554.2666667, 240.0, 0.0, 0.0, 1.0)
    distortion_model: str = "brown_conrady"
    distortion_coefficients: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    horizontal_fov_degrees: float = 60.0
    vertical_fov_degrees: float = 46.826
    sensor_aperture_mm: tuple[float, float] = (4.8, 3.6)
    focal_length_mm: float = 4.157
    f_stop: float | None = None
    focus_distance_m: float = 0.25
    clipping_range_m: tuple[float, float] = (0.01, 10.0)
    exposure_seconds: float = 0.008
    gain: float = 1.0
    white_balance_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    response_gamma: float = 2.2
    latency_seconds: float = 0.0
    frame_drop_probability: float = 0.0
    reprojection_error_px: float | None = None
    calibration_source: str | None = None

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("Camera image dimensions and fps must be positive")
        if len(self.intrinsics) != 9 or self.intrinsics[0] <= 0 or self.intrinsics[4] <= 0:
            raise ValueError("Camera intrinsics must be a 3x3 matrix with positive focal lengths")
        if self.calibration_status == "calibrated" and not self.calibration_source:
            raise ValueError("Calibrated optics require calibration_source provenance")
        if len(self.optical_quaternion_xyzw) != 4 or not math.isclose(
            sum(x * x for x in self.optical_quaternion_xyzw), 1.0, abs_tol=1e-5
        ):
            raise ValueError("Optical quaternion must be a normalized xyzw quaternion")
        for name, supported in (
            ("optical_parent_link", "gripper_body"),
            ("optical_frame_convention", "opencv"),
            ("renderer_frame_convention", "opengl"),
        ):
            if getattr(self, name) != supported:
                raise ValueError(f"{name} currently supports only {supported!r}")

    @property
    def fx(self) -> float:
        return self.intrinsics[0]

    @property
    def fy(self) -> float:
        return self.intrinsics[4]

    @property
    def cx(self) -> float:
        return self.intrinsics[2]

    @property
    def cy(self) -> float:
        return self.intrinsics[5]

    @property
    def filmback_width_mm(self) -> float:
        return self.sensor_aperture_mm[0]

    @property
    def distortion(self) -> tuple[float, ...]:
        return self.distortion_coefficients


@dataclass(frozen=True)
class RoomConfig:
    width_m: float = 3.6576
    depth_m: float = 3.6576
    height_m: float = 3.6576
    wall_thickness_m: float = 0.05
    ceiling: bool = False
    independent_materials: bool = True
    env_spacing_m: float = 4.0


@dataclass(frozen=True)
class MaterialConfig:
    families: tuple[str, ...] = ("matte", "wood", "plastic", "metal")
    color: tuple[float, float] = (0.05, 0.95)
    roughness: tuple[float, float] = (0.15, 0.95)
    metallic: tuple[float, float] = (0.0, 0.8)
    specular: tuple[float, float] = (0.1, 0.8)
    uv_scale: tuple[float, float] = (0.5, 4.0)
    friction: tuple[float, float] = (0.3, 1.0)
    restitution: tuple[float, float] = (0.0, 0.15)
    texture_ids: tuple[str, ...] = ("none", "wood_grain", "checker", "noise")


@dataclass(frozen=True)
class TableConfig:
    shapes: tuple[str, ...] = ("rectangle", "rounded_rectangle")
    width_m: tuple[float, float] = (0.8, 1.2)
    depth_m: tuple[float, float] = (0.65, 0.95)
    height_m: tuple[float, float] = (0.65, 0.85)
    thickness_m: tuple[float, float] = (0.025, 0.055)
    edge_radius_m: tuple[float, float] = (0.005, 0.025)
    material: MaterialConfig = field(default_factory=MaterialConfig)


@dataclass(frozen=True)
class LightingConfig:
    """Physical lighting priors, independent of camera/image color augmentation.

    Wrist-view spot radius is transverse to the beam [m]; oblique intersections
    naturally form elliptical footprints on the workspace plane. Emitter radius
    controls the wrist-view projector source size, independently of beam radius.
    """

    profile: Literal["stationary", "white_disco", "rgb_disco"] = "stationary"
    count: tuple[int, int] = (1, 4)
    max_lights: int = 4
    types: tuple[str, ...] = ("sphere", "spot")
    intensity: tuple[float, float] = (300.0, 1800.0)
    temperature_kelvin: tuple[float, float] = (3000.0, 8000.0)
    beam_angle_degrees: tuple[float, float] = (15.0, 70.0)
    angular_speed: tuple[float, float] = (0.3, 2.0)
    shadows_probability: float = 0.8
    backgrounds: tuple[str, ...] = ("none",)
    wall_material: MaterialConfig = field(default_factory=MaterialConfig)
    color_saturation: tuple[float, float] = (0.85, 0.85)
    saturation_period_s: float = 0.0
    motion_mode: Literal["room_sweep", "wrist_view"] = "room_sweep"
    spot_radius_m: tuple[float, float] = (0.025, 0.07)
    emitter_radius_m: float = 0.004
    edge_softness: tuple[float, float] = (0.15, 0.4)
    view_extent: tuple[float, float] = (0.25, 0.65)
    white_fill_intensity: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        for name, low, high, strict_low, strict_high in (
            ("intensity", 0, math.inf, False, False),
            ("temperature_kelvin", 0, math.inf, True, False),
            ("beam_angle_degrees", 0, 180, True, True),
            ("angular_speed", 0, math.inf, False, False),
            ("color_saturation", 0, 1, False, False),
            ("spot_radius_m", 0, math.inf, True, False),
            ("edge_softness", 0, 1, False, False),
            ("view_extent", 0, 0.9, False, False),
            ("white_fill_intensity", 0, math.inf, False, False),
        ):
            bounds = getattr(self, name)
            if (
                not isinstance(bounds, (tuple, list))
                or len(bounds) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]
                or bounds[0] < low
                or bounds[1] > high
                or (strict_low and bounds[0] == low)
                or (strict_high and bounds[1] == high)
            ):
                raise ValueError(f"{name} must be a finite ordered range in its physical domain")
        if (
            type(self.max_lights) is not int
            or self.max_lights < 0
            or not isinstance(self.count, (tuple, list))
            or len(self.count) != 2
            or any(type(value) is not int for value in self.count)
            or not 0 <= self.count[0] <= self.count[1] <= self.max_lights
        ):
            raise ValueError("Light count must be a nonnegative integer range within max_lights")
        if self.profile not in ("stationary", "white_disco", "rgb_disco"):
            raise ValueError("Invalid light profile")
        if self.motion_mode not in ("room_sweep", "wrist_view"):
            raise ValueError("Invalid light motion_mode")
        if (
            not isinstance(self.types, (tuple, list))
            or not self.types
            or any(kind not in ("sphere", "spot") for kind in self.types)
        ):
            raise ValueError("Light types must contain sphere or spot")
        if (
            not isinstance(self.backgrounds, (tuple, list))
            or not self.backgrounds
            or any(not isinstance(background, str) or not background for background in self.backgrounds)
        ):
            raise ValueError("Lighting backgrounds must be nonempty strings")
        for name, high in (("saturation_period_s", math.inf), ("shadows_probability", 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= high:
                raise ValueError(f"{name} must be finite and within its nonnegative domain")
        if (
            type(self.emitter_radius_m) not in (int, float)
            or not math.isfinite(self.emitter_radius_m)
            or self.emitter_radius_m <= 0
        ):
            raise ValueError("emitter_radius_m must be finite and positive")


@dataclass(frozen=True)
class RobotRandomizationConfig:
    num_links: int = 7
    # Variant id selects coherent prebuilt geometry; backend must update all related frames/inertia.
    geometry_variants: tuple[str, ...] = ("nominal", "scale099", "scale101")
    variant_scales: tuple[float, ...] = (1.0, 0.99, 1.01)
    mass_scale: tuple[float, float] = (0.9, 1.1)
    material: MaterialConfig = field(default_factory=MaterialConfig)

    def __post_init__(self):
        artifacts = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["artifacts"]
        bank = {
            "nominal"
            if item["file"] == "hiwonder_soarm101.usdc"
            else item["file"].removeprefix("hiwonder_soarm101_").removesuffix(".usdc"): item["scale"]
            for item in artifacts
        }
        if not self.geometry_variants or len(self.geometry_variants) != len(self.variant_scales):
            raise ValueError("Geometry variants need corresponding prebuilt asset scales")
        for name, scale in zip(self.geometry_variants, self.variant_scales):
            if name not in bank:
                raise ValueError(f"Unknown prebuilt robot geometry variant {name!r}")
            if not math.isfinite(scale) or not math.isclose(scale, bank[name], rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Geometry scale for {name!r} must match its prebuilt asset scale {bank[name]}")


@dataclass(frozen=True)
class SmudgeConfig:
    enabled: bool = True
    spot_count: tuple[int, int] = (0, 6)
    streak_count: tuple[int, int] = (0, 2)
    radius: tuple[float, float] = (0.025, 0.18)
    anisotropy: tuple[float, float] = (0.3, 1.0)
    opacity: tuple[float, float] = (0.05, 0.45)
    blur_pixels: tuple[float, float] = (0.0, 2.0)
    haze: tuple[float, float] = (0.0, 0.1)
    local_contrast: tuple[float, float] = (0.3, 0.9)
    tint: tuple[float, float] = (0.3, 0.8)
    drift_per_second: tuple[float, float] = (0.0, 0.002)


@dataclass(frozen=True)
class TaskSwitchConfig:
    task_names: tuple[str, ...] = ("peg_in_hole",)
    weights: tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class ActuatorRandomizationConfig:
    num_joints: int = 6
    effort_limit: tuple[float, float] = (0.6, 1.2)
    stiffness: tuple[float, float] = (15.0, 35.0)
    damping: tuple[float, float] = (0.5, 2.5)
    friction: tuple[float, float] = (0.01, 0.12)
    armature: tuple[float, float] = (0.0001, 0.002)
    velocity_limit: tuple[float, float] = (1.5, 4.0)
    delay_seconds: tuple[float, float] = (0.0, 0.025)
    encoder_bias_radians: tuple[float, float] = (-0.005, 0.005)
    encoder_noise_std: tuple[float, float] = (0.0, 0.003)
    correlation: float = 0.6
    backlash_enabled: bool = False
    backlash_radians: tuple[float, float] = (0.0, 0.015)
    qualification: str = "estimated; not a measured HX-30HM continuous-torque model"

    def __post_init__(self):
        for name in (
            "effort_limit",
            "stiffness",
            "damping",
            "friction",
            "armature",
            "velocity_limit",
            "delay_seconds",
            "encoder_bias_radians",
            "encoder_noise_std",
            "backlash_radians",
        ):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be a finite ordered range")
            if name != "encoder_bias_radians" and bounds[0] < 0:
                raise ValueError(f"{name} cannot be negative")
            if name in ("effort_limit", "velocity_limit") and bounds[0] <= 0:
                raise ValueError(f"{name} must be strictly positive")
        if self.num_joints < 1 or not 0 <= self.correlation <= 1:
            raise ValueError("Actuator joint count/correlation is invalid")


@dataclass(frozen=True)
class BaseRandomizationConfig:
    x_m: tuple[float, float] = (-0.1, 0.1)
    y_m: tuple[float, float] = (-0.1, 0.1)
    yaw_radians: tuple[float, float] = (-0.3, 0.3)
    support_radius_m: float = 0.1
    task_clearance_m: float = 0.3


@dataclass(frozen=True)
class ImageTransformConfig:
    mode: Literal["none", "one", "all"] = "all"
    shear_enabled: bool = True
    crop_enabled: bool = True
    rotation_enabled: bool = True
    color_enabled: bool = True
    transform_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    shear_degrees: tuple[float, float] = (-5.0, 5.0)
    crop_scale: tuple[float, float] = (0.85, 1.0)
    rotation_degrees: tuple[float, float] = (-12.0, 12.0)
    brightness: tuple[float, float] = (0.8, 1.2)
    contrast: tuple[float, float] = (0.8, 1.2)
    saturation: tuple[float, float] = (0.8, 1.2)
    hue_radians: tuple[float, float] = (-0.1, 0.1)


@dataclass(frozen=True)
class DistractorConfig:
    count: tuple[int, int] = (0, 6)
    max_count: int = 6
    shapes: tuple[str, ...] = ("box", "sphere", "cylinder")
    size_m: tuple[float, float] = (0.015, 0.045)
    mass_kg: tuple[float, float] = (0.005, 0.08)
    physical_probability: float = 0.0
    material: MaterialConfig = field(default_factory=MaterialConfig)


@dataclass(frozen=True)
class ConfettiConfig:
    count: tuple[int, int] = (20, 100)
    max_count: int = 100
    size_m: tuple[float, float] = (0.003, 0.012)
    aspect_ratio: tuple[float, float] = (0.3, 2.0)
    shapes: tuple[str, ...] = ("rectangle", "disc", "triangle")
    opacity: tuple[float, float] = (0.5, 1.0)
    reflectivity: tuple[float, float] = (0.0, 0.8)
    elevation_m: float = 0.0005
    physical_budget: int = 0


@dataclass(frozen=True)
class GroupControl:
    """Group RNG and lifecycle. Frozen retains the first realized draw and stops temporal changes.

    Disabled uses deterministic nominal/no-effect parameters. Fixed cadence draws once;
    episode cadence draws every reset_interval selected resets. Support-derived coordinates
    still follow their parent surface; incompatible retained layouts reject the reset.
    """

    enabled: bool = True
    seed: int = 0
    cadence: Literal["episode", "fixed"] = "episode"
    reset_interval: int = 1
    frozen: bool = False

    def __post_init__(self):
        if self.cadence not in ("episode", "fixed") or self.reset_interval < 1:
            raise ValueError("Group cadence/reset_interval is invalid")


@dataclass(frozen=True)
class RandomizationControls:
    table: GroupControl = field(default_factory=lambda: GroupControl(cadence="fixed"))
    room: GroupControl = field(default_factory=lambda: GroupControl(cadence="fixed"))
    robot: GroupControl = field(default_factory=lambda: GroupControl(cadence="fixed"))
    smudge: GroupControl = field(default_factory=GroupControl)
    task: GroupControl = field(default_factory=lambda: GroupControl(cadence="fixed"))
    actuator: GroupControl = field(default_factory=GroupControl)
    base: GroupControl = field(default_factory=GroupControl)
    image: GroupControl = field(default_factory=GroupControl)
    distractor: GroupControl = field(default_factory=lambda: GroupControl(cadence="fixed"))
    confetti: GroupControl = field(default_factory=lambda: GroupControl(cadence="fixed"))


@dataclass(frozen=True)
class RandomizationConfig:
    version: str = "hiwonder-v3-material-priors-coherent-assets"
    table: TableConfig = field(default_factory=TableConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    robot: RobotRandomizationConfig = field(default_factory=RobotRandomizationConfig)
    smudge: SmudgeConfig = field(default_factory=SmudgeConfig)
    task: TaskSwitchConfig = field(default_factory=TaskSwitchConfig)
    actuator: ActuatorRandomizationConfig = field(default_factory=ActuatorRandomizationConfig)
    base: BaseRandomizationConfig = field(default_factory=BaseRandomizationConfig)
    image: ImageTransformConfig = field(default_factory=ImageTransformConfig)
    distractor: DistractorConfig = field(default_factory=DistractorConfig)
    confetti: ConfettiConfig = field(default_factory=ConfettiConfig)
    controls: RandomizationControls = field(default_factory=RandomizationControls)


@dataclass(frozen=True)
class HiWonderEnvConfig:
    robot_model: str = "stock"
    num_envs: int = 16
    seed: int = 0
    task: str = "peg_insert"
    task_parameters: dict = field(default_factory=dict)
    backend: str = "auto"
    # Newton uses the certified manipulation recipe; PhysX retains its smaller CAD cook.
    robot_collision_profile: Literal["cad", "manipulation"] = "manipulation"
    robot_self_collisions: bool | None = None
    control_hz: float = 30.0
    physics_substeps: int = 8
    episode_seconds: float | None = None
    frame_stack: int = 1
    reset_families: tuple[str, ...] = ("reaching", "near_goal")
    grasp_library: str | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)
    camera: CameraProfile = field(default_factory=CameraProfile)
    room: RoomConfig = field(default_factory=RoomConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    actor_mode: Literal["rgb", "state_debug"] = "rgb"

    def __post_init__(self):
        if self.actor_mode not in ("rgb", "state_debug"):
            raise ValueError("actor_mode must be rgb or state_debug")
        if self.robot_collision_profile not in ("cad", "manipulation"):
            raise ValueError("robot_collision_profile must be cad or manipulation")
        if self.robot_self_collisions is not None and type(self.robot_self_collisions) is not bool:
            raise ValueError("robot_self_collisions must be bool or None")
        model = get_robot_model(self.robot_model)
        robot, actuator = self.randomization.robot, self.randomization.actuator
        if model.name != "stock" and (robot.num_links, actuator.num_joints) == (7, 6):
            robot = replace(robot, num_links=model.num_links)
            actuator = replace(actuator, num_joints=model.num_joints)
            object.__setattr__(self, "randomization", replace(self.randomization, robot=robot, actuator=actuator))
        if actuator.num_joints != model.num_joints:
            raise ValueError(f"Robot {model.name} requires {model.num_joints} actuator joints (six for stock)")
        if robot.num_links != model.num_links:
            raise ValueError(f"Robot {model.name} requires {model.num_links} physical links")
        if model.manifest_path.is_file():
            artifacts = json.loads(model.manifest_path.read_text(encoding="utf-8"))["artifacts"]
            scales = {item["file"]: item["scale"] for item in artifacts}
            for variant, scale in zip(robot.geometry_variants, robot.variant_scales):
                filename = model.asset_path(variant).name
                if filename not in scales or not math.isclose(scale, scales[filename], rel_tol=0, abs_tol=1e-7):
                    raise ValueError(f"Geometry variant {variant!r} scale differs from {model.name} manifest")
        if model.name != "stock" and self.camera.optical_transform_status == "CAD_mechanical_estimate":
            object.__setattr__(
                self, "camera", replace(self.camera, optical_transform_status=model.camera["optical_transform_status"])
            )
        if not isinstance(self.task_parameters, dict):
            raise ValueError("task_parameters must be a JSON object")
        try:
            encoded_parameters = json.dumps(self.task_parameters, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("task_parameters must contain finite JSON values") from error
        object.__setattr__(self, "task_parameters", json.loads(encoded_parameters))
        if self.task_parameters:
            from .tasks.catalog import build_task

            name, _, variant = self.task.partition(":")
            build_task(name, variant or None, task_parameters=self.task_parameters)
        if self.backend not in ("auto", "physx", "newton_vbd"):
            raise ValueError("backend must be auto, physx or newton_vbd")
        for group in ("table", "room", "robot", "task", "distractor", "confetti"):
            control = getattr(self.randomization.controls, group)
            if control.enabled and not control.frozen and control.cadence != "fixed":
                raise ValueError(
                    f"{group} uses a fixed scene cohort; cadence must be 'fixed' until reset-time mutation is supported"
                )
        if not isinstance(self.frame_stack, int) or self.frame_stack < 1:
            raise ValueError("frame_stack must be a positive integer")
        if not self.reset_families or any(
            name not in ("reaching", "near_goal", "stable_grasp") for name in self.reset_families
        ):
            raise ValueError("Invalid reset_families")
        if "stable_grasp" in self.reset_families and not self.grasp_library:
            raise ValueError("stable_grasp requires a physics-validated grasp_library")
        if (
            self.num_envs < 1
            or self.control_hz <= 0
            or self.physics_substeps < 1
            or (self.episode_seconds is not None and self.episode_seconds <= 0)
        ):
            raise ValueError("Environment count, control rate, physics substeps and horizon must be positive")
