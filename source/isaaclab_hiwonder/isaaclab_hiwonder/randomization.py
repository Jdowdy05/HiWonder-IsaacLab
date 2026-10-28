# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded, indexed parameter state for all ten randomization groups.

This module samples parameters; the scene adapter owns physical/render application and
task-specific collision/reachability rejection. Prebuilt geometry IDs must be qualified
before use. Clutter candidates exclude the configured task-clearance disk and robot base.
"""

import json
import math
from dataclasses import asdict, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from functools import lru_cache

import torch

from .assets.registration import MANIFEST_PATH
from .config import RandomizationConfig
from .lighting import rgb_light_colors

_GROUPS = ("table", "room", "robot", "smudge", "task", "actuator", "base", "image", "distractor", "confetti")
_SUPPORT_GROUPS = ("table", "base", "distractor", "confetti")


class InvalidLayout(ValueError):
    """A sampled support/collision layout is infeasible, rather than an invalid numeric configuration."""


@lru_cache(maxsize=1)
def _cad_base_footprint_radius() -> float:
    """Circumscribe the nominal CAD base's complete XY bounds around its support origin."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    nominal = next(item for item in manifest["artifacts"] if item["scale"] == 1.0)
    bounds = nominal["links"]["base"]["bounds_local_m"]
    return math.hypot(max(abs(row[0]) for row in bounds), max(abs(row[1]) for row in bounds))


def _validate_physical_confetti_base(confetti, base, base_cfg, geometry_scale):
    """Keep physical card footprints out of the scaled CAD base, preserving visual-only occlusion."""
    physical = confetti["physical"] & confetti["active"]
    if not physical.any():
        return
    base_radius = max(base_cfg.support_radius_m, _cad_base_footprint_radius() * float(geometry_scale))
    card_radius = confetti["size"][physical] * (1 + confetti["aspect_ratio"][physical].square()).sqrt() / 2
    distance = (confetti["position"][physical, :2] - base["position"][:2]).norm(dim=-1)
    # Reserve 2mm beyond the conservative footprint envelopes for contact margins.
    if torch.any(distance < base_radius + card_radius + 0.002):
        raise InvalidLayout("Physical confetti footprint collides with CAD base envelope")


def table_outline(shape: str, width: float, depth: float, radius: float) -> torch.Tensor:
    """Return the actual convex CCW tabletop polygon [V,2], shared with scene mesh authoring."""
    w, d = float(width) / 2, float(depth) / 2
    if min(w, d) <= 0:
        raise ValueError("Table outline dimensions must be positive")
    if shape in ("ellipse", "circle", "hexagon", "octagon"):
        count = {"ellipse": 64, "circle": 64, "hexagon": 6, "octagon": 8}[shape]
        if shape == "circle":
            w = d = min(w, d)
        angles = torch.arange(count, dtype=torch.float64) * (2 * math.pi / count)
        return torch.stack((w * angles.cos(), d * angles.sin()), -1)
    if shape == "rounded_rectangle":
        radius = min(float(radius), w, d)
        pieces = []
        for x, y, start in (
            (w - radius, d - radius, 0),
            (-w + radius, d - radius, 90),
            (-w + radius, -d + radius, 180),
            (w - radius, -d + radius, 270),
        ):
            angles = (torch.arange(8, dtype=torch.float64) * 90 / 8 + start) * math.pi / 180
            pieces.append(torch.stack((x + radius * angles.cos(), y + radius * angles.sin()), -1))
        return torch.cat(pieces)
    if shape != "rectangle":
        raise ValueError(f"Unsupported table shape: {shape}")
    return torch.tensor([[-w, -d], [w, -d], [w, d], [-w, d]], dtype=torch.float64)


def _table_support(table, shape, centers, radii):
    outline = table_outline(shape, table["width"], table["depth"], table["edge_radius"])
    edges = outline.roll(-1, 0) - outline
    length = edges.norm(dim=-1)
    keep = length > 1e-12
    normals = torch.stack((edges[keep, 1], -edges[keep, 0]), -1) / length[keep, None]
    limits = (outline[keep] * normals).sum(-1)
    radius = torch.as_tensor(radii, dtype=torch.float64)
    return (centers.double() @ normals.T + radius[..., None] <= limits + 1e-10).all(-1)


def _nominal_profile(profile):
    """Collapse typed range profiles to midpoint values and categorical pools to their first item."""
    values = {}
    for item in dataclass_fields(profile):
        value = getattr(profile, item.name)
        if is_dataclass(value):
            value = _nominal_profile(value)
        elif isinstance(value, tuple) and value:
            if isinstance(value[0], str):
                value = value[:1]
            elif len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                midpoint = (value[0] + value[1]) / 2
                if all(isinstance(v, int) for v in value):
                    midpoint = int(midpoint)
                value = (midpoint, midpoint)
        values[item.name] = value
    # Keep the first configured asset and its associated scale as one coherent pair.
    if "geometry_variants" in values:
        values.update(
            geometry_variants=profile.geometry_variants[:1],
            variant_scales=profile.variant_scales[:1],
            mass_scale=(1.0, 1.0),
        )
    if "task_names" in values:
        values["weights"] = (1.0,)
    return replace(profile, **values)


def _distractor_positions(n, count, radius, table, table_shape, base, base_cfg, rng):
    """Place bounded spheres in nonoverlapping supported slots outside protected task/base disks."""
    if int(count) > 0 and (2 * radius > float(table["width"]) or 2 * radius > float(table["depth"])):
        raise InvalidLayout("Table has no valid support inset for requested distractors")
    grid_x = torch.linspace(-float(table["width"]) / 2 + radius, float(table["width"]) / 2 - radius, 12)
    grid_y = torch.linspace(-float(table["depth"]) / 2 + radius, float(table["depth"]) / 2 - radius, 12)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    candidates = torch.stack((xx.flatten(), yy.flatten()), -1)
    candidates = candidates[torch.randperm(len(candidates), generator=rng)]
    candidates = candidates[_table_support(table, table_shape, candidates, radius)]
    chosen = []
    for candidate in candidates if n else []:
        if abs(candidate[0]) + radius > table["width"] / 2 or abs(candidate[1]) + radius > table["depth"] / 2:
            continue
        if torch.linalg.vector_norm(candidate) < base_cfg.task_clearance_m + radius:
            continue
        if torch.linalg.vector_norm(candidate - base["position"][:2]) < base_cfg.support_radius_m + radius:
            continue
        if any(torch.linalg.vector_norm(candidate - prior) < 2 * radius + 0.005 for prior in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) == n:
            break
    if len(chosen) < int(count):
        raise InvalidLayout("Cannot place requested distractors without overlap/support violation")
    position = torch.zeros((n, 3))
    if chosen:
        position[: len(chosen), :2] = torch.stack(chosen)
    return position


def _uniform(rng, bounds, shape=()):
    if not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
        raise ValueError(f"Invalid sampling range: {bounds}")
    return torch.rand(shape, generator=rng) * (bounds[1] - bounds[0]) + bounds[0]


def _integer(rng, bounds, shape=()):
    if bounds[0] < 0 or bounds[1] < bounds[0]:
        raise ValueError(f"Invalid integer range: {bounds}")
    return torch.randint(bounds[0], bounds[1] + 1, shape, generator=rng)


def _choice(rng, pool, shape=()):
    if not pool:
        raise ValueError("Randomization pools cannot be empty")
    return _integer(rng, (0, len(pool) - 1), shape)


def _material(rng, profile, shape=(), prefix="material_"):
    """Sample family-conditioned PBR priors inside the configured numeric ranges.

    Presets select subranges of roughness/metallic/specular, expressed as fractions
    of each configured interval. Constant numeric intervals override that variation.
    These are appearance priors, not measured material or friction models; texture
    selection and physical surface ranges remain explicit independent controls.
    """
    presets = {
        "matte": {"roughness": (0.7, 1.0), "metallic": (0.0, 0.05), "specular": (0.0, 0.35)},
        "wood": {"roughness": (0.35, 0.85), "metallic": (0.0, 0.0), "specular": (0.15, 0.55)},
        "plastic": {"roughness": (0.15, 0.65), "metallic": (0.0, 0.05), "specular": (0.35, 0.8)},
        "metal": {"roughness": (0.0, 0.55), "metallic": (0.75, 1.0), "specular": (0.6, 1.0)},
    }
    for family in profile.families:
        if family not in presets:
            raise ValueError(f"Unknown material family {family!r}; choose {tuple(presets)}")
    values = {
        prefix + "color": _uniform(rng, profile.color, (*shape, 3)),
        prefix + "roughness": _uniform(rng, profile.roughness, shape),
        prefix + "metallic": _uniform(rng, profile.metallic, shape),
        prefix + "specular": _uniform(rng, profile.specular, shape),
        prefix + "uv_scale": _uniform(rng, profile.uv_scale, shape),
        prefix + "friction": _uniform(rng, profile.friction, shape),
        prefix + "restitution": _uniform(rng, profile.restitution, shape),
        prefix + "family_id": _choice(rng, profile.families, shape),
        prefix + "texture_id": _choice(rng, profile.texture_ids, shape),
    }
    for family_id, family in enumerate(profile.families):
        selected = values[prefix + "family_id"] == family_id
        for name, (minimum, maximum) in presets[family].items():
            low, high = getattr(profile, name)
            original = values[prefix + name]
            effective = low + (high - low) * minimum + (original - low) * (maximum - minimum)
            values[prefix + name] = torch.where(selected, effective, original)
    return values


def _confetti_positions(rng, table, table_shape, capacity, count, margin, radii):
    xy = torch.zeros((capacity, 2))
    if count == 0:
        return xy
    if 2 * margin >= min(float(table["width"]), float(table["depth"])):
        raise InvalidLayout("Confetti footprint has no valid tabletop support inset")
    bounds = (
        (-float(table["width"]) / 2 + margin, float(table["width"]) / 2 - margin),
        (-float(table["depth"]) / 2 + margin, float(table["depth"]) / 2 - margin),
    )
    positions = torch.stack((_uniform(rng, bounds[0], (count,)), _uniform(rng, bounds[1], (count,))), -1)
    for _ in range(128):
        invalid = ~_table_support(table, table_shape, positions, radii[:count])
        if not invalid.any():
            xy[:count] = positions
            return xy
        positions[invalid, 0] = _uniform(rng, bounds[0], (int(invalid.sum()),))
        positions[invalid, 1] = _uniform(rng, bounds[1], (int(invalid.sum()),))
    raise InvalidLayout("Confetti footprint has no valid tabletop support")


class RandomizationManager:
    """Episode state with independent per-environment random streams and temporal lights.

    Sampling runs on CPU per reset, then transfers the resulting small parameter records to
    ``device``. Frame-step light updates are batched device operations and draw no randomness.
    """

    def __init__(
        self,
        num_envs: int,
        config: RandomizationConfig | None = None,
        seed: int = 0,
        device: str = "cpu",
        bootstrap_attempts: int = 1,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs, self.config, self.device = num_envs, config or RandomizationConfig(), torch.device(device)
        self.generators = {
            group: [
                torch.Generator().manual_seed(
                    (seed + i * 1000003 + j * 1000000007 + getattr(self.config.controls, group).seed) % (2**63 - 1)
                )
                for i in range(num_envs)
            ]
            for j, group in enumerate(_GROUPS)
        }
        self.reset_counts = {group: torch.full((num_envs,), -1, dtype=torch.long) for group in _GROUPS}
        changes = {}
        for group in _GROUPS:
            if not getattr(self.config.controls, group).enabled:
                name = "lighting" if group == "room" else group
                changes[name] = _nominal_profile(getattr(self.config, name))
        if "smudge" in changes:
            changes["smudge"] = replace(
                changes["smudge"],
                enabled=False,
                spot_count=(0, 0),
                streak_count=(0, 0),
                haze=(0, 0),
                drift_per_second=(0, 0),
            )
        if "image" in changes:
            changes["image"] = replace(changes["image"], mode="none")
        for name in ("distractor", "confetti"):
            if name in changes:
                changes[name] = replace(changes[name], count=(0, 0))
        if "base" in changes:
            changes["base"] = replace(changes["base"], x_m=(0, 0), y_m=(0, 0), yaw_radians=(0, 0))
        if "lighting" in changes:
            changes["lighting"] = replace(changes["lighting"], profile="stationary")
        self._sampling_config = replace(self.config, **changes)
        self._state: dict[str, dict[str, torch.Tensor]] = {}
        self._initialized = torch.zeros(num_envs, dtype=torch.bool)
        self.bootstrap_attempts = torch.zeros(num_envs, dtype=torch.long)
        self._bootstrap_rejection = False
        if bootstrap_attempts < 1:
            raise ValueError("bootstrap_attempts must be positive")
        if bootstrap_attempts == 1:
            self.reset()
            self.bootstrap_attempts[:] = 1
        else:
            self._bootstrap_rejection = True
            try:
                for i in range(num_envs):
                    for attempt in range(bootstrap_attempts):
                        try:
                            self.reset([i])
                        except InvalidLayout:
                            if attempt + 1 == bootstrap_attempts:
                                raise InvalidLayout(
                                    f"No supported initial cohort {i} after {bootstrap_attempts} attempts"
                                ) from None
                        else:
                            self.bootstrap_attempts[i] = attempt + 1
                            break
            finally:
                self._bootstrap_rejection = False

    def _ids(self, env_ids) -> torch.Tensor:
        ids = torch.arange(self.num_envs) if env_ids is None else torch.as_tensor(env_ids, device="cpu")
        if ids.ndim != 1 or ids.dtype == torch.bool or not torch.all(ids == ids.long()):
            raise ValueError("env_ids must be a one-dimensional sequence of integer IDs")
        ids = ids.long()
        if torch.any((ids < 0) | (ids >= self.num_envs)) or len(torch.unique(ids)) != len(ids):
            raise ValueError("env_ids must be unique valid environment indices")
        return ids

    def _sample(
        self, rngs: dict[str, torch.Generator], previous: dict | None, resample: set[str]
    ) -> dict[str, dict[str, torch.Tensor]]:
        cfg = self._sampling_config
        base_cfg = cfg.base
        confetti_cfg = cfg.confetti

        def generator(group):
            if not getattr(self.config.controls, group).enabled:
                return torch.Generator().manual_seed(0)
            rng = rngs[group]
            return rng

        if "table" in resample:
            rng = generator("table")
            table_cfg = cfg.table
            table = dict(
                width=_uniform(rng, table_cfg.width_m),
                depth=_uniform(rng, table_cfg.depth_m),
                height=_uniform(rng, table_cfg.height_m),
                thickness=_uniform(rng, table_cfg.thickness_m),
                edge_radius=_uniform(rng, table_cfg.edge_radius_m),
                shape_id=_choice(rng, table_cfg.shapes),
                **_material(rng, table_cfg.material),
            )
            if min(table["width"], table["depth"], table["height"], table["thickness"]) <= 0:
                raise ValueError("Table dimensions must be positive")
        else:
            table = previous["table"]
        table_shape = cfg.table.shapes[int(table["shape_id"])]
        if "base" in resample:
            rng = generator("base")
            x, y = _uniform(rng, base_cfg.x_m), _uniform(rng, base_cfg.y_m)
            if not _table_support(table, table_shape, torch.stack((x, y)), base_cfg.support_radius_m):
                raise InvalidLayout("Base placement exceeds table support; narrow base ranges or enlarge table")
            yaw = _uniform(rng, base_cfg.yaw_radians)
            base = dict(
                position=torch.stack((x, y, table["height"])),
                quaternion_xyzw=torch.tensor([0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]),
            )
        else:
            base = previous["base"]
        base = {key: value.clone() for key, value in base.items()}
        base["position"][2] = table["height"]
        if not _table_support(table, table_shape, base["position"][:2], base_cfg.support_radius_m):
            raise InvalidLayout("Retained base has no table support")
        if "room" in resample:
            rng = generator("room")
            light_cfg = cfg.lighting
            if light_cfg.count[1] > light_cfg.max_lights or light_cfg.profile not in (
                "stationary",
                "white_disco",
                "rgb_disco",
            ):
                raise ValueError("Invalid light capacity/profile")
            lights = light_cfg.max_lights
            count = _integer(rng, light_cfg.count)
            phase = _uniform(rng, (0, 2 * math.pi), (lights,))
            room = dict(
                **_material(rng, light_cfg.wall_material, (4,), prefix="wall_"),
                background_id=_choice(rng, light_cfg.backgrounds),
                light_count=count,
                light_active=torch.arange(lights) < count,
                light_type_id=_choice(rng, light_cfg.types, (lights,)),
                light_position=torch.stack(
                    (
                        _uniform(rng, (-1.2, 1.2), (lights,)),
                        _uniform(rng, (-1.2, 1.2), (lights,)),
                        _uniform(rng, (2.2, 3.2), (lights,)),
                    ),
                    -1,
                ),
                light_direction=torch.zeros((lights, 3)),
                light_intensity=_uniform(rng, light_cfg.intensity, (lights,)),
                light_color=torch.ones((lights, 3)),
                light_temperature_kelvin=_uniform(rng, light_cfg.temperature_kelvin, (lights,)),
                light_beam_angle_degrees=_uniform(rng, light_cfg.beam_angle_degrees, (lights,)),
                light_shadows=_uniform(rng, (0, 1), (lights,)) < light_cfg.shadows_probability,
                light_phase=phase,
                light_angular_speed=_uniform(rng, light_cfg.angular_speed, (lights,)),
                elapsed_seconds=torch.tensor(0.0, dtype=torch.float64),
                # Append draws so existing room fields retain their seeded realization.
                light_saturation=_uniform(rng, light_cfg.color_saturation, (lights,)),
                light_saturation_phase=_uniform(rng, (0, 2 * math.pi), (lights,)),
                light_spot_radius_m=_uniform(rng, light_cfg.spot_radius_m, (lights,)),
                light_edge_softness=_uniform(rng, light_cfg.edge_softness, (lights,)),
                light_view_extent=_uniform(rng, light_cfg.view_extent, (lights,)),
                white_fill_intensity=_uniform(rng, light_cfg.white_fill_intensity),
            )
        else:
            room = previous["room"]
        if "robot" in resample:
            rng = generator("robot")
            robot_cfg = cfg.robot
            if len(robot_cfg.variant_scales) != len(robot_cfg.geometry_variants):
                raise ValueError("Geometry variants need corresponding coherent scales")
            variant = _choice(rng, robot_cfg.geometry_variants)
            scale = torch.tensor(robot_cfg.variant_scales)[variant]
            mass_scale = _uniform(rng, robot_cfg.mass_scale, (robot_cfg.num_links,)) * scale.pow(3)
            robot = dict(
                geometry_variant_id=variant,
                geometry_scale=scale,
                mass_scale=mass_scale,
                inertia_scale=mass_scale * scale.square(),
                **_material(rng, robot_cfg.material, (robot_cfg.num_links,)),
            )
        else:
            robot = previous["robot"]
        if "actuator" in resample:
            rng = generator("actuator")
            actuator_cfg = cfg.actuator
            if not 0 <= actuator_cfg.correlation <= 1:
                raise ValueError("Actuator correlation must be between zero and one")
            shape = (actuator_cfg.num_joints,)
            shared = _uniform(rng, (0, 1))
            actuator = {}
            for key in (
                "effort_limit",
                "stiffness",
                "damping",
                "friction",
                "armature",
                "velocity_limit",
                "delay_seconds",
                "encoder_bias_radians",
                "encoder_noise_std",
            ):
                bounds = getattr(actuator_cfg, key)
                unit = shared * actuator_cfg.correlation + _uniform(rng, (0, 1), shape) * (1 - actuator_cfg.correlation)
                actuator[key] = unit * (bounds[1] - bounds[0]) + bounds[0]
            actuator["backlash_radians"] = (
                _uniform(rng, actuator_cfg.backlash_radians, shape)
                if actuator_cfg.backlash_enabled
                else torch.zeros(shape)
            )
            if torch.any(actuator["effort_limit"] <= 0) or torch.any(actuator["velocity_limit"] <= 0):
                raise ValueError("Actuator limits must be positive")
        else:
            actuator = previous["actuator"]
        if "task" in resample:
            rng = generator("task")
            weights = torch.tensor(cfg.task.weights)
            if (
                len(cfg.task.task_names) != len(weights)
                or torch.any(weights < 0)
                or not torch.isfinite(weights).all()
                or weights.sum() <= 0
            ):
                raise ValueError("Task names need valid nonnegative sampling weights")
            task = dict(task_id=torch.multinomial(weights, 1, generator=rng).squeeze(0))
        else:
            task = previous["task"]
        if "smudge" in resample:
            rng = generator("smudge")
            smudge = dict(
                seed=_integer(rng, (0, 2**31 - 1)),
                spot_count=_integer(rng, cfg.smudge.spot_count),
                streak_count=_integer(rng, cfg.smudge.streak_count),
                haze=_uniform(rng, cfg.smudge.haze),
                offset=torch.zeros(2),
                drift_velocity=_uniform(rng, cfg.smudge.drift_per_second, (2,))
                * torch.sign(_uniform(rng, (-1, 1), (2,))),
            )
            if not cfg.smudge.enabled:
                smudge["spot_count"] = torch.tensor(0)
                smudge["streak_count"] = torch.tensor(0)
                smudge["haze"] = torch.tensor(0.0)
                smudge["drift_velocity"] = torch.zeros(2)
        else:
            smudge = previous["smudge"]
        if "image" in resample:
            rng = generator("image")
            image = dict(seed=_integer(rng, (0, 2**31 - 1)))
            if not self.config.controls.image.enabled:
                image["seed"] = torch.tensor(0)
        else:
            image = previous["image"]
        if "distractor" in resample:
            rng = generator("distractor")
            distractor_cfg = cfg.distractor
            n = distractor_cfg.max_count
            count = _integer(rng, distractor_cfg.count)
            if distractor_cfg.count[1] > n:
                raise ValueError("Distractor count exceeds capacity")
            size = _uniform(rng, distractor_cfg.size_m, (n, 3))
            # Conservative bounding-sphere support/contact rejection, task-specific geometry is adapter-owned.
            radius = math.sqrt(3) * distractor_cfg.size_m[1] / 2
            position = _distractor_positions(n, count, radius, table, table_shape, base, base_cfg, rng)
            position[:, 2] = table["height"] + size[:, 2] / 2 + 0.001
            distractor_yaw = _uniform(rng, (-math.pi, math.pi), (n,))
            orientation = torch.zeros((n, 4))
            orientation[:, 2], orientation[:, 3] = (distractor_yaw / 2).sin(), (distractor_yaw / 2).cos()
            distractor = dict(
                count=count,
                active=torch.arange(n) < count,
                shape_id=_choice(rng, distractor_cfg.shapes, (n,)),
                size=size,
                position=position,
                quaternion_xyzw=orientation,
                mass=_uniform(rng, distractor_cfg.mass_kg, (n,)),
                physical=_uniform(rng, (0, 1), (n,)) < distractor_cfg.physical_probability,
                **_material(rng, distractor_cfg.material, (n,)),
            )
        else:
            distractor = previous["distractor"]
        distractor = {key: value.clone() for key, value in distractor.items()}
        distractor["position"][:, 2] = table["height"] + distractor["size"][:, 2] / 2 + 0.001
        active = distractor["active"]
        centers = distractor["position"][active, :2]
        radii = distractor["size"][active].norm(dim=-1) / 2
        if not _table_support(table, table_shape, centers, radii).all():
            raise InvalidLayout("Retained distractor layout has no table support")
        if torch.any((centers - base["position"][:2]).norm(dim=-1) < radii + base_cfg.support_radius_m):
            raise InvalidLayout("Retained distractor layout collides with base")
        if "confetti" in resample:
            rng = generator("confetti")
            n = confetti_cfg.max_count
            count = _integer(rng, confetti_cfg.count)
            if confetti_cfg.count[1] > n or confetti_cfg.physical_budget > n or confetti_cfg.elevation_m <= 0:
                raise ValueError("Invalid confetti capacity/elevation")
            size = _uniform(rng, confetti_cfg.size_m, (n,))
            aspect = _uniform(rng, confetti_cfg.aspect_ratio, (n,))
            margin = confetti_cfg.size_m[1] * max(1.0, confetti_cfg.aspect_ratio[1])
            piece_radius = size * torch.maximum(aspect, torch.ones_like(aspect))
            xy = _confetti_positions(rng, table, table_shape, n, int(count), margin, piece_radius)
            position = torch.cat((xy, torch.full((n, 1), float(table["height"]) + confetti_cfg.elevation_m)), -1)
            active = torch.arange(n) < count
            confetti = dict(
                count=count,
                active=active,
                shape_id=_choice(rng, confetti_cfg.shapes, (n,)),
                size=size,
                aspect_ratio=aspect,
                position=position,
                yaw=_uniform(rng, (-math.pi, math.pi), (n,)),
                color=_uniform(rng, (0, 1), (n, 3)),
                opacity=_uniform(rng, confetti_cfg.opacity, (n,)),
                reflectivity=_uniform(rng, confetti_cfg.reflectivity, (n,)),
                seed=_integer(rng, (0, 2**31 - 1)),
                physical=(torch.arange(n) < confetti_cfg.physical_budget) & active,
                coverage=((size.square() * aspect * active).sum() / (table["width"] * table["depth"])).clamp(0, 1),
            )
        else:
            confetti = previous["confetti"]
        confetti = {key: value.clone() for key, value in confetti.items()}
        confetti["position"][:, 2] = table["height"] + confetti_cfg.elevation_m
        active = confetti["active"]
        confetti["coverage"] = (
            (confetti["size"].square() * confetti["aspect_ratio"] * active).sum() / (table["width"] * table["depth"])
        ).clamp(0, 1)
        margin = confetti["size"][active] * torch.maximum(
            confetti["aspect_ratio"][active], torch.ones_like(confetti["aspect_ratio"][active])
        )
        if not _table_support(table, table_shape, confetti["position"][active, :2], margin).all():
            raise InvalidLayout("Retained confetti layout has no table support")
        _validate_physical_confetti_base(confetti, base, base_cfg, robot["geometry_scale"])
        return dict(
            table=table,
            room=room,
            robot=robot,
            smudge=smudge,
            task=task,
            actuator=actuator,
            base=base,
            image=image,
            distractor=distractor,
            confetti=confetti,
        )

    def reset(
        self, env_ids=None, groups: list[str] | None = None, force: bool = False, max_attempts: int = 1
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Replace selected records atomically, with optional bounded rejection of newly sampled bases.

        Rejected bases advance only their own stream; all other draws and cadence
        counts occur once. Exhaustion restores the complete transaction. The
        default remains a single atomic draw, including structural redraw callers.
        """
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        ids = self._ids(env_ids)
        if not len(ids):
            return self.snapshot(ids)
        selected = set(_GROUPS if groups is None else groups)
        if not selected.issubset(_GROUPS):
            raise ValueError("Unknown randomization group")
        # Sample all requested records before modifying state. Restore RNGs if rejection occurs.
        rng_states = {group: [self.generators[group][i].get_state() for i in ids.tolist()] for group in _GROUPS}
        previous_counts = {group: counts.clone() for group, counts in self.reset_counts.items()}
        try:
            records = []
            for i in ids.tolist():
                previous = (
                    {
                        group: {key: value[i].clone().cpu() for key, value in values.items()}
                        for group, values in self._state.items()
                    }
                    if self._initialized[i]
                    else None
                )
                resample = set()
                for group in _GROUPS:
                    control = getattr(self.config.controls, group)
                    if group in selected:
                        self.reset_counts[group][i] += 1
                    if previous is None or (
                        group in selected
                        and control.enabled
                        and not control.frozen
                        and (
                            force
                            or (
                                control.cadence == "episode"
                                and int(self.reset_counts[group][i]) % control.reset_interval == 0
                            )
                        )
                    ):
                        resample.add(group)
                rngs = {group: self.generators[group][i] for group in _GROUPS}
                attempt_states = {group: rng.get_state() for group, rng in rngs.items()}
                base_control = self.config.controls.base
                retry_base = "base" in resample and base_control.enabled and not base_control.frozen
                for attempt in range(max_attempts):
                    try:
                        record = self._sample(rngs, previous, resample)
                    except InvalidLayout as error:
                        if not retry_base or max_attempts == 1:
                            raise
                        if attempt + 1 == max_attempts:
                            raise InvalidLayout(
                                f"No supported base for environment {i} after {max_attempts} attempts: {error}"
                            ) from None
                        for group, rng in rngs.items():
                            if group != "base":
                                rng.set_state(attempt_states[group])
                    else:
                        records.append(record)
                        break
        except Exception as error:
            for group in _GROUPS:
                control = getattr(self.config.controls, group)
                if (
                    self._bootstrap_rejection
                    and isinstance(error, InvalidLayout)
                    and group in _SUPPORT_GROUPS
                    and control.enabled
                    and not control.frozen
                ):
                    continue
                for i, state in zip(ids.tolist(), rng_states[group]):
                    self.generators[group][i].set_state(state)
            self.reset_counts = previous_counts
            raise
        target_ids = ids.to(self.device)
        for group, fields in records[0].items():
            if group not in self._state:
                self._state[group] = {}
            for key in fields:
                values = torch.stack([record[group][key] for record in records]).to(self.device)
                if key not in self._state[group]:
                    self._state[group][key] = torch.empty(
                        (self.num_envs, *values.shape[1:]), device=self.device, dtype=values.dtype
                    )
                self._state[group][key][target_ids] = values
        self._update_lights(target_ids)
        self._initialized[ids] = True
        return self.snapshot(ids)

    def redraw_cohorts(self, env_ids=None, max_attempts: int = 32) -> dict:
        """Construction-only coherent redraw; bypass cadence, while honoring enable/frozen controls.

        Rejected support draws advance only enabled support-group streams. Accepted frozen
        appearance/robot state and every unselected environment remain unchanged. The scene
        adapter must call this before building the fixed physical bank, never during rollouts.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        ids = self._ids(env_ids)
        self._bootstrap_rejection = True
        try:
            for i in ids.tolist():
                for attempt in range(max_attempts):
                    try:
                        self.reset([i], groups=list(_SUPPORT_GROUPS), force=True)
                    except InvalidLayout:
                        if attempt + 1 == max_attempts:
                            raise InvalidLayout(
                                f"No supported redraw for cohort {i} after {max_attempts} attempts"
                            ) from None
                    else:
                        self.bootstrap_attempts[i] += attempt + 1
                        break
        finally:
            self._bootstrap_rejection = False
        return self.snapshot(ids)

    def _update_lights(self, ids: torch.Tensor) -> None:
        room = self._state["room"]
        angle = room["light_phase"][ids] + room["light_angular_speed"][ids] * room["elapsed_seconds"][ids, None]
        profile = self._sampling_config.lighting.profile
        if profile == "stationary":
            angle = room["light_phase"][ids]
        direction = torch.stack((0.6 * angle.cos(), 0.6 * angle.sin(), torch.full_like(angle, -1)), -1)
        room["light_direction"][ids] = (direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)).to(
            room["light_direction"].dtype
        )
        if profile == "rgb_disco":
            lighting = self._sampling_config.lighting
            if lighting.saturation_period_s > 0:
                phase = room["light_saturation_phase"][ids] + (
                    room["elapsed_seconds"][ids, None] * (2 * math.pi / lighting.saturation_period_s)
                )
                low, high = lighting.color_saturation
                room["light_saturation"][ids] = (low + (high - low) * (phase.sin() + 1) / 2).to(
                    room["light_saturation"].dtype
                )
            room["light_color"][ids] = rgb_light_colors(angle, room["light_saturation"][ids]).to(
                room["light_color"].dtype
            )
        else:
            room["light_color"][ids] = 1.0

    def step(self, dt: float, env_ids=None, groups=None) -> dict[str, dict[str, torch.Tensor]]:
        """Advance temporal state [s]; groups projects returned copies, never the updates."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if groups is not None and not set(groups).issubset(_GROUPS):
            raise ValueError("Unknown randomization group")
        ids = self._ids(env_ids).to(self.device)
        if self.config.controls.room.enabled and not self.config.controls.room.frozen:
            self._state["room"]["elapsed_seconds"][ids] += dt
            self._update_lights(ids)
        if self.config.controls.smudge.enabled and not self.config.controls.smudge.frozen:
            self._state["smudge"]["offset"][ids] += self._state["smudge"]["drift_velocity"][ids] * dt
        return self.snapshot(ids.cpu(), groups=groups)

    def snapshot(self, env_ids=None, groups=None) -> dict[str, dict[str, torch.Tensor]]:
        """Return owned tensors in selected ID order, optionally projecting group names."""
        ids = self._ids(env_ids).to(self.device)
        selected = set(_GROUPS if groups is None else groups)
        if not selected.issubset(_GROUPS):
            raise ValueError("Unknown randomization group")
        return {
            group: {key: value[ids].clone() for key, value in fields.items()}
            for group, fields in self._state.items()
            if group in selected
        }

    def state_dict(self) -> dict:
        return dict(
            version=2,
            config=asdict(self.config),
            num_envs=self.num_envs,
            samples=self.snapshot(),
            rng_states={group: [rng.get_state().clone() for rng in rngs] for group, rngs in self.generators.items()},
            reset_counts={group: value.clone() for group, value in self.reset_counts.items()},
            bootstrap_attempts=self.bootstrap_attempts.clone(),
        )

    def load_state_dict(self, state: dict) -> None:
        """Restore parameter state and per-env streams; changed nuisance distributions are rejected."""
        if state["version"] != 2 or state["num_envs"] != self.num_envs or state["config"] != asdict(self.config):
            raise ValueError(
                "Randomization schema/environment/config mismatch; version SGS histories when changing distributions"
            )
        samples = state["samples"]
        if (
            samples.keys() != self._state.keys()
            or set(state["rng_states"]) != set(_GROUPS)
            or set(state["reset_counts"]) != set(_GROUPS)
        ):
            raise ValueError("Corrupt randomization checkpoint")
        if any(
            len(state["rng_states"][group]) != self.num_envs or state["reset_counts"][group].shape != (self.num_envs,)
            for group in _GROUPS
        ):
            raise ValueError("Corrupt randomization generator/cadence state")
        for group, fields in self._state.items():
            if samples[group].keys() != fields.keys():
                raise ValueError("Randomization checkpoint field mismatch")
            for key, value in fields.items():
                candidate = samples[group][key]
                if candidate.shape != value.shape or candidate.dtype != value.dtype:
                    raise ValueError("Randomization checkpoint tensor mismatch")
        generators = {
            group: [torch.Generator().set_state(value.cpu()) for value in values]
            for group, values in state["rng_states"].items()
        }
        self._state = {
            group: {key: value.to(self.device).clone() for key, value in fields.items()}
            for group, fields in samples.items()
        }
        self.generators = generators
        self.reset_counts = {group: value.cpu().clone() for group, value in state["reset_counts"].items()}
        if "bootstrap_attempts" in state:
            self.bootstrap_attempts = state["bootstrap_attempts"].cpu().clone()
        self._initialized[:] = True

    @property
    def image_config(self):
        """Effective image config, including disabled-group bypass. Use to construct ImageRandomizer."""
        return self._sampling_config.image

    @property
    def smudge_config(self):
        """Effective smudge config, including disabled-group bypass."""
        return self._sampling_config.smudge
