# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cached USD light editing, independent of sensors, physics and renderer settings."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from ..config import CameraProfile, LightingConfig
from ..lighting import rgb_light_colors, validate_preview_controls, wrist_view_targets


def author_fill_light(stage: Usd.Stage, room_path: str, position: tuple[float, float, float], intensity: float) -> str:
    """Author an unshaped white source, isolated to this room's environment subtree."""
    path = room_path + "/Lights/WhiteFill"
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateRadiusAttr(0.25)
    light.CreateNormalizeAttr(True)
    light.CreateIntensityAttr(float(intensity))
    light.CreateColorAttr(Gf.Vec3f(1))
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    env_path = room_path.rsplit("/", 1)[0] or room_path
    for name in ("lightLink", "shadowLink"):
        collection = Usd.CollectionAPI.Apply(light.GetPrim(), name)
        collection.CreateIncludeRootAttr(False)
        collection.CreateExpansionRuleAttr("expandPrims")
        collection.CreateIncludesRel().SetTargets([env_path])
    return path


@dataclass
class _Source:
    """Own schema handles and attributes for one source; resolved only at construction."""

    light: UsdLux.LightAPI
    translate: UsdGeom.XformOp
    orient: UsdGeom.XformOp
    color: Usd.Attribute
    intensity: Usd.Attribute
    radius: Usd.Attribute
    cone: Usd.Attribute | None
    softness: Usd.Attribute | None


class SceneLightingRig:
    """Apply CPU room samples through cached USD handles, without looking up prims per frame.

    Positions in samples are local to each environment. Camera poses and saved targets
    are world positions and xyzw OpenCV optical quaternions. Returned state owns all
    tensor storage; loading validates before editing USD and immediately restores the
    recorded appearance. Keep the rig alive as long as its stage is in use.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        room_paths: list[dict],
        fill_light_paths: list[str],
        lighting: LightingConfig,
        camera: CameraProfile,
        env_origins: torch.Tensor,
        support_heights: torch.Tensor,
        frozen: bool = False,
    ) -> None:
        self._stage = stage
        self._lighting = lighting
        self._camera = camera
        self._origins = env_origins.detach().to(device="cpu").clone()
        self._support_z = self._origins[:, 2] + support_heights.detach().to(device="cpu")
        self._n = len(room_paths)
        counts = [len(paths["lights"]) for paths in room_paths]
        if len(fill_light_paths) != self._n or len(set(counts)) > 1:
            raise ValueError("Each room needs one fill and the same number of declared light sources")
        if self._origins.shape != (self._n, 3) or self._support_z.shape != (self._n,):
            raise ValueError("Environment origins/support heights do not match room count")
        self._l = counts[0] if counts else 0
        self._frozen = frozen
        self._overrides: dict[str, float] = {}
        self._effective_elapsed: torch.Tensor | None = None
        self._input_elapsed: torch.Tensor | None = None
        self._parameters: dict[str, torch.Tensor] | None = None
        self._sources = []
        for paths in room_paths:
            row = []
            for path in paths["lights"]:
                prim = self._required_light(path)
                api = UsdLux.LightAPI(prim)
                xform = UsdGeom.Xformable(prim)
                ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
                translate = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
                orient = ops.get(UsdGeom.XformOp.TypeOrient) or xform.AddOrientOp()
                shaped = prim.HasAPI(UsdLux.ShapingAPI)
                if lighting.motion_mode == "wrist_view":
                    UsdLux.ShapingAPI.Apply(prim)
                    # Select normalized cone-softness semantics for narrow spots:
                    # https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
                    prim.CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(2505)
                    # RGB saturation controls emission without an inherited Kelvin tint.
                    api.CreateEnableColorTemperatureAttr(False)
                    shaped = True
                shaping = UsdLux.ShapingAPI(prim) if shaped else None
                row.append(
                    _Source(
                        api,
                        translate,
                        orient,
                        api.GetColorAttr(),
                        api.GetIntensityAttr(),
                        UsdLux.SphereLight(prim).GetRadiusAttr(),
                        shaping.CreateShapingConeAngleAttr() if shaped else None,
                        shaping.CreateShapingConeSoftnessAttr() if shaped else None,
                    )
                )
            self._sources.append(row)
        self._fills = [UsdLux.LightAPI(self._required_light(path)).GetIntensityAttr() for path in fill_light_paths]

    @property
    def last_parameters(self) -> dict[str, torch.Tensor] | None:
        """Return owned copies of the last applied world targets and visual parameters."""
        return None if self._parameters is None else {key: value.clone() for key, value in self._parameters.items()}

    def set_preview(self, controls: dict | None) -> dict[str, float]:
        """Merge validated overrides; None clears overrides and resumes the sampled clock."""
        normalized = {} if controls is None else validate_preview_controls(controls)
        if controls is None:
            self._overrides = {}
            self._effective_elapsed = None
            self._input_elapsed = None
        else:
            self._overrides.update(normalized)
        return dict(self._overrides)

    def update(
        self,
        room: dict,
        camera_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Apply CPU samples using a caller-supplied optical pose for wrist-view lights.

        After the first full initialization, ``env_ids`` edits only those rooms;
        other rooms retain their visual parameters and both preview clocks exactly.
        This allows reset-pose publication without moving retained wrist targets.
        """
        if self._lighting.motion_mode == "wrist_view" and camera_pose is None:
            raise ValueError("wrist_view lighting requires an optical camera_pose")
        elapsed = room["elapsed_seconds"].to(dtype=torch.float64)
        if self._input_elapsed is None:
            effective = elapsed.clone()
        else:
            delta = elapsed - self._input_elapsed
            effective = torch.where(
                delta >= 0,
                self._effective_elapsed + delta * self._overrides.get("speed", 1.0),
                elapsed,
            )
        angle = room["light_phase"].to(torch.float64)
        if self._lighting.profile != "stationary":
            angle = angle + room["light_angular_speed"] * effective[:, None]
        colors = room["light_color"].clone()
        if "saturation" in self._overrides or "speed" in self._overrides:
            saturation = room["light_saturation"]
            if "saturation" in self._overrides:
                saturation = torch.full_like(saturation, self._overrides["saturation"])
            # The sampler owns saturation's independent period; preview speed
            # scales only the spatial orbit and hue, matching exported profiles.
            if self._lighting.profile == "rgb_disco" or "saturation" in self._overrides:
                colors = rgb_light_colors(angle, saturation).to(colors.dtype)
        positions = room["light_position"].clone()
        directions = room["light_direction"].clone()
        cone = room["light_beam_angle_degrees"].clone()
        softness = room["light_edge_softness"].clamp(0, 1).clone()
        targets = positions + self._origins[:, None] + directions
        visible = torch.zeros((self._n, self._l), dtype=torch.bool)
        if self._lighting.motion_mode == "wrist_view":
            fallback = self._origins.clone()
            fallback[:, 2] = self._support_z
            targets, visible = wrist_view_targets(
                *camera_pose,
                angle,
                room["light_view_extent"],
                self._support_z,
                fallback,
                self._camera.intrinsics,
                self._camera.width,
                self._camera.height,
                max_distance_m=min(4.0, self._camera.clipping_range_m[1]),
            )
            directions = targets - (positions + self._origins[:, None])
            distances = directions.norm(dim=-1).clamp_min(1e-6)
            directions = directions / distances[..., None]
            radius = (
                torch.full_like(distances, self._overrides["radius_m"])
                if "radius_m" in self._overrides
                else room["light_spot_radius_m"]
            )
            cone = torch.rad2deg(torch.atan(radius / distances))
        elif "speed" in self._overrides:
            directions = torch.stack((0.6 * angle.cos(), 0.6 * angle.sin(), -torch.ones_like(angle)), -1)
            directions = directions / directions.norm(dim=-1, keepdim=True)
            targets = positions + self._origins[:, None] + directions
        if self._frozen and self._parameters is not None:
            positions = self._parameters["positions"].clone()
            directions = self._parameters["directions"].clone()
            targets = self._parameters["target_positions"].clone()
            visible = self._parameters["visible_ray_mask"].clone()
            cone = self._parameters["cone_angles"].clone()
        parameters = {
            "positions": positions,
            "directions": directions,
            "colors": colors,
            "intensities": room["light_intensity"] * room["light_active"] * self._overrides.get("power", 1.0),
            "cone_angles": cone,
            "softness": softness,
            "emitter_radii": torch.full(
                (self._n, self._l),
                self._lighting.emitter_radius_m if self._lighting.motion_mode == "wrist_view" else 0.04,
            ),
            "fill_intensities": torch.full((self._n,), self._overrides["fill_intensity"])
            if "fill_intensity" in self._overrides
            else room["white_fill_intensity"].clone(),
            "target_positions": targets,
            "visible_ray_mask": visible,
        }
        if env_ids is not None and self._parameters is not None:
            selected = torch.zeros(self._n, dtype=torch.bool)
            selected[torch.as_tensor(env_ids, device="cpu", dtype=torch.long).flatten()] = True
            for key, value in parameters.items():
                retained = self._parameters[key].clone()
                retained[selected] = value[selected].to(retained.dtype)
                parameters[key] = retained
            if self._input_elapsed is not None:
                effective = torch.where(selected, effective, self._effective_elapsed)
                elapsed = torch.where(selected, elapsed, self._input_elapsed)
        self._apply(parameters)
        self._effective_elapsed = effective.clone()
        self._input_elapsed = elapsed.clone()

    def state_dict(self) -> dict:
        """Snapshot owned CPU tensors and preview clock for exact appearance replay."""
        return {
            "version": 1,
            "overrides": dict(self._overrides),
            "effective_elapsed": None if self._effective_elapsed is None else self._effective_elapsed.clone(),
            "input_elapsed": None if self._input_elapsed is None else self._input_elapsed.clone(),
            "parameters": self.last_parameters,
        }

    def load_state_dict(self, state: dict) -> None:
        """Validate all saved tensors before restoring visual USD attributes immediately."""
        if state.get("version") != 1:
            raise ValueError("Unsupported lighting checkpoint version")
        overrides = validate_preview_controls(state["overrides"])
        clocks = []
        for key in ("effective_elapsed", "input_elapsed"):
            value = state[key]
            clocks.append(None if value is None else self._checked_tensor(value, (self._n,), key))
        if (clocks[0] is None) != (clocks[1] is None):
            raise ValueError("Lighting clocks must both be present or absent")
        parameters = state["parameters"]
        if parameters is not None:
            shapes = {
                "positions": (self._n, self._l, 3),
                "directions": (self._n, self._l, 3),
                "colors": (self._n, self._l, 3),
                "target_positions": (self._n, self._l, 3),
                "fill_intensities": (self._n,),
            }
            shapes.update(
                {
                    key: (self._n, self._l)
                    for key in ("intensities", "cone_angles", "softness", "emitter_radii", "visible_ray_mask")
                }
            )
            if not isinstance(parameters, dict) or set(parameters) != set(shapes):
                raise ValueError("Invalid lighting parameter fields")
            parameters = {key: self._checked_tensor(parameters[key], shape, key) for key, shape in shapes.items()}
            if parameters["visible_ray_mask"].dtype != torch.bool:
                raise ValueError("visible_ray_mask must be boolean")
            if (parameters["directions"].norm(dim=-1) < 1e-6).any():
                raise ValueError("Lighting directions must be nonzero")
            for key in ("colors", "intensities", "fill_intensities", "cone_angles", "softness", "emitter_radii"):
                if (parameters[key] < 0).any():
                    raise ValueError(f"Lighting {key} must be nonnegative")
            if (
                (parameters["colors"] > 1).any()
                or (parameters["softness"] > 1).any()
                or (parameters["cone_angles"] > 180).any()
            ):
                raise ValueError("Lighting color, softness or cone is outside its domain")
        if parameters is not None:
            # A GUI or another USD author may have edited live attributes without
            # changing our cache. Checkpoint replay must restore actual appearance.
            self._apply(parameters, force=True)
        else:
            self._parameters = None
        self._overrides = overrides
        self._effective_elapsed, self._input_elapsed = clocks

    def _required_light(self, path: str) -> Usd.Prim:
        prim = self._stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdLux.LightAPI):
            raise ValueError(f"Missing declared light or invalid light prim: {path}")
        return prim

    def _apply(self, parameters: dict[str, torch.Tensor], force: bool = False) -> None:
        previous = self._parameters

        def changed(key: str, i: int, j: int | None = None) -> bool:
            if previous is None or force:
                return True
            return not torch.equal(
                previous[key][i] if j is None else previous[key][i, j],
                parameters[key][i] if j is None else parameters[key][i, j],
            )

        with Sdf.ChangeBlock():
            for i, row in enumerate(self._sources):
                for j, source in enumerate(row):
                    if changed("positions", i, j):
                        vector = (
                            Gf.Vec3d if source.translate.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f
                        )
                        source.translate.Set(vector(*map(float, parameters["positions"][i, j])))
                    if changed("directions", i, j):
                        direction = Gf.Vec3d(*map(float, parameters["directions"][i, j])).GetNormalized()
                        quaternion = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction).GetQuat()
                        if source.orient.GetPrecision() != UsdGeom.XformOp.PrecisionDouble:
                            quaternion = Gf.Quatf(quaternion)
                        source.orient.Set(quaternion)
                    if changed("colors", i, j):
                        source.color.Set(Gf.Vec3f(*map(float, parameters["colors"][i, j])))
                    for key, attribute in (
                        ("intensities", source.intensity),
                        ("emitter_radii", source.radius),
                        ("cone_angles", source.cone),
                        ("softness", source.softness),
                    ):
                        if attribute and changed(key, i, j):
                            attribute.Set(float(parameters[key][i, j]))
                if changed("fill_intensities", i):
                    self._fills[i].Set(float(parameters["fill_intensities"][i]))
        self._parameters = {key: value.detach().cpu().clone() for key, value in parameters.items()}

    @staticmethod
    def _checked_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid lighting {name}: expected finite tensor shape {shape}")
        return value.detach().cpu().clone()
