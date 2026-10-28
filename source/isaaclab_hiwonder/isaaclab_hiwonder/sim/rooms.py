# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD room authoring with owned materials, linked lights and instanced confetti.

Geometry dimensions are baked in meters before physics import. Runtime numeric material
changes are applied through Isaac Lab VisualMaterial assets by the environment adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .asset_cache import author_mesh_geometry, relative_asset_path


@dataclass(frozen=True)
class MaterialSample:
    """One sampled material, distinct from the distribution configuration."""

    color: tuple[float, float, float] = (0.55, 0.55, 0.55)
    roughness: float = 0.65
    metallic: float = 0.0
    specular: float = 0.5
    opacity: float = 1.0
    uv_scale: tuple[float, float] = (1.0, 1.0)
    texture_path: str | None = None


@dataclass(frozen=True)
class RoomParameters:
    """Concrete per-room authoring parameters, with internal room dimensions."""

    width: float = 3.6576
    depth: float = 3.6576
    height: float = 3.6576
    wall_thickness: float = 0.05
    wall_clearance: float = 0.3
    table_width: float = 1.0
    table_depth: float = 0.8
    table_height: float = 0.75
    table_thickness: float = 0.04
    table_shape: str = "rectangle"
    table_edge_radius: float = 0.02
    table_material: MaterialSample = field(default_factory=MaterialSample)
    wall_materials: tuple[MaterialSample, ...] = field(default_factory=lambda: (MaterialSample(),) * 4)
    floor_material: MaterialSample = field(default_factory=lambda: MaterialSample(color=(0.15, 0.16, 0.18)))
    light_positions: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 2.8),)
    light_colors: tuple[tuple[float, float, float], ...] = ((1.0, 1.0, 1.0),)
    light_intensities: tuple[float, ...] = (1200.0,)
    light_beam_angles: tuple[float, ...] = (60.0,)
    confetti_count: int = 0
    confetti_size: tuple[float, float] = (0.004, 0.018)
    seed: int = 0

    def __post_init__(self):
        if min(self.width, self.depth, self.height, self.wall_thickness, self.table_thickness) <= 0:
            raise ValueError("Room and table dimensions must be positive")
        if self.table_width > self.width - 2 * self.wall_clearance:
            raise ValueError("Table width violates wall clearance")
        if self.table_depth > self.depth - 2 * self.wall_clearance:
            raise ValueError("Table depth violates wall clearance")
        if self.table_height <= self.table_thickness or len(self.wall_materials) != 4:
            raise ValueError("Table support height and four wall materials are required")
        if self.confetti_count < 0:
            raise ValueError("Confetti count must be nonnegative")


@dataclass
class RoomPaths:
    root: str
    table: str
    walls: tuple[str, ...]
    floor: str
    materials: dict[str, str]
    lights: tuple[str, ...]
    confetti: str


def update_material(stage: Usd.Stage, path: str, sample: MaterialSample) -> None:
    """Author a local MDL material and a portable PreviewSurface fallback."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    output.SetRenderType("material")
    material.CreateSurfaceOutput("mdl").ConnectToSource(output)
    values = {
        "diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*sample.color)),
        "diffuse_tint": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*sample.color)),
        "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, sample.roughness),
        "metallic_constant": (Sdf.ValueTypeNames.Float, sample.metallic),
        "specular_level": (Sdf.ValueTypeNames.Float, sample.specular),
        "opacity_constant": (Sdf.ValueTypeNames.Float, sample.opacity),
        "enable_opacity": (Sdf.ValueTypeNames.Bool, sample.opacity < 1),
        "texture_scale": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(*sample.uv_scale)),
    }
    for key, (kind, value) in values.items():
        shader.CreateInput(key, kind).Set(value)
    texture_path = relative_asset_path(stage, sample.texture_path) if sample.texture_path else None
    if texture_path:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    else:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath())
    preview = UsdShade.Shader.Define(stage, path + "/PreviewShader")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*sample.color))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(sample.roughness)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(sample.metallic)
    preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0 + 2.0 * sample.specular)
    preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(sample.opacity)
    material.CreateSurfaceOutput().ConnectToSource(preview.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    if sample.texture_path:
        reader = UsdShade.Shader.Define(stage, path + "/UVReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, path + "/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*sample.color, 1.0))
        preview.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")


def bind_material(stage: Usd.Stage, prim: Usd.Prim, material_path: str) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)))


def author_mesh(
    stage: Usd.Stage,
    path: str,
    points,
    faces,
    *,
    material: str | None = None,
    collision: bool = True,
    convex: bool = False,
) -> UsdGeom.Mesh:
    """Author actual mesh topology and per-face UVs before backend import."""
    mesh = author_mesh_geometry(stage, path, points, faces)
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
            "convexHull" if convex else "none"
        )
    if material:
        bind_material(stage, mesh.GetPrim(), material)
    return mesh


def author_box(stage: Usd.Stage, path: str, size, center, *, material=None, collision=True) -> UsdGeom.Mesh:
    x, y, z = (float(v) / 2 for v in size)
    points = (
        np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z], [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]])
        + center
    )
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return author_mesh(stage, path, points, faces, material=material, collision=collision, convex=True)


def _table_outline(parameters: RoomParameters) -> np.ndarray:
    from isaaclab_hiwonder.randomization import table_outline

    return table_outline(
        parameters.table_shape, parameters.table_width, parameters.table_depth, parameters.table_edge_radius
    ).numpy()


def _author_table(stage: Usd.Stage, path: str, parameters: RoomParameters, material: str):
    outline = _table_outline(parameters)
    n = len(outline)
    bottom, top = parameters.table_height - parameters.table_thickness, parameters.table_height
    points = [(x, y, bottom) for x, y in outline] + [(x, y, top) for x, y in outline]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces.extend((i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n))
    return author_mesh(stage, path, points, faces, material=material, convex=True)


def _author_confetti(stage: Usd.Stage, path: str, parameters: RoomParameters) -> None:
    """Instanced cards have no individual rigid-body simulation cost."""
    rng = np.random.default_rng(parameters.seed)
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    prototype_paths = []
    for i in range(12):
        material_path = f"{path}/Looks/Color{i}"
        update_material(stage, material_path, MaterialSample(color=tuple(rng.uniform(0.1, 0.95, 3)), roughness=0.6))
        proto = f"{path}/Prototypes/Card{i}"
        points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]
        face = [(0, 1, 2)] if i % 3 == 0 else [(0, 1, 2, 3)]
        author_mesh(stage, proto, points, face, material=material_path, collision=False)
        prototype_paths.append(proto)
    instancer.CreatePrototypesRel().SetTargets(prototype_paths)
    count = parameters.confetti_count
    positions = []
    while len(positions) < count:
        x = rng.uniform(-parameters.table_width / 2, parameters.table_width / 2)
        y = rng.uniform(-parameters.table_depth / 2, parameters.table_depth / 2)
        if parameters.table_shape in ("circle", "ellipse"):
            if (2 * x / parameters.table_width) ** 2 + (2 * y / parameters.table_depth) ** 2 > 0.95:
                continue
        positions.append(Gf.Vec3f(float(x), float(y), parameters.table_height + float(rng.uniform(0.0003, 0.002))))
    sizes = rng.uniform(*parameters.confetti_size, count)
    yaws = rng.uniform(-math.pi, math.pi, count)
    instancer.CreatePositionsAttr().Set(positions)
    instancer.CreateProtoIndicesAttr().Set(rng.integers(0, len(prototype_paths), count).tolist())
    instancer.CreateScalesAttr().Set([Gf.Vec3f(float(s), float(s * rng.uniform(0.4, 1.5)), 1.0) for s in sizes])
    instancer.CreateOrientationsAttr().Set([Gf.Quath(math.cos(a / 2), Gf.Vec3h(0, 0, math.sin(a / 2))) for a in yaws])


def author_room(stage: Usd.Stage, room_path: str, parameters: RoomParameters) -> RoomPaths:
    """Author one room without a ceiling or any shared world-space walls/materials."""
    UsdGeom.Xform.Define(stage, room_path)
    materials = {"table": room_path + "/Looks/Table", "floor": room_path + "/Looks/Floor"}
    update_material(stage, materials["table"], parameters.table_material)
    update_material(stage, materials["floor"], parameters.floor_material)
    floor_path = room_path + "/Floor"
    author_box(
        stage, floor_path, (parameters.width, parameters.depth, 0.04), (0, 0, -0.02), material=materials["floor"]
    )
    wall_paths = []
    w, d, h, t = parameters.width, parameters.depth, parameters.height, parameters.wall_thickness
    boxes = [
        ((t, d + 2 * t, h), (-w / 2 - t / 2, 0, h / 2)),
        ((t, d + 2 * t, h), (w / 2 + t / 2, 0, h / 2)),
        ((w, t, h), (0, -d / 2 - t / 2, h / 2)),
        ((w, t, h), (0, d / 2 + t / 2, h / 2)),
    ]
    for i, ((size, center), sample) in enumerate(zip(boxes, parameters.wall_materials, strict=True)):
        material = f"{room_path}/Looks/Wall{i}"
        materials[f"wall_{i}"] = material
        update_material(stage, material, sample)
        wall_path = f"{room_path}/Walls/Wall{i}"
        author_box(stage, wall_path, size, center, material=material)
        wall_paths.append(wall_path)
    table_path = room_path + "/Tabletop"
    _author_table(stage, table_path, parameters, materials["table"])
    for i, (x, y) in enumerate([(-0.35, -0.25), (-0.35, 0.25), (0.35, -0.25), (0.35, 0.25)]):
        x = math.copysign(min(abs(x), parameters.table_width * 0.3), x)
        y = math.copysign(min(abs(y), parameters.table_depth * 0.3), y)
        height = parameters.table_height - parameters.table_thickness
        author_box(
            stage,
            f"{room_path}/TableLegs/Leg{i}",
            (0.035, 0.035, height),
            (x, y, height / 2),
            material=materials["table"],
        )
    lights = []
    env_path = room_path.rsplit("/", 1)[0] or room_path
    for i, position in enumerate(parameters.light_positions):
        light_path = f"{room_path}/Lights/Light{i}"
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateRadiusAttr(0.04)
        light.CreateIntensityAttr(float(parameters.light_intensities[i % len(parameters.light_intensities)]))
        light.CreateColorAttr(Gf.Vec3f(*parameters.light_colors[i % len(parameters.light_colors)]))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(float(parameters.light_beam_angles[i % len(parameters.light_beam_angles)]))
        shaping.CreateShapingConeSoftnessAttr(0.2)
        for collection_name in ("lightLink", "shadowLink"):
            collection = Usd.CollectionAPI.Apply(light.GetPrim(), collection_name)
            collection.CreateIncludeRootAttr(False)
            collection.CreateExpansionRuleAttr("expandPrims")
            collection.CreateIncludesRel().SetTargets([env_path])
        lights.append(light_path)
    confetti = room_path + "/Confetti"
    _author_confetti(stage, confetti, parameters)
    return RoomPaths(room_path, table_path, tuple(wall_paths), floor_path, materials, tuple(lights), confetti)


def save_room(path: str | Path, parameters: RoomParameters) -> RoomPaths:
    """Write a portable source room for heterogeneous scene cloning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    paths = author_room(stage, "/Room", parameters)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Room"))
    stage.GetRootLayer().Save()
    return paths
