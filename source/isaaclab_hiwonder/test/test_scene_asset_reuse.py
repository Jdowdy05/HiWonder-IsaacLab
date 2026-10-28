# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable scene data reuse preserves composed geometry and local ownership."""

import hashlib
import os
import shutil
from pathlib import Path

import pytest
from isaaclab_hiwonder.sim.rooms import MaterialSample, author_mesh, update_material
from isaaclab_hiwonder.sim.scene_config import _texture
from PIL import Image, PngImagePlugin

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

POINTS = [(-1, -2, 0), (1, -2, 0), (1, 2, 0), (-1, 2, 0)]
FACES = [(0, 1, 2, 3)]


def geometry_source(mesh):
    """Find the real contributing geometry layer through composed USD specs."""
    root = mesh.GetPrim().GetStage().GetRootLayer()
    layers = {spec.layer.realPath for spec in mesh.GetPointsAttr().GetPropertyStack() if spec.layer != root}
    assert len(layers) == 1, "Identical mesh data must be held in one reusable source layer"
    return Path(layers.pop())


def test_reused_geometry_preserves_local_transform_material_and_physics_ownership(tmp_path):
    # A shared material, instanceable physics prim, or geometry-only source with
    # local attributes removed would break this independently editable USD scene.
    stage = Usd.Stage.CreateNew(str(tmp_path / "room.usda"))
    for name, color in (("A", (1, 0, 0)), ("B", (0, 0, 1))):
        update_material(stage, f"/Room/Looks/{name}", MaterialSample(color=color))
    first = author_mesh(stage, "/Room/A", POINTS, FACES, material="/Room/Looks/A", convex=True)
    second = author_mesh(stage, "/Room/B", POINTS, FACES, material="/Room/Looks/B", collision=False)
    first.AddTranslateOp().Set(Gf.Vec3d(2, 0, 1))
    UsdPhysics.RigidBodyAPI.Apply(first.GetPrim())
    UsdPhysics.MassAPI.Apply(first.GetPrim()).CreateMassAttr(0.3)
    assert geometry_source(first) == geometry_source(second)
    for mesh in (first, second):
        assert mesh.GetPrim().GetTypeName() == "Mesh"
        assert not mesh.GetPrim().IsInstanceable()
        assert not mesh.GetPrim().IsInstance()
        assert list(map(tuple, mesh.GetPointsAttr().Get())) == POINTS
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [4]
        assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 3]
        assert mesh.GetSubdivisionSchemeAttr().Get() == "none"
        assert mesh.GetDoubleSidedAttr().Get() is True
        assert mesh.GetNormalsAttr().Get() is None
        uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        assert uv.GetInterpolation() == "faceVarying"
        assert list(map(tuple, uv.Get())) == [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert tuple(UsdGeom.Xformable(first).GetLocalTransformation().ExtractTranslation()) == (2, 0, 1)
    assert tuple(UsdGeom.Xformable(second).GetLocalTransformation().ExtractTranslation()) == (0, 0, 0)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(first.GetPrim())
    assert tuple(bounds.ComputeAlignedRange().GetMin()) == (1, -2, 1)
    assert tuple(bounds.ComputeAlignedRange().GetMax()) == (3, 2, 1)
    assert first.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert first.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MeshCollisionAPI(first).GetApproximationAttr().Get() == "convexHull"
    assert UsdPhysics.MassAPI(first).GetMassAttr().Get() == pytest.approx(0.3)
    assert not second.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert not second.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    update_material(stage, "/Room/Looks/A", MaterialSample(color=(0, 1, 0)))
    assert tuple(UsdShade.Shader(stage.GetPrimAtPath("/Room/Looks/B/Shader")).GetInput("diffuse_tint").Get()) == (
        0,
        0,
        1,
    )
    for mesh, material in ((first, "/Room/Looks/A"), (second, "/Room/Looks/B")):
        bound, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        assert str(bound.GetPath()) == material
    first.GetPointsAttr().Set([Gf.Vec3f(9, 9, 9)] * 4)
    assert list(map(tuple, second.GetPointsAttr().Get())) == POINTS


@pytest.mark.parametrize("cached_first", [False, True], ids=["existing-source", "cache-with-appended-source"])
def test_reauthoring_existing_mesh_preserves_reference_owned_physics_transform_and_material(tmp_path, cached_first):
    # Replacing the reference list would erase inherited physics, materials and
    # user attributes. A previously cached mesh can gain such references later.
    source_path = tmp_path / "source.usda"
    source = Usd.Stage.CreateNew(str(source_path))
    profile = UsdGeom.Mesh.Define(source, "/Profile")
    profile.CreatePointsAttr().Set([Gf.Vec3f(0, 0, 0)] * 4)
    profile.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.5, 0.6))
    UsdPhysics.MassAPI.Apply(profile.GetPrim()).CreateMassAttr(0.7)
    UsdPhysics.CollisionAPI.Apply(profile.GetPrim()).CreateCollisionEnabledAttr(False)
    profile.GetPrim().CreateAttribute("sourceNote", Sdf.ValueTypeNames.String).Set("source-owned")
    material = UsdShade.Material.Define(source, "/Profile/Looks/Surface")
    UsdShade.MaterialBindingAPI.Apply(profile.GetPrim()).Bind(material)
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / "room.usda"))
    mesh = (
        author_mesh(stage, "/Room/Mesh", POINTS, FACES, collision=False)
        if cached_first
        else UsdGeom.Mesh.Define(stage, "/Room/Mesh")
    )
    mesh.GetPrim().GetReferences().AddReference(str(source_path), "/Profile")
    reference_assets = mesh.GetPrim().GetMetadata("references").GetAddedOrExplicitItems()
    result = author_mesh(stage, "/Room/Mesh", POINTS, [(3, 2, 1, 0)], collision=False)
    assert result.GetPrim().HasAPI(UsdPhysics.MassAPI)
    assert UsdPhysics.MassAPI(result).GetMassAttr().Get() == pytest.approx(0.7)
    assert result.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.CollisionAPI(result).GetCollisionEnabledAttr().Get() is False
    assert result.GetPrim().GetAttribute("sourceNote").Get() == "source-owned"
    assert tuple(result.GetLocalTransformation().ExtractTranslation()) == (0.4, 0.5, 0.6)
    bound, _ = UsdShade.MaterialBindingAPI(result).ComputeBoundMaterial()
    assert str(bound.GetPath()) == "/Room/Mesh/Looks/Surface"
    assert result.GetPrim().GetMetadata("references").GetAddedOrExplicitItems() == reference_assets
    assert list(map(tuple, result.GetPointsAttr().Get())) == POINTS
    assert list(result.GetFaceVertexIndicesAttr().Get()) == [3, 2, 1, 0]


def test_reauthoring_keeps_authored_geometry_time_samples_and_metadata(tmp_path):
    # Setting a new default geometry value must retain authored animation and
    # attribute metadata; removing/recreating properties silently discards both.
    stage = Usd.Stage.CreateNew(str(tmp_path / "room.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Room/Mesh")
    points = mesh.CreatePointsAttr()
    points.Set([Gf.Vec3f(x, y, 2) for x, y, _ in POINTS], Usd.TimeCode(1))
    points.SetDocumentation("measured clip geometry")
    points.SetCustomDataByKey("sourceTag", "measured")
    mesh.CreateFaceVertexIndicesAttr().Set([0, 2, 1, 3], Usd.TimeCode(1))
    result = author_mesh(stage, "/Room/Mesh", POINTS, FACES, collision=False)
    assert result.GetPointsAttr().GetTimeSamples() == [1]
    assert list(map(tuple, result.GetPointsAttr().Get(Usd.TimeCode(1)))) == [
        (-1, -2, 2),
        (1, -2, 2),
        (1, 2, 2),
        (-1, 2, 2),
    ]
    assert result.GetPointsAttr().GetDocumentation() == "measured clip geometry"
    assert result.GetPointsAttr().GetCustomDataByKey("sourceTag") == "measured"
    assert list(result.GetFaceVertexIndicesAttr().Get(Usd.TimeCode(1))) == [0, 2, 1, 3]
    assert list(map(tuple, result.GetPointsAttr().Get())) == POINTS
    assert list(result.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 3]


@pytest.mark.parametrize("edit_file", [None, "sub/edit.usda"], ids=["root-layer", "nested-edit-layer"])
def test_geometry_inputs_select_distinct_assets_and_folder_moves_keep_references_valid(tmp_path, edit_file):
    # Reusing a different winding/point array or writing absolute cache references
    # would silently change geometry or break an archived run after relocation.
    original = tmp_path / "original"
    original.mkdir()
    stage = Usd.Stage.CreateNew(str(original / "room.usda"))
    if edit_file:
        (original / edit_file).parent.mkdir()
        edit = Sdf.Layer.CreateNew(str(original / edit_file))
        stage.GetRootLayer().subLayerPaths.append(edit_file)
        stage.SetEditTarget(edit)
    texture = _texture("checker", original / "textures", 17)
    update_material(stage, "/Room/Looks/Table", MaterialSample(texture_path=texture))
    external = tmp_path / "external.png"
    Image.new("RGB", (2, 3), (8, 19, 27)).save(external)
    update_material(stage, "/Room/Looks/External", MaterialSample(texture_path=str(external)))
    first = author_mesh(stage, "/Room/A", POINTS, FACES)
    flipped = author_mesh(stage, "/Room/B", POINTS, [(3, 2, 1, 0)])
    raised = author_mesh(stage, "/Room/C", [(x, y, 1) for x, y, _ in POINTS], FACES)
    assert len({geometry_source(mesh) for mesh in (first, flipped, raised)}) == 3
    stage.GetEditTarget().GetLayer().Save()
    stage.GetRootLayer().Save()
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    stage = Usd.Stage.Open(str(moved / "room.usda"))
    for path in ("/Room/A", "/Room/B", "/Room/C"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        assert geometry_source(mesh).is_relative_to(moved)
    texture_asset = UsdShade.Shader(stage.GetPrimAtPath("/Room/Looks/Table/Shader")).GetInput("diffuse_texture").Get()
    assert not Path(texture_asset.path).is_absolute()
    assert Path(texture_asset.resolvedPath).is_relative_to(moved)
    assert Path(texture_asset.resolvedPath).read_bytes() == Path(texture).read_bytes()
    external_asset = (
        UsdShade.Shader(stage.GetPrimAtPath("/Room/Looks/External/Shader")).GetInput("diffuse_texture").Get()
    )
    assert Path(external_asset.path) == external
    assert Path(external_asset.resolvedPath).read_bytes() == external.read_bytes()
    assert list(stage.GetPrimAtPath("/Room/B").GetAttribute("faceVertexIndices").Get()) == [3, 2, 1, 0]
    assert list(map(tuple, stage.GetPrimAtPath("/Room/C").GetAttribute("points").Get())) == [
        (-1, -2, 1),
        (1, -2, 1),
        (1, 2, 1),
        (-1, 2, 1),
    ]


def test_anonymous_edit_layer_keeps_resolvable_assets_without_a_filesystem_anchor(tmp_path):
    # A saved root does not give an anonymous session layer a valid relative
    # anchor. Its geometry and texture references must still resolve correctly.
    stage = Usd.Stage.CreateNew(str(tmp_path / "room.usda"))
    stage.SetEditTarget(stage.GetSessionLayer())
    texture = _texture("checker", tmp_path / "textures", 17)
    update_material(stage, "/Room/Looks/Table", MaterialSample(texture_path=texture))
    mesh = author_mesh(stage, "/Room/Mesh", POINTS, FACES)
    source = geometry_source(mesh)
    reference = mesh.GetPrim().GetMetadata("references").GetAddedOrExplicitItems()[0]
    assert Path(reference.assetPath) == source
    assert list(map(tuple, mesh.GetPointsAttr().Get())) == POINTS
    asset = UsdShade.Shader(stage.GetPrimAtPath("/Room/Looks/Table/Shader")).GetInput("diffuse_texture").Get()
    assert Path(asset.path) == Path(texture)
    assert Path(asset.resolvedPath).read_bytes() == Path(texture).read_bytes()


def test_corrupt_geometry_asset_is_repaired_before_a_new_room_references_it(tmp_path):
    # Accepting a partial or changed cache file would poison all future rooms.
    stage = Usd.Stage.CreateNew(str(tmp_path / "first.usda"))
    source = geometry_source(author_mesh(stage, "/Room/A", POINTS, FACES))
    expected = source.read_bytes()
    source.write_bytes(b"x" * len(expected))
    other = Usd.Stage.CreateNew(str(tmp_path / "second.usda"))
    mesh = author_mesh(other, "/Room/A", POINTS, FACES)
    assert geometry_source(mesh) == source
    assert source.read_bytes() == expected
    assert list(map(tuple, mesh.GetPointsAttr().Get())) == POINTS
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("save_source", [False, True])
def test_repair_reloads_an_already_open_geometry_source_instead_of_using_stale_composition(tmp_path, save_source):
    # USD retains source layers across stages. Repairing only disk bytes would
    # leave a previously loaded, accidentally edited source active in new rooms.
    stage = Usd.Stage.CreateNew(str(tmp_path / "first.usda"))
    source = geometry_source(author_mesh(stage, "/Room/A", POINTS, FACES))
    asset = Usd.Stage.Open(str(source))
    UsdGeom.Mesh(asset.GetDefaultPrim()).GetPointsAttr().Set([Gf.Vec3f(9, 9, 9)] * 4)
    if save_source:
        asset.GetRootLayer().Save()
    other = Usd.Stage.CreateNew(str(tmp_path / "second.usda"))
    mesh = author_mesh(other, "/Room/A", POINTS, FACES)
    assert list(map(tuple, mesh.GetPointsAttr().Get())) == POINTS


@pytest.mark.parametrize(
    ("identifier", "pixel_sha256"),
    [
        ("checker", "f264eff9e140e0383e20180a7f43e6b47d613b6d563b89514a3914ca9e7ae0ee"),
        ("wood_grain", "d3f5a10d2cebfd75dbaac3a2da6fb4d4089c55834e13c4c117ead27cf6be0905"),
        ("noise", "2d98d140492a63a8f60f94bbd6ad1e4f8da1a3317aee9c32b782ac2565406e63"),
    ],
)
def test_procedural_texture_reuse_preserves_baseline_pixels_and_repairs_partial_files(
    tmp_path, identifier, pixel_sha256
):
    # Fixed pixel receipts were collected before the change. Regeneration on a
    # valid hit, altered seeds, compression-only checks, or partial reuse fail.
    path = Path(_texture(identifier, tmp_path, 17))
    original = path.read_bytes()
    with Image.open(path) as image:
        assert image.mode == "RGB" and image.size == (128, 128)
        assert hashlib.sha256(image.tobytes()).hexdigest() == pixel_sha256
    path.touch()
    modified = path.stat().st_mtime_ns
    assert Path(_texture(identifier, tmp_path, 17)) == path
    assert path.stat().st_mtime_ns == modified
    other_seed = Path(_texture(identifier, tmp_path, 18))
    if identifier == "checker":
        assert other_seed == path
    else:
        assert other_seed != path
        assert other_seed.read_bytes() != original
    path.write_bytes(b"partial")
    assert Path(_texture(identifier, tmp_path, 17)) == path
    assert path.read_bytes() == original
    other = Path(_texture(identifier, tmp_path / "other", 17))
    assert other.parent == tmp_path / "other"
    assert other.read_bytes() == original


def test_external_texture_snapshot_preserves_file_bytes_and_invalidates_changed_source(tmp_path):
    # Caching by path/mtime or re-encoding a supplied image would return stale
    # pixels or discard metadata/alpha that the renderer needs.
    source = tmp_path / "camera.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("profile-note", "keep bytes and alpha")
    Image.new("RGBA", (2, 3), (8, 19, 27, 43)).save(source, pnginfo=metadata)
    original = source.read_bytes()
    original_stat = source.stat()
    output = tmp_path / "assets"
    first = Path(_texture(str(source), output, 0))
    assert first.parent == output
    assert first.suffix == ".png"
    assert first.read_bytes() == original
    Image.new("RGBA", (2, 3), (31, 59, 83, 43)).save(source, pnginfo=metadata)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = Path(_texture(str(source), output, 0))
    assert second != first
    assert second.read_bytes() == source.read_bytes()
    assert first.read_bytes() == original
