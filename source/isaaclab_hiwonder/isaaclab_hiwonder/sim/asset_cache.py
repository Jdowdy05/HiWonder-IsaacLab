# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-run immutable mesh and image assets; mutable scene state stays local."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

from pxr import Gf, Sdf, Usd, UsdGeom

# Include the producer itself and its serialization/numeric dependencies, so a
# changed implementation cannot silently consume assets from an earlier build.
_PRODUCER = hashlib.sha256(
    Path(__file__).read_bytes() + f"{np.__version__}:{pillow_version}:{Usd.GetVersion()}".encode()
).digest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cached_asset(directory: Path, key: str, suffix: str, build: Callable[[], bytes]) -> tuple[Path, bool]:
    """Reuse only bytes with a complete matching receipt; publish files atomically."""
    path = directory / (key + suffix)
    receipt = path.with_name(path.name + ".json")
    try:
        data = path.read_bytes()
        expected = {"key": key, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        if json.loads(receipt.read_text(encoding="utf-8")) == expected:
            return path, False
    except (OSError, ValueError):
        pass
    data = build()
    _atomic_write(path, data)
    _atomic_write(
        receipt,
        json.dumps({"key": key, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}).encode(),
    )
    return path, True


def relative_asset_path(stage: Usd.Stage, path: str) -> str:
    """Anchor assets inside this run to the authoring layer for portable exports."""
    root_path = stage.GetRootLayer().realPath
    layer_path = stage.GetEditTarget().GetLayer().realPath
    source = Path(path)
    if root_path and layer_path and source.is_absolute() and source.resolve().is_relative_to(Path(root_path).parent):
        try:
            return os.path.relpath(source, Path(layer_path).parent).replace("\\", "/")
        except ValueError:
            # Different filesystem drives cannot have a relative asset path.
            pass
    return path


def _set_mesh_geometry(mesh: UsdGeom.Mesh, points: np.ndarray, faces: tuple[tuple[int, ...], ...]) -> None:
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr().Set([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr().Set([int(index) for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr().Set(True)
    coordinates = []
    for face in faces:
        face_points = points[list(face)]
        spans = np.ptp(face_points, axis=0)
        axes = np.argsort(spans)[-2:]
        base = face_points[:, axes].min(axis=0)
        scale = np.maximum(spans[axes], 1e-8)
        coordinates.extend(Gf.Vec2f(*((point[axes] - base) / scale)) for point in face_points)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(coordinates)


def author_mesh_geometry(stage: Usd.Stage, path: str, points, faces) -> UsdGeom.Mesh:
    """Reference identical geometry while leaving physics, bindings and overrides editable."""
    points = np.asarray(points, dtype=float)
    faces = tuple(tuple(int(index) for index in face) for face in faces)
    existing = bool(stage.GetPrimAtPath(path))
    mesh = UsdGeom.Mesh.Define(stage, path)
    layer_path = stage.GetRootLayer().realPath
    if not layer_path or existing:
        # Reauthoring must retain every pre-existing composition arc and its
        # inherited state, including references appended after our first bake.
        _set_mesh_geometry(mesh, points, faces)
        return mesh
    key = hashlib.sha256(_PRODUCER)
    key.update(np.asarray(points.shape, dtype="<i8").tobytes())
    key.update(np.asarray(points, dtype="<f8").tobytes())
    key.update(np.asarray([len(face) for face in faces], dtype="<i8").tobytes())
    key.update(np.asarray([index for face in faces for index in face], dtype="<i8").tobytes())

    def build() -> bytes:
        source = Usd.Stage.CreateInMemory()
        geometry = UsdGeom.Mesh.Define(source, "/Mesh")
        _set_mesh_geometry(geometry, points, faces)
        source.SetDefaultPrim(geometry.GetPrim())
        return source.GetRootLayer().ExportToString().encode()

    asset, replaced = _cached_asset(Path(layer_path).parent / "geometry", key.hexdigest(), ".usda", build)
    cached_layer = Sdf.Layer.Find(str(asset))
    if cached_layer and (replaced or cached_layer.dirty):
        cached_layer.Reload(force=True)
    mesh.GetPrim().GetReferences().AddReference(relative_asset_path(stage, str(asset)), "/Mesh")
    return mesh


def texture_asset(identifier: str, directory: Path, seed: int) -> str | None:
    """Snapshot source bytes or reuse a deterministic RGB texture without re-encoding inputs."""
    if identifier == "none":
        return None
    source = Path(identifier)
    if source.is_file():
        data = source.read_bytes()
        key = hashlib.sha256(_PRODUCER + source.suffix.encode() + data).hexdigest()
        asset, _ = _cached_asset(directory, key, source.suffix, lambda: data)
        return str(asset.resolve())
    if identifier not in ("wood_grain", "checker", "noise"):
        raise ValueError(f"Missing texture/background asset {identifier}")
    # Checker consumes no random values, so its seed does not affect its bytes.
    key = hashlib.sha256(_PRODUCER + f"{identifier}:{seed if identifier != 'checker' else ''}".encode()).hexdigest()

    def build() -> bytes:
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[:128, :128]
        if identifier == "checker":
            pattern = 0.3 + 0.6 * ((x // 16 + y // 16) % 2)
        elif identifier == "wood_grain":
            pattern = 0.45 + 0.25 * np.sin(x * 0.35 + np.sin(y * 0.035) * 2) + rng.normal(0, 0.04, x.shape)
        else:
            pattern = rng.uniform(0.25, 0.95, x.shape)
        rgb = np.repeat(np.clip(pattern[..., None] * 255, 0, 255).astype(np.uint8), 3, axis=-1)
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        return stream.getvalue()

    asset, _ = _cached_asset(directory, key, ".png", build)
    return str(asset.resolve())
