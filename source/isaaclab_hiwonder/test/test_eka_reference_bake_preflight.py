# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Candidate/source/cache preflight only; never import or cook a Newton model."""

import gzip
import hashlib
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from pxr import Usd, UsdPhysics

from isaaclab_hiwonder.assets.robot_models import get_robot_model

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("reference_baker", ROOT / "tools/bake_eka_collision_references.py")
baker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baker)


@pytest.fixture
def candidate(tmp_path):
    model = get_robot_model("eka_v6")
    target = tmp_path / "candidate"
    target.mkdir()
    names = [model.asset_path(v).name for v in model.geometry_variants]
    names += ["manifest.json", "source.step", "source_geometry.npz", "precision_caps.npz"]
    for name in names:
        shutil.copy2(model.asset_directory / name, target / name)
    return target


def test_all_shipped_candidates_preflight_before_any_outputs_and_receipt_cannot_be_overwritten(candidate):
    model, expected, inherited = baker.preflight(candidate)
    assert model.asset_directory == candidate.resolve()
    assert model.manifest_path == candidate.resolve() / "manifest.json"
    assert set(expected) == {"nominal", "scale099", "scale101"}
    hand = {"gripper_body", "eka_left_proximal", "eka_left_tip", "eka_right_proximal", "eka_right_tip"}
    assert all({path.split("/")[1] for path in paths} == hand and paths for paths in expected.values())
    assert all(groups for groups in inherited.values())
    assert not list(candidate.glob("collision_reference_*.json.gz"))
    (candidate / "reference-bake-receipt.json").write_text("previous receipt")
    with pytest.raises(FileExistsError, match="receipt|existing|replace"):
        baker.preflight(candidate)
    assert (candidate / "reference-bake-receipt.json").read_text() == "previous receipt"


@pytest.mark.parametrize("damage", ("source", "root_model", "collider"))
def test_rehashed_candidate_cannot_replace_registered_source_or_omit_hand_collision(candidate, damage):
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = candidate / get_robot_model("eka_v6").asset_path("scale101").name
    if damage == "source":
        (candidate / "source.step").write_bytes(b"different STEP with a self-consistent candidate hash")
        manifest["source"]["sha256"] = hashlib.sha256((candidate / "source.step").read_bytes()).hexdigest()
        message = "source|STEP"
    else:
        stage = Usd.Stage.Open(str(path))
        if damage == "root_model":
            stage.GetDefaultPrim().SetCustomDataByKey("hiwonder:robotModel", "stock")
            message = "model|root"
        else:
            prim = stage.GetPrimAtPath("/HiWonder/eka_left_tip/collisions/eka_12")
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            message = "decomposition|collider|hand"
        stage.GetRootLayer().Save()
        next(row for row in manifest["artifacts"] if row["file"] == path.name)["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        baker.preflight(candidate)
    assert not list(candidate.glob("collision_reference_*.json.gz"))
    assert not (candidate / "reference-bake-receipt.json").exists()


def test_inherited_stock_reader_rejects_corrupt_cache_envelope(monkeypatch, tmp_path):
    stock = get_robot_model("stock")
    shutil.copy2(stock.asset_path(), tmp_path / stock.asset_path().name)
    with gzip.open(stock.asset_directory / "collision_reference_nominal.json.gz", "rb") as stream:
        envelope = json.load(stream)
    envelope["payload_sha256"] = "0" * 64
    with gzip.open(tmp_path / "collision_reference_nominal.json.gz", "wb") as stream:
        stream.write(json.dumps(envelope).encode())
    monkeypatch.setattr(baker, "STOCK", replace(stock, asset_directory=tmp_path))
    with pytest.raises(ValueError, match="checksum|identity"):
        baker.read_stock("nominal")


def test_post_cook_guard_cannot_certify_empty_or_incomplete_hand_groups():
    paths = {
        f"/{body}/collisions/eka_{i}"
        for i, body in enumerate(
            ("gripper_body", "eka_left_proximal", "eka_left_tip", "eka_right_proximal", "eka_right_tip")
        )
    }
    with pytest.raises(RuntimeError, match="hand|component|group"):
        baker.validate_cooked_groups([], paths)
    groups = [{"original": path, "parts": [{"body": "/" + path.split("/")[1]}]} for path in sorted(paths)]
    baker.validate_cooked_groups(groups, paths)
    with pytest.raises(RuntimeError, match="hand|component|group"):
        baker.validate_cooked_groups(groups[:-1], paths)
