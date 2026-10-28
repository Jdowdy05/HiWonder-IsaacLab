# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Selected-model validator rejects wrong ownership, disconnected fingers and bad inertials."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_hiwonder.assets.robot_models import get_robot_model

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("selected_validator", ROOT / "tools/validate_hiwonder_asset.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.mark.parametrize("variant", ("nominal", "scale099", "scale101"))
def test_explicit_eka_validation_preserves_topology_camera_and_mass_guards(variant):
    model = get_robot_model("eka_v6")
    result = validator.validate_asset(model.asset_path(variant), model.manifest_path, robot_model="eka_v6")
    assert result["rigid_bodies"] == 10 and result["revolute_joints"] == 9
    assert result["maximum_joint_frame_error_m"] < 1e-7
    assert result["camera_lens_error_m"] < 1e-7
    assert result["maximum_component_bound_change_mm"] < 2
    assert result["physics_executed"] is False


def test_eka_asset_and_manifest_require_explicit_matching_model():
    model = get_robot_model("eka_v6")
    with pytest.raises(ValueError, match="model|manifest"):
        validator.validate_asset(model.asset_path(), model.manifest_path)
    stock = get_robot_model("stock")
    with pytest.raises(ValueError, match="model|manifest"):
        validator.validate_asset(model.asset_path(), stock.manifest_path, robot_model="eka_v6")


@pytest.mark.parametrize("damage", ("pivot", "mass", "visual"))
def test_rehashed_eka_mutation_still_fails_physical_registration(tmp_path, damage):
    model = get_robot_model("eka_v6")
    target = tmp_path / model.asset_path().name
    target.write_bytes(model.asset_path().read_bytes())
    stage = Usd.Stage.Open(str(target))
    tip = stage.GetPrimAtPath("/HiWonder/eka_left_tip")
    if damage == "pivot":
        operation = UsdGeom.Xform(tip).GetOrderedXformOps()[0]
        operation.Set(operation.Get() + Gf.Vec3d(0.007, 0, 0))
        message = "Disconnected joint frame"
    elif damage == "mass":
        UsdPhysics.MassAPI(tip).GetMassAttr().Set(-0.01)
        message = "mass|inertia"
    else:
        visual = UsdGeom.Xformable(stage.GetPrimAtPath("/HiWonder/eka_left_tip/visuals/eka_12"))
        visual.AddTranslateOp().Set(Gf.Vec3d(0.003, 0, 0))
        message = "component bound"
    stage.GetRootLayer().Save()
    manifest = json.loads(model.manifest_path.read_text())
    next(row for row in manifest["artifacts"] if row["file"] == target.name)["sha256"] = hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        validator.validate_asset(target, manifest_path, robot_model="eka_v6")
