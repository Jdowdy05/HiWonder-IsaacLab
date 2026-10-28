# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""USD frame/units boundary: catch disconnected pivots after an asset edit.

No existing upstream test owns the user-supplied HiWonder registration. The
negative control moves a body while preserving file integrity to ensure that
the check detects physical frame disagreement rather than a hash discrepancy.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pxr import Gf, Usd, UsdGeom

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("asset_validator", ROOT / "tools" / "validate_hiwonder_asset.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def test_asset_frames_and_disconnected_pivot_negative_control(tmp_path):
    assets = ROOT / "isaaclab_hiwonder" / "assets" / "generated"
    manifest = json.loads((assets / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        result = validator.validate_asset(assets / artifact["file"], assets / "manifest.json")
        assert result["maximum_joint_frame_error_m"] < 1e-7
        assert result["camera_lens_error_m"] < 1e-7
    target = tmp_path / "hiwonder_soarm101.usdc"
    target.write_bytes((assets / target.name).read_bytes())
    stage = Usd.Stage.Open(str(target))
    body = UsdGeom.Xform(stage.GetPrimAtPath("/HiWonder/upper_arm"))
    operation = body.GetOrderedXformOps()[0]
    operation.Set(operation.Get() + Gf.Vec3d(0.007, 0, 0))
    stage.GetRootLayer().Save()
    manifest["artifacts"][0]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Disconnected joint frame"):
        validator.validate_asset(target, manifest_path)
