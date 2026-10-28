# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import pytest
from isaaclab_hiwonder.sim.profiles import load_profile


def test_profile_rejects_misspelled_distribution(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"randomization": {"actuator": {"stiffnes": [1, 2]}}}))
    with pytest.raises(ValueError, match="stiffnes"):
        load_profile(path)


def test_profile_preserves_defaults_and_nested_group_controls(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "camera": {"width": 640},
                "randomization": {
                    "controls": {"image": {"frozen": True}},
                    "lighting": {"profile": "rgb_disco"},
                    "actuator": {"stiffness": [20, 30]},
                },
            }
        )
    )
    profile = load_profile(path)
    assert profile.camera.height == 480
    assert profile.randomization.actuator.stiffness == (20, 30)
    assert profile.randomization.controls.image.frozen
    assert not profile.randomization.controls.smudge.frozen


def test_environment_profile_rejects_unimplemented_episode_room_mutation(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"randomization": {"controls": {"room": {"cadence": "episode"}}}}))
    with pytest.raises(ValueError, match="room.*fixed|fixed.*room"):
        load_profile(path)
    path.write_text(json.dumps({"randomization": {"controls": {"room": {"cadence": "fixed"}}}}))
    assert load_profile(path).randomization.controls.room.cadence == "fixed"
