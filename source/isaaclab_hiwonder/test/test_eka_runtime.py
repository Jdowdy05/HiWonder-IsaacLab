# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Model-bound control, reset and contact contracts without launching physics."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from isaaclab_hiwonder.assets.robot_models import get_robot_model
from isaaclab_hiwonder.config import HiWonderEnvConfig
from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.sim.profiles import profile_from_dict


def test_eka_profile_roundtrip_derives_motor_link_shapes_without_changing_stock():
    eka = HiWonderEnvConfig(robot_model="eka_v6")
    assert eka.randomization.actuator.num_joints == 9
    assert eka.randomization.robot.num_links == 10
    restored = profile_from_dict(asdict(eka))
    assert restored == eka
    stock = HiWonderEnvConfig()
    assert stock.robot_model == "stock"
    assert stock.randomization.actuator.num_joints == 6
    assert stock.randomization.robot.num_links == 7
    with pytest.raises(ValueError, match="joint|actuator"):
        replace(
            eka, randomization=replace(eka.randomization, actuator=replace(eka.randomization.actuator, num_joints=8))
        )


def test_eka_profile_checks_its_selected_manifest_scale(monkeypatch, tmp_path):
    from isaaclab_hiwonder import config

    manifest = tmp_path / "manifest.json"
    model = replace(get_robot_model("eka_v6"), manifest_path=manifest)
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"file": "hiwonder_eka_v6.usdc", "scale": 1.0},
                    {"file": "hiwonder_eka_v6_scale099.usdc", "scale": 0.97},
                    {"file": "hiwonder_eka_v6_scale101.usdc", "scale": 1.01},
                ]
            }
        )
    )
    monkeypatch.setattr(config, "get_robot_model", lambda name: model)
    with pytest.raises(ValueError, match="scale|variant|manifest"):
        HiWonderEnvConfig(robot_model="eka_v6")


def test_eka_realized_motor_rows_apply_family_scales_and_indexed_reset_isolation():
    cfg = HiWonderEnvConfig(robot_model="eka_v6")
    cfg = replace(
        cfg,
        randomization=replace(
            cfg.randomization,
            actuator=replace(cfg.randomization.actuator, effort_limit=(1, 1), stiffness=(25, 25), damping=(1, 1)),
        ),
    )
    manager = RandomizationManager(3, cfg.randomization, seed=4, robot_model=cfg.robot_model)
    before = manager.snapshot()
    assert before["actuator"]["effort_limit"].shape == (3, 9)
    assert before["robot"]["mass_scale"].shape == (3, 10)
    torch.testing.assert_close(before["actuator"]["effort_limit"][0], torch.tensor([1.0] * 5 + [0.12] * 4))
    torch.testing.assert_close(before["actuator"]["stiffness"][0], torch.tensor([25.0] * 5 + [0.5] * 4))
    manager.reset([1])
    for group in before:
        for name, old in before[group].items():
            assert torch.equal(old[[0, 2]], manager.snapshot()[group][name][[0, 2]]), (group, name)


def test_distal_pinch_aliases_and_whole_hand_release_are_distinct():
    from isaaclab_hiwonder.sim.contact_filtering import body_contact_aliases

    eka = get_robot_model("eka_v6")
    assert body_contact_aliases(eka, "eka_left_tip") == ("left_finger", "gripper")
    assert body_contact_aliases(eka, "eka_right_tip") == ("right_finger", "gripper")
    assert body_contact_aliases(eka, "eka_left_proximal") == ("gripper",)
    assert body_contact_aliases(eka, "gripper_body") == ("gripper",)
    assert body_contact_aliases(eka, "forearm") == ()
    assert body_contact_aliases("stock", "gripper_body") == ("left_finger",)


def test_grasp_receipt_requires_selected_asset_and_full_joint_states(tmp_path):
    import hashlib

    from isaaclab_hiwonder.sim.resets import load_grasp_library

    asset = tmp_path / "robot.usdc"
    asset.write_bytes(b"new robot")
    model = replace(get_robot_model("eka_v6"), asset_directory=tmp_path, asset_prefix="robot")
    receipt = dict(
        status="PASS_PHYSICS_GRASP",
        hold_seconds=0.6,
        jaw_contacts={"left_finger": True, "right_finger": True},
        robot_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        joint_positions=[0.0] * 6,
        joint_targets=[0.0] * 6,
    )
    path = tmp_path / "grasp.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="nine|9|joint"):
        load_grasp_library(path, robot_model=model)
    receipt.update(joint_positions=[0.0] * 9, joint_targets=[0.0] * 9)
    path.write_text(json.dumps(receipt))
    assert len(load_grasp_library(path, robot_model=model)["joint_positions"]) == 9
    assert len(load_grasp_library(receipt, robot_model=model)["joint_positions"]) == 9
    receipt["joint_positions"][8] = float("nan")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="finite"):
        load_grasp_library(path, robot_model=model)


def test_checkpoint_robot_preflight_rejects_cross_model_even_for_policy_only():
    from isaaclab_hiwonder.sim.robot_contract import validate_checkpoint_robot

    saved = {"hiwonder": {"metadata": {"profile": {"robot_model": "stock"}}, "environment": {}}}
    validate_checkpoint_robot(saved, "stock")
    with pytest.raises(ValueError, match="robot|model"):
        validate_checkpoint_robot(saved, "eka_v6")
    legacy = {"hiwonder": {"metadata": {"profile": {}}, "environment": {}}}
    validate_checkpoint_robot(legacy, "stock")
    with pytest.raises(ValueError, match="robot|model"):
        validate_checkpoint_robot(legacy, "eka_v6")
    malformed = {
        "hiwonder": {
            "metadata": {"profile": {"robot_model": "eka_v6"}, "observation_shapes": {"proprio": [2, 18]}},
            "environment": {},
        }
    }
    with pytest.raises(ValueError, match="shape|proprio|dimension"):
        validate_checkpoint_robot(malformed, "eka_v6")


def test_native_motor_mapping_resolves_every_eka_joint_in_permuted_backend_order():
    from isaaclab_hiwonder.sim.robot_contract import robot_motor_indices

    # Deliberately interleave both environments and reverse every backend joint.
    names = (
        "right_finger_tip",
        "right_finger_root",
        "left_finger_tip",
        "left_finger_root",
        "wrist_roll",
        "wrist_flex",
        "elbow_flex",
        "shoulder_lift",
        "shoulder_pan",
    )
    labels = [f"/World/envs/env_{env}/Room/Robot/joints/{name}" for name in names for env in (1, 0)]
    indices = robot_motor_indices(labels)
    assert indices == [17, 15, 13, 11, 9, 7, 5, 3, 1, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    with pytest.raises(ValueError, match="mapping|topology|joint"):
        robot_motor_indices(labels[:-1])
    with pytest.raises(ValueError, match="duplicate|mapping"):
        robot_motor_indices(labels + [labels[0]])


def test_native_motor_mapping_ignores_the_fixed_root_anchor():
    from isaaclab_hiwonder.sim.robot_contract import robot_motor_indices

    names = ("fixed_base", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    labels = [f"/World/envs/env_0/Room/Robot/joints/{name}" for name in names]
    assert robot_motor_indices(labels) == [1, 2, 3, 4, 5, 6]


def test_scene_sensor_owners_and_native_aliases_cover_tips_and_palm():
    from types import SimpleNamespace

    from isaaclab_hiwonder.sim.contact_filtering import robot_contact_actors
    from isaaclab_hiwonder.tasks import build_task
    from isaaclab_hiwonder.tasks.runtime import TaskRuntime

    paths, aliases = robot_contact_actors("eka_v6", "/Room/Robot")
    assert len(paths) == 5
    assert paths["robot_contact_eka_left_tip"] == "/Room/Robot/eka_left_tip"
    assert aliases["robot_contact_eka_left_tip"] == ("left_finger", "gripper")
    assert aliases["robot_contact_gripper_body"] == ("gripper",)
    room = "/World/envs/env_0/Room"
    model = SimpleNamespace(
        body_label=[room + "/Robot/eka_left_tip", room + "/Robot/gripper_body"],
        shape_body=np.array([0, 1, -1]),
        shape_label=["tip", "palm", room + "/Tabletop/top"],
    )
    runtime = TaskRuntime.__new__(TaskRuntime)
    runtime.robot_model = get_robot_model("eka_v6")
    runtime._contact_topology = None
    mapped, _ = runtime._native_contact_mapping(model, [(0, build_task("cloth_fold"), room + "/Task", {})], {})
    assert mapped == [((0, "left_finger"), (0, "gripper")), ((0, "gripper"),), ((0, "table"),)]


def test_eka_position_witness_and_reset_append_four_open_finger_positions():
    from isaaclab_hiwonder.assets.kinematics import forward_kinematics
    from isaaclab_hiwonder.sim.reachability import solve_position
    from isaaclab_hiwonder.sim.resets import make_reset
    from isaaclab_hiwonder.tasks import build_task

    model = get_robot_model("eka_v6")
    witness = solve_position([0.2, 0, 0.05], robot_model=model)
    assert witness is not None and witness.shape == (9,)
    np.testing.assert_allclose(witness[5:], [-np.pi / 9, 0, -np.pi / 9, 0])
    local = forward_kinematics(witness, robot_model=model)["tool_center"][:3, 3]
    np.testing.assert_allclose([-local[1], local[0], local[2]], [0.2, 0, 0.05], atol=0.004)
    reset = make_reset(build_task("peg_insert"), 0, seed=4, robot_model=model)
    assert reset.robot_q.shape == (9,) and reset.ik_witnesses.shape[1] == 9
    np.testing.assert_allclose(reset.robot_q[5:], [-np.pi / 9, 0, -np.pi / 9, 0])


def test_all_thirty_task_critic_layouts_fit_eka_without_truncation():
    from isaaclab_hiwonder.sim.observations import pack_privileged_rows, privileged_fields
    from isaaclab_hiwonder.tasks import (
        GARMENTS,
        KNOT_VARIANTS,
        TASK_NAMES,
        ActorState,
        SceneState,
        TaskEpisode,
        build_task,
    )
    from isaaclab_hiwonder.tasks.screw_cap import ScrewCap

    cfg = HiWonderEnvConfig(robot_model="eka_v6")
    nuisance = RandomizationManager(1, cfg.randomization, robot_model="eka_v6").snapshot()
    tasks = [name for name in TASK_NAMES if name not in ("cloth_fold", "rope_thread")]
    tasks += ["cloth_fold:" + variant for variant in GARMENTS]
    tasks += ["rope_thread:" + variant for variant in KNOT_VARIANTS]
    tasks += ["bottle_cap:close", "roll_ball:table"]
    assert len(tasks) == 30
    for name in tasks + ["cube_stack:six"]:
        task_name, _, variant = name.partition(":")
        task = build_task(
            task_name,
            variant if variant and variant != "six" else None,
            task_parameters={"count": 6} if variant == "six" else None,
        )
        state = SceneState(
            actors=dict(task.initial),
            joint_positions={joint.name: 0.0 for joint in task.joints},
            joint_velocities={joint.name: 0.0 for joint in task.joints},
        )
        for spec in task.deformables:
            state.deformable_positions[spec.name] = spec.vertices.copy()
            if spec.kind == "cable":
                state.actors[spec.name + "/segment_0"] = ActorState(spec.vertices[0])
            if spec.kind == "plastic_shell":
                state.crease_rest_angles[spec.name] = np.zeros(3)
        if task.name == "bottle_cap":
            ScrewCap(task.parameters).observe(state)
        robot = torch.cat((torch.arange(9.0), torch.zeros(9), torch.tensor([0.0, 0, 0, 0, 0, 0, 1]), torch.zeros(1)))
        row = privileged_fields(robot, 9, torch.zeros(40), TaskEpisode(task), nuisance, state, 0)
        critic, layout = pack_privileged_rows([row], capacity=320)
        assert critic.shape == (1, 320) and torch.isfinite(critic).all(), name
        assert layout[0]["robot/joint_position"] == (0, 9)
        torch.testing.assert_close(critic[0, :9], torch.arange(9.0))
        if name == "shape_sort":
            assert layout[0]["contact/count"][1] == 267
            with pytest.raises(ValueError, match="capacity"):
                pack_privileged_rows([row], capacity=256)
