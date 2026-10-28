# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real USD contracts for the cached physical light adapter (no Kit or GPU)."""

import importlib
import math

import pytest
import torch
from isaaclab_hiwonder.config import CameraProfile, LightingConfig

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics


@pytest.fixture
def adapter():
    name = "isaaclab_hiwonder.sim.lighting"
    assert importlib.util.find_spec(name) is not None, "Physical light rig module is missing"
    return importlib.import_module(name)


@pytest.fixture
def scene(adapter):
    stage = Usd.Stage.CreateInMemory()
    root = "/World/envs/env_0/Room"
    UsdGeom.Xform.Define(stage, root)
    light = UsdLux.SphereLight.Define(stage, root + "/Lights/Light0")
    light.CreateRadiusAttr(0.04)
    light.CreateIntensityAttr(300)
    light.CreateColorAttr(Gf.Vec3f(1))
    light.CreateEnableColorTemperatureAttr(True)
    light.CreateColorTemperatureAttr(3500)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
    fill = adapter.author_fill_light(stage, root, (0, 0, 2.5), 17)
    rigid = UsdGeom.Cube.Define(stage, root + "/Body")
    UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(rigid.GetPrim()).CreateMassAttr(0.6)
    camera = UsdGeom.Camera.Define(stage, root + "/Camera")
    camera.CreateFocalLengthAttr(2.1)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10))
    return stage, light, fill


def room():
    return {
        "light_position": torch.tensor([[[0.0, 0, 2]]]),
        "light_direction": torch.tensor([[[0.6, 0, -0.8]]]),
        "light_color": torch.tensor([[[0.2, 0.3, 0.4]]]),
        "light_intensity": torch.tensor([[300.0]]),
        "light_active": torch.tensor([[True]]),
        "light_beam_angle_degrees": torch.tensor([[27.0]]),
        "light_phase": torch.tensor([[0.0]]),
        "light_angular_speed": torch.tensor([[1.0]]),
        "elapsed_seconds": torch.tensor([0.0], dtype=torch.float64),
        "light_saturation": torch.tensor([[1.0]]),
        "light_spot_radius_m": torch.tensor([[0.1]]),
        "light_edge_softness": torch.tensor([[0.3]]),
        "light_view_extent": torch.tensor([[0.0]]),
        "white_fill_intensity": torch.tensor([17.0]),
    }


def rig(adapter, scene, mode="wrist_view", frozen=False):
    stage, light, fill = scene
    return adapter.SceneLightingRig(
        stage,
        [{"lights": [str(light.GetPath())]}],
        [fill],
        LightingConfig(profile="rgb_disco", motion_mode=mode),
        CameraProfile(),
        torch.tensor([[0.0, 0, 0]]),
        torch.tensor([0.0]),
        frozen=frozen,
    )


def down_camera(x=0.0):
    return torch.tensor([[x, 0, 1.0]]), torch.tensor([[1.0, 0, 0, 0]])


def color(light):
    return tuple(light.GetColorAttr().Get())


def test_fill_is_white_unshaped_and_links_only_its_room_environment(adapter, scene):
    stage, _, path = scene
    fill = UsdLux.SphereLight(stage.GetPrimAtPath(path))
    assert color(fill) == (1, 1, 1)
    assert fill.GetRadiusAttr().Get() == pytest.approx(0.25)
    assert fill.GetNormalizeAttr().Get() is True
    assert not fill.GetPrim().HasAPI(UsdLux.ShapingAPI)
    for name in ("lightLink", "shadowLink"):
        collection = Usd.CollectionAPI(fill, name)
        query = collection.ComputeMembershipQuery()
        assert query.IsPathIncluded("/World/envs/env_0/Room/Body")
        assert not query.IsPathIncluded("/World/envs/env_1/Room/Body")
        assert collection.GetIncludeRootAttr().Get() is False


def test_wrist_aim_uses_optical_pose_and_transverse_radius_without_touching_protected_prims(adapter, scene):
    stage, light, _ = scene
    protected = {
        str(p.GetPath()): str(p) + str([(str(a.GetName()), a.Get()) for a in p.GetAttributes()])
        for p in (
            stage.GetPrimAtPath("/World/envs/env_0/Room/Body"),
            stage.GetPrimAtPath("/World/envs/env_0/Room/Camera"),
        )
    }
    controller = rig(adapter, scene)
    controller.update(room(), down_camera(1.0))
    transform = UsdGeom.Xformable(light).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    direction = transform.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
    assert tuple(direction) == pytest.approx((1 / math.sqrt(5), 0, -2 / math.sqrt(5)), abs=1e-6)
    shaping = UsdLux.ShapingAPI(light)
    assert shaping.GetShapingConeAngleAttr().Get() == pytest.approx(2.560638973, abs=1e-5)
    assert shaping.GetShapingConeSoftnessAttr().Get() == pytest.approx(0.3)
    assert light.GetPrim().GetAttribute("omni:rtx:usdluxVersion").Get() == 2505
    assert type(light.GetPrim().GetAttribute("omni:rtx:usdluxVersion").Get()) is int
    assert light.GetEnableColorTemperatureAttr().Get() is False
    assert light.GetRadiusAttr().Get() == pytest.approx(0.004)
    assert controller.last_parameters["visible_ray_mask"].tolist() == [[True]]
    assert controller.last_parameters["target_positions"].tolist() == [[[1, 0, 0]]]
    for path, before in protected.items():
        prim = stage.GetPrimAtPath(path)
        assert str(prim) + str([(str(a.GetName()), a.Get()) for a in prim.GetAttributes()]) == before


def test_room_sweep_keeps_sampled_color_direction_and_unshaped_sphere(adapter, scene):
    _, light, _ = scene
    controller = rig(adapter, scene, "room_sweep")
    controller.update(room())
    assert not light.GetPrim().HasAPI(UsdLux.ShapingAPI)
    assert not light.GetPrim().HasAttribute("omni:rtx:usdluxVersion")
    assert light.GetEnableColorTemperatureAttr().Get() is True
    assert light.GetColorTemperatureAttr().Get() == 3500
    assert color(light) == pytest.approx((0.2, 0.3, 0.4))
    assert light.GetRadiusAttr().Get() == pytest.approx(0.04)
    transform = UsdGeom.Xformable(light).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert tuple(transform.TransformDir(Gf.Vec3d(0, 0, -1))) == pytest.approx((0.6, 0, -0.8), abs=1e-6)


def test_preview_saturation_power_and_fill_are_independent_and_atomic(adapter, scene):
    stage, light, path = scene
    controller = rig(adapter, scene)
    controller.set_preview({"saturation": 0, "power": 0, "fill_intensity": 29})
    controller.update(room(), down_camera())
    assert color(light) == (1, 1, 1)
    assert light.GetIntensityAttr().Get() == 0
    assert UsdLux.SphereLight(stage.GetPrimAtPath(path)).GetIntensityAttr().Get() == 29
    with pytest.raises(ValueError):
        controller.set_preview({"power": 2, "saturation": -1})
    assert controller.state_dict()["overrides"]["power"] == 0
    controller.set_preview({"saturation": 1, "power": 2})
    controller.update(room(), down_camera())
    assert color(light) == (1, 0, 0)
    assert light.GetIntensityAttr().Get() == 600
    sample = room()
    sample["light_active"][:] = False
    controller.update(sample, down_camera())
    assert light.GetIntensityAttr().Get() == 0


def test_speed_change_preserves_phase_then_scales_future_color_and_motion(adapter, scene):
    _, light, _ = scene
    controller = rig(adapter, scene)
    sample = room()
    sample["light_view_extent"][:] = 0.5
    sample["elapsed_seconds"][:] = 1
    controller.set_preview({"speed": 1})
    controller.update(sample, down_camera())
    before = color(light)
    target = controller.last_parameters["target_positions"]
    controller.set_preview({"speed": 0})
    controller.update(sample, down_camera())
    assert color(light) == before
    torch.testing.assert_close(controller.last_parameters["target_positions"], target)
    sample["elapsed_seconds"][:] = 8
    controller.update(sample, down_camera())
    assert color(light) == before
    torch.testing.assert_close(controller.last_parameters["target_positions"], target)
    controller.set_preview({"speed": 2})
    sample["elapsed_seconds"][:] = 8.5
    controller.update(sample, down_camera())
    # HSV angle 2 rad: green with a small red component.
    assert color(light) == pytest.approx((0.090140683, 1, 0), abs=1e-6)
    assert not torch.equal(controller.last_parameters["target_positions"], target)


def test_checkpoint_restores_usd_immediately_and_owns_copies_and_future_clock(adapter, scene):
    stage, light, path = scene
    controller = rig(adapter, scene)
    sample = room()
    controller.set_preview({"saturation": 1, "speed": 0.5, "fill_intensity": 23})
    controller.update(sample, down_camera(0.2))
    saved = controller.state_dict()
    expected_layer = stage.GetRootLayer().ExportToString()
    # Checkpoint replay owns the recorded appearance even when a GUI/external
    # USD editor changed the live attributes without changing the rig cache.
    light.GetColorAttr().Set(Gf.Vec3f(0, 0, 1))
    light.GetIntensityAttr().Set(999)
    controller.load_state_dict(saved)
    assert stage.GetRootLayer().ExportToString() == expected_layer
    sample["elapsed_seconds"][:] = 1
    controller.update(sample, down_camera(0.7))
    future_color = color(light)
    controller.load_state_dict(saved)
    assert stage.GetRootLayer().ExportToString() == expected_layer
    saved["parameters"]["colors"][:] = 0
    saved["overrides"]["fill_intensity"] = 100
    assert color(light) == (1, 0, 0)
    assert UsdLux.SphereLight(stage.GetPrimAtPath(path)).GetIntensityAttr().Get() == 23
    controller.update(sample, down_camera(0.7))
    assert color(light) == future_color
    invalid = controller.state_dict()
    invalid["parameters"]["directions"][0, 0, 0] = float("nan")
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        controller.load_state_dict(invalid)
    assert stage.GetRootLayer().ExportToString() == before


def test_frozen_rig_does_not_follow_later_camera_pose(adapter, scene):
    _, light, _ = scene
    controller = rig(adapter, scene, frozen=True)
    controller.update(room(), down_camera(0.2))
    first = UsdGeom.Xformable(light).GetLocalTransformation()
    controller.update(room(), down_camera(0.8))
    assert UsdGeom.Xformable(light).GetLocalTransformation() == first
    torch.testing.assert_close(
        controller.last_parameters["target_positions"], torch.tensor([[[0.2, 0, 0]]], dtype=torch.float64)
    )


def test_declared_missing_lights_fail_at_construction(adapter, scene):
    stage, _, path = scene
    with pytest.raises(ValueError, match="Missing.*light"):
        adapter.SceneLightingRig(
            stage,
            [{"lights": ["/Missing"]}],
            [path],
            LightingConfig(),
            CameraProfile(),
            torch.zeros((1, 3)),
            torch.zeros(1),
        )


def test_world_origin_and_local_support_height_reach_usd_and_bounded_horizon_fallback(adapter, scene):
    stage, light, fill = scene
    root = UsdGeom.Xform(stage.GetPrimAtPath("/World/envs/env_0/Room"))
    root.AddTranslateOp().Set(Gf.Vec3d(4, 5, 1))
    controller = adapter.SceneLightingRig(
        stage,
        [{"lights": [str(light.GetPath())]}],
        [fill],
        LightingConfig(profile="rgb_disco", motion_mode="wrist_view"),
        CameraProfile(),
        torch.tensor([[4.0, 5, 1]]),
        torch.tensor([0.4]),
    )
    controller.update(room(), (torch.tensor([[4.0, 5, 2]]), torch.tensor([[1.0, 0, 0, 0]])))
    torch.testing.assert_close(
        controller.last_parameters["target_positions"], torch.tensor([[[4.0, 5, 1.4]]], dtype=torch.float64)
    )
    transform = UsdGeom.Xformable(light).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert tuple(transform.ExtractTranslation()) == (4, 5, 3)
    assert tuple(transform.TransformDir(Gf.Vec3d(0, 0, -1))) == pytest.approx((0, 0, -1))
    root_half = math.sqrt(0.5)
    controller.update(room(), (torch.tensor([[4.0, 5, 2]]), torch.tensor([[0, root_half, 0, root_half]])))
    assert controller.last_parameters["visible_ray_mask"].tolist() == [[False]]
    torch.testing.assert_close(
        controller.last_parameters["target_positions"], torch.tensor([[[4.0, 5, 1.4]]], dtype=torch.float64)
    )


def test_preview_radius_changes_actual_cone_and_clear_restores_sampled_appearance(adapter, scene):
    _, light, _ = scene
    controller = rig(adapter, scene)
    controller.set_preview({"radius_m": 0.25, "speed": 0, "saturation": 0})
    controller.update(room(), down_camera())
    shaping = UsdLux.ShapingAPI(light)
    assert shaping.GetShapingConeAngleAttr().Get() == pytest.approx(7.12501635, abs=1e-5)
    assert controller.set_preview(None) == {}
    sample = room()
    sample["elapsed_seconds"][:] = 2
    controller.update(sample, down_camera())
    assert shaping.GetShapingConeAngleAttr().Get() == pytest.approx(2.862405226, abs=1e-5)
    assert color(light) == pytest.approx((0.2, 0.3, 0.4))
    assert controller.state_dict()["effective_elapsed"].tolist() == [2]


def test_indexed_update_preserves_other_room_usd_clock_and_future_continuation(adapter):
    def make_scene():
        stage = Usd.Stage.CreateInMemory()
        lights, fills, paths = [], [], []
        for index in range(2):
            root = f"/World/envs/env_{index}/Room"
            UsdGeom.Xform.Define(stage, root).AddTranslateOp().Set(Gf.Vec3d(4 * index, 0, 0))
            light = UsdLux.SphereLight.Define(stage, root + "/Lights/Light0")
            light.CreateRadiusAttr(0.04)
            light.CreateIntensityAttr(300)
            light.CreateColorAttr(Gf.Vec3f(1))
            lights.append(light)
            paths.append({"lights": [str(light.GetPath())]})
            fills.append(adapter.author_fill_light(stage, root, (0, 0, 2.5), 17))
        controller = adapter.SceneLightingRig(
            stage,
            paths,
            fills,
            LightingConfig(profile="rgb_disco", motion_mode="wrist_view"),
            CameraProfile(),
            torch.tensor([[0.0, 0, 0], [4.0, 0, 0]]),
            torch.zeros(2),
        )
        return stage, lights, controller

    def visual_values(light):
        return {str(attr.GetName()): attr.Get() for attr in light.GetPrim().GetAttributes()}

    stage, lights, controller = make_scene()
    reference_stage, reference_lights, reference = make_scene()
    sample = {key: value.repeat((2,) + (1,) * (value.ndim - 1)) for key, value in room().items()}
    sample["light_view_extent"][:] = 0.5
    pose = torch.tensor([[0.0, 0, 1], [4.0, 0, 1]]), torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    for current in (controller, reference):
        current.set_preview({"speed": 0.5})
        current.update(sample, pose)
    before = controller.state_dict()
    unchanged_usd = visual_values(lights[1])
    sample["elapsed_seconds"][:] = 1
    sample["light_intensity"][0] = 500
    pose[0][0, 0] = 0.3
    pose[0][1, 0] = 4.8
    controller.update(sample, pose, env_ids=torch.tensor([0]))
    after = controller.state_dict()
    assert visual_values(lights[1]) == unchanged_usd
    for key in before["parameters"]:
        torch.testing.assert_close(after["parameters"][key][1], before["parameters"][key][1], atol=0, rtol=0)
    for key in ("effective_elapsed", "input_elapsed"):
        torch.testing.assert_close(after[key][1], before[key][1], atol=0, rtol=0)
    assert controller.last_parameters["target_positions"][0, 0, 0] != before["parameters"]["target_positions"][0, 0, 0]
    assert lights[0].GetIntensityAttr().Get() == 500
    sample["elapsed_seconds"][:] = 2
    for current in (controller, reference):
        current.update(sample, pose)
    assert visual_values(lights[1]) == visual_values(reference_lights[1])
    for key in controller.last_parameters:
        torch.testing.assert_close(
            controller.last_parameters[key][1], reference.last_parameters[key][1], atol=0, rtol=0
        )
    # Keep both stage owners alive until every cached schema has been consumed.
    assert stage and reference_stage


def test_speed_freezes_hue_but_uses_authoritative_saturation_cycle_sample(adapter, scene):
    stage, light, fill = scene
    controller = adapter.SceneLightingRig(
        stage,
        [{"lights": [str(light.GetPath())]}],
        [fill],
        LightingConfig(profile="rgb_disco", motion_mode="wrist_view", color_saturation=(0, 1), saturation_period_s=4),
        CameraProfile(),
        torch.zeros((1, 3)),
        torch.zeros(1),
    )
    sample = room()
    sample["light_saturation"][:] = 0.5
    sample["light_saturation_phase"] = torch.zeros((1, 1))
    controller.set_preview({"speed": 0})
    controller.update(sample, down_camera())
    assert color(light) == (1, 0.5, 0.5)
    sample["elapsed_seconds"][:] = 1
    sample["light_saturation"][:] = 1
    controller.update(sample, down_camera())
    assert color(light) == (1, 0, 0)
    assert controller.state_dict()["effective_elapsed"].tolist() == [0]
    controller.set_preview({"saturation": 0.25})
    controller.update(sample, down_camera())
    assert color(light) == (1, 0.75, 0.75)
