# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch
from isaaclab_hiwonder.config import ConfettiConfig, DistractorConfig, LightingConfig, RandomizationConfig
from isaaclab_hiwonder.randomization import RandomizationManager


def test_subset_reset_order_is_independent_and_base_is_supported_by_sampled_table():
    a = RandomizationManager(3, seed=123)
    b = RandomizationManager(3, seed=123)
    before = a.snapshot([0, 2])
    a.reset([1])
    b.reset([0])
    b.reset([2])
    b.reset([1])
    for group, fields in a.snapshot([1]).items():
        for key, value in fields.items():
            assert torch.equal(value, b.snapshot([1])[group][key]), (group, key)
    for group, fields in before.items():
        for key, value in fields.items():
            assert torch.equal(value, a.snapshot([0, 2])[group][key]), (group, key)
    full = a.snapshot()
    assert torch.equal(full["base"]["position"][:, 2], full["table"]["height"])
    assert torch.all(full["actuator"]["effort_limit"] > 0)
    assert torch.std(full["actuator"]["effort_limit"]) > 0
    assert torch.std(full["robot"]["geometry_scale"]) > 0
    assert torch.all(full["confetti"]["position"][..., 2] > full["table"]["height"][:, None])


def test_temporal_disco_is_indexed_and_checkpoint_replays_future_samples():
    manager = RandomizationManager(
        2,
        RandomizationConfig(
            lighting=LightingConfig(profile="rgb_disco", color_saturation=(0.1, 0.9), saturation_period_s=2.0)
        ),
        seed=7,
    )
    before = manager.snapshot()
    manager.step(0.7, [0])
    after = manager.snapshot()
    assert not torch.equal(before["room"]["light_direction"][0], after["room"]["light_direction"][0])
    assert torch.equal(before["room"]["light_direction"][1], after["room"]["light_direction"][1])
    assert not torch.equal(before["room"]["light_saturation"][0], after["room"]["light_saturation"][0])
    assert torch.equal(before["room"]["light_saturation"][1], after["room"]["light_saturation"][1])
    assert torch.all((after["room"]["light_saturation"] >= 0.1) & (after["room"]["light_saturation"] <= 0.9))
    assert torch.equal(before["room"]["light_intensity"], after["room"]["light_intensity"])
    manager.step(100_000_000.0, [0])
    elapsed = manager.snapshot()["room"]["elapsed_seconds"][0]
    manager.step(0.01, [0])
    assert float(manager.snapshot()["room"]["elapsed_seconds"][0] - elapsed) == pytest.approx(0.01, abs=1e-8)
    clone = RandomizationManager(2, manager.config, seed=888)
    clone.load_state_dict(manager.state_dict())
    manager.step(0.125, [0])
    clone.step(0.125, [0])
    manager.reset([1])
    clone.reset([1])
    for group, fields in manager.snapshot().items():
        for key, value in fields.items():
            assert torch.equal(value, clone.snapshot()[group][key]), (group, key)


def test_zero_clutter_budget_and_slow_smudge_drift_are_supported():
    cfg = RandomizationConfig(
        distractor=DistractorConfig(count=(0, 0), max_count=0),
        confetti=ConfettiConfig(count=(0, 0), max_count=0),
        lighting=LightingConfig(profile="white_disco", count=(0, 0), max_lights=0),
    )
    manager = RandomizationManager(2, cfg)
    before = manager.snapshot()
    manager.step(1.0, [1])
    after = manager.snapshot()
    assert after["distractor"]["position"].shape == (2, 0, 3)
    assert after["room"]["light_color"].shape == (2, 0, 3)
    assert after["confetti"]["coverage"].tolist() == [0, 0]
    assert not torch.equal(before["smudge"]["offset"][1], after["smudge"]["offset"][1])
    assert torch.equal(before["smudge"]["offset"][0], after["smudge"]["offset"][0])
    assert torch.all(after["room"]["light_color"] == 1)


def test_clutter_has_support_excludes_task_and_preserves_body_clearances():
    cfg = RandomizationConfig(distractor=DistractorConfig(count=(6, 6)), confetti=ConfettiConfig(physical_budget=3))
    state = RandomizationManager(3, cfg, seed=99, bootstrap_attempts=32).snapshot()
    objects = state["distractor"]
    for env in range(3):
        centers = objects["position"][env, :, :2]
        radii = objects["size"][env].norm(dim=-1) / 2
        assert torch.all(centers.norm(dim=-1) - radii >= 0.3)
        pairwise = torch.cdist(centers, centers)
        pairwise.fill_diagonal_(float("inf"))
        assert torch.all(pairwise > radii[:, None] + radii[None, :])
        assert torch.all(centers[:, 0].abs() + radii < state["table"]["width"][env] / 2)
        assert torch.all(centers[:, 1].abs() + radii < state["table"]["depth"][env] / 2)
    assert state["confetti"]["physical"].sum(1).tolist() == [3, 3, 3]


def test_narrow_table_rejects_clutter_without_conservative_support_area():
    from isaaclab_hiwonder.config import BaseRandomizationConfig, TableConfig

    cfg = RandomizationConfig(
        table=TableConfig(width_m=(0.15, 0.15), depth_m=(3, 3)),
        base=BaseRandomizationConfig(x_m=(0, 0), y_m=(0, 0), support_radius_m=0.025, task_clearance_m=0.1),
        distractor=DistractorConfig(count=(1, 1), max_count=1, size_m=(0.2, 0.2), shapes=("box",)),
    )
    with pytest.raises(ValueError, match="support"):
        RandomizationManager(1, cfg, seed=1)


@pytest.mark.parametrize(
    "field,bounds",
    [
        ("stiffness", (float("nan"), 35)),
        ("stiffness", (35, 15)),
        ("damping", (-1, 1)),
        ("encoder_noise_std", (-0.01, 0.01)),
    ],
)
def test_invalid_actuator_domains_rejected_before_sampling(field, bounds):
    from dataclasses import replace

    from isaaclab_hiwonder.config import ActuatorRandomizationConfig

    with pytest.raises(ValueError, match=field):
        actuator = replace(ActuatorRandomizationConfig(), **{field: bounds})
        RandomizationManager(1, RandomizationConfig(actuator=actuator), seed=1)


def test_group_streams_freeze_selective_reset_and_nominal_disabled_effects():
    from dataclasses import replace

    from isaaclab_hiwonder.config import GroupControl, RandomizationControls

    controls = RandomizationControls(
        robot=GroupControl(frozen=True),
        room=GroupControl(cadence="fixed", frozen=True),
        smudge=GroupControl(enabled=False),
        confetti=GroupControl(enabled=False),
        actuator=GroupControl(seed=123),
    )
    a = RandomizationManager(
        2, RandomizationConfig(controls=controls, lighting=LightingConfig(profile="rgb_disco")), seed=9
    )
    initial = a.snapshot()
    a.reset([0], groups=["image"])
    selective = a.snapshot()
    assert not torch.equal(initial["image"]["seed"][0], selective["image"]["seed"][0])
    assert torch.equal(initial["actuator"]["stiffness"], selective["actuator"]["stiffness"])
    a.reset([0])
    a.step(0.5, [0])
    after = a.snapshot()
    assert torch.equal(initial["robot"]["geometry_scale"], after["robot"]["geometry_scale"])
    assert torch.equal(initial["room"]["light_phase"], after["room"]["light_phase"])
    assert torch.equal(initial["room"]["elapsed_seconds"], after["room"]["elapsed_seconds"])
    assert torch.equal(initial["room"]["light_color"], after["room"]["light_color"])
    for key, value in initial["robot"].items():
        assert torch.equal(value, after["robot"][key])
    assert torch.all(after["confetti"]["count"] == 0)
    assert torch.all(after["smudge"]["spot_count"] == 0) and torch.all(after["smudge"]["haze"] == 0)
    b = RandomizationManager(2, replace(a.config, image=replace(a.config.image, rotation_enabled=False)), seed=9)
    b.reset([0])
    b.step(0.5, [0])
    assert torch.equal(after["actuator"]["stiffness"], b.snapshot()["actuator"]["stiffness"])
    restored = RandomizationManager(2, a.config, seed=42)
    restored.load_state_dict(a.state_dict())
    a.reset([0])
    restored.reset([0])
    assert torch.equal(a.snapshot()["image"]["seed"], restored.snapshot()["image"]["seed"])


def test_group_reset_interval_preserves_realized_samples_between_scheduled_resets():
    from isaaclab_hiwonder.config import GroupControl, RandomizationControls

    manager = RandomizationManager(
        1, RandomizationConfig(controls=RandomizationControls(image=GroupControl(reset_interval=2))), seed=2
    )
    initial = manager.snapshot()["image"]["seed"]
    manager.reset()
    assert torch.equal(initial, manager.snapshot()["image"]["seed"])
    manager.reset()
    assert not torch.equal(initial, manager.snapshot()["image"]["seed"])


def test_group_random_draw_budget_cannot_shift_other_groups_and_disabled_effects_bypass():
    from dataclasses import replace

    from isaaclab_hiwonder.config import GroupControl, RandomizationControls
    from isaaclab_hiwonder.images import ImageRandomizer

    cfg = RandomizationConfig(
        controls=RandomizationControls(
            image=GroupControl(enabled=False), smudge=GroupControl(enabled=False), room=GroupControl(enabled=False)
        )
    )
    a = RandomizationManager(2, cfg, seed=9)
    b = RandomizationManager(2, replace(cfg, lighting=replace(cfg.lighting, max_lights=5)), seed=9)
    assert torch.equal(a.snapshot()["actuator"]["stiffness"], b.snapshot()["actuator"]["stiffness"])
    assert torch.equal(a.snapshot()["room"]["light_phase"][0], a.snapshot()["room"]["light_phase"][1])
    augmenter = ImageRandomizer(a.image_config, a.smudge_config)
    image = torch.randint(0, 256, (2, 16, 20, 3), dtype=torch.uint8)
    assert torch.equal(augmenter(image, augmenter.sample_parameters(2, 16, 20, seed=42)), image)


def test_selective_reset_never_rejects_a_discarded_hypothetical_frozen_group():
    from isaaclab_hiwonder.config import BaseRandomizationConfig, GroupControl, RandomizationControls, TableConfig

    config = RandomizationConfig(
        table=TableConfig(width_m=(0.15, 0.15), depth_m=(3, 3)),
        base=BaseRandomizationConfig(x_m=(0, 0), y_m=(0, 0), support_radius_m=0.025, task_clearance_m=0.1),
        distractor=DistractorConfig(count=(0, 1), max_count=1, size_m=(0.2, 0.2), shapes=("box",)),
        controls=RandomizationControls(distractor=GroupControl(frozen=True)),
    )
    manager = RandomizationManager(1, config, seed=2)
    before = manager.snapshot()
    assert before["distractor"]["count"].tolist() == [0]
    for _ in range(3):
        manager.reset(groups=["image"])
    after = manager.snapshot()
    assert torch.equal(before["distractor"]["count"], after["distractor"]["count"])
    assert torch.equal(before["table"]["width"], after["table"]["width"])
    assert not torch.equal(before["image"]["seed"], after["image"]["seed"])


@pytest.mark.parametrize("shape", ["circle", "hexagon"])
def test_nonrectangular_tables_support_actual_clutter_and_confetti_footprints(shape):
    import math

    from isaaclab_hiwonder.config import TableConfig

    cfg = RandomizationConfig(
        table=TableConfig(shapes=(shape,), width_m=(1, 1), depth_m=(1, 1)),
        distractor=DistractorConfig(count=(2, 2), max_count=2, size_m=(0.02, 0.02), shapes=("box",)),
        confetti=ConfettiConfig(
            count=(100, 100), max_count=100, size_m=(0.01, 0.01), aspect_ratio=(1, 1), shapes=("rectangle",)
        ),
    )
    state = RandomizationManager(1, cfg, seed=19).snapshot()

    def corners(centers, sizes, yaw):
        signs = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / 2
        offsets = signs[None] * sizes[:, None]
        x = offsets[..., 0] * yaw.cos()[:, None] - offsets[..., 1] * yaw.sin()[:, None]
        y = offsets[..., 0] * yaw.sin()[:, None] + offsets[..., 1] * yaw.cos()[:, None]
        return centers[:, None, :] + torch.stack((x, y), -1)

    objects = state["distractor"]
    yaw = 2 * torch.atan2(objects["quaternion_xyzw"][0, :, 2], objects["quaternion_xyzw"][0, :, 3])
    object_points = corners(objects["position"][0, :, :2], objects["size"][0, :, :2], yaw)
    pieces = state["confetti"]
    piece_points = corners(pieces["position"][0, :, :2], pieces["size"][0, :, None].expand(-1, 2), pieces["yaw"][0])
    points = torch.cat((object_points.flatten(0, 1), piece_points.flatten(0, 1)))
    if shape == "circle":
        assert torch.all(points.norm(dim=-1) <= 0.5 + 1e-6)
    else:
        assert torch.all(points[:, 1].abs() <= math.sqrt(3) / 4 + 1e-6)
        assert torch.all(math.sqrt(3) * points[:, 0].abs() + points[:, 1].abs() <= math.sqrt(3) / 2 + 1e-6)


def test_bootstrap_redraws_only_coherent_support_groups_before_physics_bake():
    from isaaclab_hiwonder.config import GroupControl, RandomizationControls, TableConfig

    cfg = RandomizationConfig(
        table=TableConfig(shapes=("circle",), width_m=(1, 1), depth_m=(0.65, 1.0)),
        distractor=DistractorConfig(count=(6, 6)),
        controls=RandomizationControls(table=GroupControl(cadence="fixed")),
    )
    manager = RandomizationManager(2, cfg, seed=1, bootstrap_attempts=32)
    assert manager.bootstrap_attempts[0] > 1
    before = manager.snapshot()
    manager.redraw_cohorts([0], max_attempts=32)
    after = manager.snapshot()
    assert not torch.equal(before["table"]["depth"][0], after["table"]["depth"][0])
    assert torch.equal(before["robot"]["geometry_scale"], after["robot"]["geometry_scale"])
    assert torch.equal(before["room"]["light_phase"], after["room"]["light_phase"])
    assert torch.equal(after["base"]["position"][:, 2], after["table"]["height"])
    for group, values in before.items():
        for key, value in values.items():
            assert torch.equal(value[1], after[group][key][1]), (group, key)


def test_robot_variant_samples_agree_with_prebuilt_asset_bank():
    cfg = RandomizationConfig()
    state = RandomizationManager(20, cfg, seed=12).snapshot()["robot"]
    actual_bank = {"nominal": 1.0, "scale099": 0.99, "scale101": 1.01}
    for index, scale in zip(state["geometry_variant_id"].tolist(), state["geometry_scale"].tolist()):
        assert scale == pytest.approx(actual_bank[cfg.robot.geometry_variants[index]])


def test_disabled_robot_keeps_the_first_configured_asset_and_its_coherent_scale():
    from isaaclab_hiwonder.config import GroupControl, RandomizationControls, RobotRandomizationConfig

    cfg = RandomizationConfig(
        robot=RobotRandomizationConfig(geometry_variants=("scale099",), variant_scales=(0.99,)),
        controls=RandomizationControls(robot=GroupControl(enabled=False)),
    )
    manager = RandomizationManager(1, cfg, seed=4)
    sample = manager.snapshot()["robot"]
    assert cfg.robot.geometry_variants[int(sample["geometry_variant_id"][0])] == "scale099"
    assert float(sample["geometry_scale"][0]) == pytest.approx(0.99)
    torch.testing.assert_close(sample["mass_scale"], torch.full((1, 7), 0.99**3))
    torch.testing.assert_close(sample["inertia_scale"], torch.full((1, 7), 0.99**5))


def test_material_families_change_effective_pbr_priors_within_requested_ranges():
    from isaaclab_hiwonder.config import MaterialConfig, TableConfig

    sampled = {}
    for family in ("wood", "metal"):
        config = RandomizationConfig(
            table=TableConfig(material=MaterialConfig(families=(family,), texture_ids=("none",))),
            distractor=DistractorConfig(count=(0, 0), max_count=0),
            confetti=ConfettiConfig(count=(0, 0), max_count=0),
        )
        sampled[family] = RandomizationManager(4, config, seed=19).snapshot()["table"]
    # Dielectric wood versus metallic material is a rendering/physics prior, not just a tag.
    assert torch.all(sampled["wood"]["material_metallic"] == 0)
    assert torch.all((sampled["metal"]["material_metallic"] >= 0.6) & (sampled["metal"]["material_metallic"] <= 0.8))
    assert torch.all(sampled["wood"]["material_roughness"] >= 0.15)
    assert torch.all(sampled["metal"]["material_roughness"] <= 0.95)
    with pytest.raises(ValueError, match="material family|Material family"):
        bad = RandomizationConfig(table=TableConfig(material=MaterialConfig(families=("unknown",))))
        RandomizationManager(1, bad)


def test_base_rejection_retries_preserve_other_draws_and_future_retained_env_streams():
    from isaaclab_hiwonder.config import BaseRandomizationConfig, TableConfig
    from isaaclab_hiwonder.randomization import InvalidLayout

    config = RandomizationConfig(
        table=TableConfig(shapes=("rectangle",), width_m=(1.2, 1.2), depth_m=(0.95, 0.95)),
        base=BaseRandomizationConfig(x_m=(-0.25, 0.25), y_m=(-0.25, 0.25)),
        distractor=DistractorConfig(count=(6, 6), physical_probability=1.0),
    )
    manager = RandomizationManager(2, config, seed=0, bootstrap_attempts=32)
    groups = ["actuator", "base", "image", "smudge"]
    for _ in range(33):
        manager.reset([0], groups=groups)
    before = manager.state_dict()
    # The actual episode-reset draw34 rejects a supported base beside fixed clutter.
    with pytest.raises(InvalidLayout, match="collides with base"):
        manager.reset([0], groups=groups)
    reference = RandomizationManager(2, config, seed=99, bootstrap_attempts=32)
    reference.load_state_dict(before)
    reference.reset([0], groups=["actuator", "image", "smudge"])
    manager.reset([0], groups=groups, max_attempts=32)
    accepted = manager.snapshot()
    for group in ("actuator", "image", "smudge"):
        for key, value in reference.snapshot()[group].items():
            assert torch.equal(accepted[group][key], value), (group, key)
    for group in ("table", "room", "robot", "distractor", "confetti"):
        for key, value in before["samples"][group].items():
            assert torch.equal(accepted[group][key], value), (group, key)
    for group, fields in before["samples"].items():
        for key, value in fields.items():
            assert torch.equal(accepted[group][key][1], value[1]), (group, key)
        expected_count = before["reset_counts"][group] + torch.tensor([int(group in groups), 0])
        assert torch.equal(manager.reset_counts[group], expected_count)
    active = accepted["distractor"]["active"][0]
    centers = accepted["distractor"]["position"][0, active, :2]
    radius = accepted["distractor"]["size"][0, active].norm(dim=-1) / 2
    assert torch.all((centers - accepted["base"]["position"][0, :2]).norm(dim=-1) >= radius + 0.1)
    # Compare real future draws, not just retained RNG-state bytes.
    manager.reset([1], groups=groups, max_attempts=32)
    reference.reset([1], groups=groups, max_attempts=32)
    for group, fields in manager.snapshot([1]).items():
        for key, value in fields.items():
            assert torch.equal(value, reference.snapshot([1])[group][key]), (group, key)


def test_exhausted_base_retry_budget_rolls_back_the_whole_transaction():
    from isaaclab_hiwonder.config import BaseRandomizationConfig, TableConfig
    from isaaclab_hiwonder.randomization import InvalidLayout

    config = RandomizationConfig(
        table=TableConfig(shapes=("rectangle",), width_m=(1.0, 1.0), depth_m=(0.8, 0.8)),
        base=BaseRandomizationConfig(x_m=(-1.0, 1.0), y_m=(-1.0, 1.0)),
        distractor=DistractorConfig(count=(0, 0), max_count=0),
        confetti=ConfettiConfig(count=(0, 0), max_count=0),
    )
    manager = RandomizationManager(2, config, seed=0, bootstrap_attempts=32)
    before = manager.state_dict()
    with pytest.raises(InvalidLayout, match="after 2 attempts"):
        manager.reset([0], groups=["actuator", "base", "image", "smudge"], max_attempts=2)
    after = manager.state_dict()
    for group, fields in before["samples"].items():
        for key, value in fields.items():
            assert torch.equal(value, after["samples"][group][key]), (group, key)
        assert torch.equal(before["reset_counts"][group], after["reset_counts"][group])
        for old, new in zip(before["rng_states"][group], after["rng_states"][group]):
            assert torch.equal(old, new), group


def test_initial_physical_cards_clear_the_cad_base_while_visual_cards_can_occlude_it():
    from dataclasses import replace

    from isaaclab_hiwonder.config import BaseRandomizationConfig, RobotRandomizationConfig

    config = RandomizationConfig(
        robot=RobotRandomizationConfig(geometry_variants=("nominal",), variant_scales=(1.0,)),
        base=BaseRandomizationConfig(support_radius_m=0.001),
        confetti=ConfettiConfig(physical_budget=2),
    )
    state = RandomizationManager(2, config, seed=14, bootstrap_attempts=32).snapshot()
    cards = state["confetti"]
    radii = cards["size"] * (1 + cards["aspect_ratio"].square()).sqrt() / 2
    distance = (cards["position"][..., :2] - state["base"]["position"][:, None, :2]).norm(dim=-1)
    # The supplied nominal CAD bounds require at least a 101.7mm base envelope,
    # regardless of a profile's underreported 1mm support radius. Seed14/env1/card0
    # previously started 446.8um inside a recorded native base collision hull.
    assert torch.all(distance[cards["physical"]] >= 0.1017 + radii[cards["physical"]])
    visual_config = replace(config, confetti=replace(config.confetti, physical_budget=0))
    visual = RandomizationManager(2, visual_config, seed=14, bootstrap_attempts=32).snapshot()
    # These pieces may still occlude the robot because they have no collision body.
    separation = (visual["confetti"]["position"][1, 0, :2] - visual["base"]["position"][1, :2]).norm()
    assert float(separation) < 0.03
    assert not visual["confetti"]["physical"].any()


def test_retained_physical_cards_reject_new_base_poses_without_redrawing_cards():
    from isaaclab_hiwonder.config import BaseRandomizationConfig, RobotRandomizationConfig, TableConfig

    config = RandomizationConfig(
        table=TableConfig(shapes=("rectangle",), width_m=(1.2, 1.2), depth_m=(0.95, 0.95)),
        base=BaseRandomizationConfig(x_m=(-0.35, 0.35), y_m=(-0.35, 0.35), support_radius_m=0.001),
        robot=RobotRandomizationConfig(geometry_variants=("nominal",), variant_scales=(1.0,)),
        distractor=DistractorConfig(count=(0, 0), max_count=0),
        confetti=ConfettiConfig(physical_budget=2),
    )
    manager = RandomizationManager(1, config, seed=0, bootstrap_attempts=32)
    original_cards = manager.snapshot()["confetti"]
    for _ in range(16):
        manager.reset([0], groups=["actuator", "base", "image", "smudge"], max_attempts=32)
        state = manager.snapshot()
        cards = state["confetti"]
        physical = cards["physical"][0]
        radii = cards["size"][0, physical] * (1 + cards["aspect_ratio"][0, physical].square()).sqrt() / 2
        distance = (cards["position"][0, physical, :2] - state["base"]["position"][0, :2]).norm(dim=-1)
        assert torch.all(distance >= 0.1017 + radii)
    for key, value in original_cards.items():
        assert torch.equal(manager.snapshot()["confetti"][key], value), key


def test_projected_snapshots_own_copies_and_do_not_change_temporal_or_future_reset_state():
    manager = RandomizationManager(2, seed=123)
    reference = RandomizationManager(2, seed=123)
    projected = manager.snapshot([1, 0], groups=["actuator"])
    assert list(projected) == ["actuator"]
    torch.testing.assert_close(
        projected["actuator"]["stiffness"], reference.snapshot()["actuator"]["stiffness"][[1, 0]]
    )
    projected["actuator"]["stiffness"].zero_()
    assert torch.all(manager.snapshot(groups=["actuator"])["actuator"]["stiffness"] > 0)
    stepped = manager.step(0.5, [1], groups=["room"])
    assert list(stepped) == ["room"]
    reference.step(0.5, [1])
    for item in (manager, reference):
        item.reset([0], groups=["actuator", "image"])
    for group, fields in reference.snapshot().items():
        for key, expected in fields.items():
            assert torch.equal(manager.snapshot()[group][key], expected), (group, key)
    before = manager.snapshot()["room"]["elapsed_seconds"].clone()
    with pytest.raises(ValueError, match="group"):
        manager.step(1, groups=["missing"])
    assert torch.equal(manager.snapshot()["room"]["elapsed_seconds"], before)


def test_temporal_saturation_reaches_configured_extrema_on_its_own_clock():
    cfg = RandomizationConfig(
        lighting=LightingConfig(profile="rgb_disco", color_saturation=(0.1, 0.9), saturation_period_s=2)
    )
    manager = RandomizationManager(1, cfg, seed=7)
    checkpoint = manager.state_dict()
    checkpoint["samples"]["room"]["light_saturation_phase"].zero_()
    manager.load_state_dict(checkpoint)
    maximum = manager.step(0.5, groups=["room"])["room"]
    torch.testing.assert_close(maximum["light_saturation"], torch.full((1, 4), 0.9))
    minimum = manager.step(1.0, groups=["room"])["room"]
    torch.testing.assert_close(minimum["light_saturation"], torch.full((1, 4), 0.1))


@pytest.mark.parametrize("profile", ["stationary", "white_disco", "rgb_disco"])
def test_fixed_zero_saturation_and_white_profiles_remain_white_with_independent_intensity(profile):
    config = RandomizationConfig(lighting=LightingConfig(profile=profile, color_saturation=(0, 0)))
    manager = RandomizationManager(1, config)
    before = manager.snapshot(groups=["room"])["room"]
    after = manager.step(1.25, groups=["room"])["room"]
    torch.testing.assert_close(after["light_color"], torch.ones((1, 4, 3)))
    assert torch.equal(before["light_saturation"], after["light_saturation"])
    assert torch.equal(before["light_intensity"], after["light_intensity"])
