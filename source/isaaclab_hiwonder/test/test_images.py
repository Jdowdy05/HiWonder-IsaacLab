# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

import pytest
import torch
from isaaclab_hiwonder.config import ImageTransformConfig, SmudgeConfig
from isaaclab_hiwonder.images import ImageParameters, ImageRandomizer


def test_rotation_padding_stays_black_after_brightness_and_episode_smudge():
    config = ImageTransformConfig(
        mode="all", shear_degrees=(0, 0), crop_scale=(1, 1), rotation_degrees=(35, 35), brightness=(1.5, 1.5)
    )
    randomizer = ImageRandomizer(config, SmudgeConfig(spot_count=(2, 2), haze=(0.2, 0.2)))
    params = randomizer.sample_parameters(2, 40, 60, seed=12)
    original = torch.full((2, 40, 60, 3), 150, dtype=torch.uint8)
    output, mask = randomizer.apply(original, params, return_mask=True)
    assert output.shape == original.shape and output.dtype == original.dtype
    assert (~mask).any() and mask.any()
    assert torch.all(output[~mask] == 0)
    assert output[mask].float().mean() > 100
    assert torch.equal(output, randomizer(original, params))
    assert torch.all(original == 150)


def test_one_mode_selects_one_transform_and_none_bypasses_geometric_and_color_changes():
    randomizer = ImageRandomizer(ImageTransformConfig(mode="one"), SmudgeConfig(enabled=False))
    params = randomizer.sample_parameters(20, 12, 18, seed=1)
    assert torch.all(params.enabled.sum(dim=1) == 1)
    assert torch.unique(params.enabled, dim=0).shape[0] > 1
    pixels = torch.randint(0, 256, (20, 12, 18, 3), dtype=torch.uint8)
    bypass = ImageRandomizer(ImageTransformConfig(mode="none"), SmudgeConfig(enabled=False))
    assert torch.equal(bypass(pixels, bypass.sample_parameters(20, 12, 18, seed=1)), pixels)


def test_persistent_smudge_changes_clean_image_but_replays_without_rng():
    randomizer = ImageRandomizer(
        ImageTransformConfig(mode="none"), SmudgeConfig(spot_count=(3, 3), streak_count=(2, 2), opacity=(0.8, 0.8))
    )
    params = randomizer.sample_parameters(1, 50, 50, seed=19)
    image = torch.full((1, 50, 50, 3), 200, dtype=torch.uint8)
    output = randomizer(image, params)
    assert not torch.equal(output, image)
    torch.manual_seed(999)
    assert torch.equal(output, randomizer(image, params))
    restored = ImageParameters.from_state_dict(params.state_dict())
    assert torch.equal(output, randomizer(image, restored))


def test_shear_crop_and_hue_have_observable_effects_with_reusable_records():
    randomizer = ImageRandomizer(
        ImageTransformConfig(
            mode="all",
            rotation_enabled=False,
            shear_degrees=(20, 20),
            crop_scale=(0.6, 0.6),
            brightness=(1, 1),
            contrast=(1, 1),
            saturation=(1, 1),
            hue_radians=(0.8, 0.8),
        ),
        SmudgeConfig(enabled=False),
    )
    params = randomizer.sample_parameters(1, 30, 40, seed=81)
    image = torch.zeros((1, 30, 40, 3), dtype=torch.uint8)
    image[:, 6:24, 8:32, 0] = 255
    output = randomizer(image, params)
    assert (output[..., 1] > 20).any()
    assert (output[..., 0] > 0).sum() > (image[..., 0] > 0).sum()
    assert params.matrix[0, 0, 1].abs() > 0.1


def test_smudge_drift_uses_explicit_replayable_offsets():
    randomizer = ImageRandomizer(ImageTransformConfig(mode="none"), SmudgeConfig(spot_count=(2, 2), haze=(0, 0)))
    params = randomizer.sample_parameters(1, 32, 40, seed=5)
    image = torch.full((1, 32, 40, 3), 180, dtype=torch.uint8)
    offset = torch.tensor([[0.2, -0.1]])
    moved = randomizer.apply(image, params, smudge_offset=offset)
    assert not torch.equal(moved, randomizer(image, params))
    assert torch.equal(moved, randomizer.apply(image, params, smudge_offset=offset))


def test_crop_then_rotation_preserves_uncovered_canvas_corners_after_color():
    config = ImageTransformConfig(
        mode="all",
        shear_enabled=False,
        crop_scale=(0.5, 0.5),
        rotation_degrees=(45, 45),
        brightness=(1.5, 1.5),
        contrast=(0.8, 0.8),
    )
    randomizer = ImageRandomizer(config, SmudgeConfig(haze=(0.2, 0.2)))
    parameters = randomizer.sample_parameters(1, 40, 40, seed=1)
    output, mask = randomizer.apply(torch.full((1, 40, 40, 3), 255, dtype=torch.uint8), parameters, return_mask=True)
    assert (~mask).sum() > 100
    assert torch.all(output[:, 0, 0] == 0)
    assert torch.all(output[:, 0, -1] == 0)
    assert torch.all(output[:, -1, 0] == 0)
    assert torch.all(output[:, -1, -1] == 0)
    assert torch.all(output[~mask] == 0)
    assert torch.all(output[:, 20, 20] > 100)


def test_smudge_local_contrast_reduces_grayscale_spatial_variance():
    from dataclasses import replace

    smudge = SmudgeConfig(
        spot_count=(1, 1),
        streak_count=(0, 0),
        radius=(0.5, 0.5),
        anisotropy=(1, 1),
        opacity=(0.7, 0.7),
        blur_pixels=(0, 0),
        haze=(0, 0),
        tint=(0, 0),
        local_contrast=(1, 1),
    )
    full = ImageRandomizer(ImageTransformConfig(mode="none"), smudge)
    reduced = ImageRandomizer(ImageTransformConfig(mode="none"), replace(smudge, local_contrast=(0, 0)))
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    image = (((xx + yy) % 2) * 255).byte()[None, :, :, None].expand(1, 16, 16, 3).clone()
    full_output = full(image, full.sample_parameters(1, 16, 16, seed=4))
    reduced_output = reduced(image, reduced.sample_parameters(1, 16, 16, seed=4))
    assert reduced_output.float().std() < full_output.float().std() * 0.85


def test_snapshot_sampling_separates_smudge_seed_from_geometry_and_enforces_consumed_counts():
    from isaaclab_hiwonder.randomization import RandomizationManager

    snapshot = RandomizationManager(2, seed=8).snapshot()
    snapshot["smudge"]["spot_count"][:] = 0
    snapshot["smudge"]["streak_count"][:] = 0
    snapshot["smudge"]["haze"][:] = 0.27
    randomizer = ImageRandomizer()
    clean = randomizer.sample_from_snapshot(snapshot, 32, 40)
    assert torch.all(clean.smudge_alpha == 0)
    torch.testing.assert_close(clean.haze, torch.full((2,), 0.27))
    snapshot["smudge"]["spot_count"][:] = 2
    snapshot["smudge"]["seed"] += 13
    changed = randomizer.sample_from_snapshot(snapshot, 32, 40)
    assert changed.smudge_alpha.max() > 0.01
    assert torch.equal(clean.matrix, changed.matrix)
    assert torch.equal(clean.brightness, changed.brightness)
    snapshot["image"]["seed"] += 21
    image_changed = randomizer.sample_from_snapshot(snapshot, 32, 40)
    assert torch.equal(changed.smudge_alpha, image_changed.smudge_alpha)
    assert not torch.equal(changed.matrix, image_changed.matrix)


def test_repeated_image_replay_reuses_sampling_work_without_exposing_cached_masks():
    """Unchanged episode records must avoid rebuilding sampling grids on every observation."""
    randomizer = ImageRandomizer(ImageTransformConfig(mode="none"), SmudgeConfig(enabled=False))
    parameters = randomizer.sample_parameters(1, 4, 6)
    image = torch.arange(72, dtype=torch.uint8).reshape(1, 4, 6, 3)
    first, mask = randomizer.apply(image, parameters, return_mask=True, smudge_offset=torch.zeros((1, 2)))
    assert torch.equal(first, image)
    mask.zero_()
    first.zero_()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True) as profile:
        replay, replay_mask = randomizer.apply(image, parameters, return_mask=True, smudge_offset=torch.zeros((1, 2)))
    assert torch.equal(replay, image)
    assert replay_mask.all()
    assert not any(event.key in ("aten::affine_grid_generator", "aten::cos") for event in profile.key_averages())


def test_mutable_image_records_refresh_geometry_canvas_and_hue_with_owned_restore():
    """Indexed resets and public tensor replacements must invalidate only derived data."""
    randomizer = ImageRandomizer(ImageTransformConfig(mode="none"), SmudgeConfig(enabled=False))
    parameters = randomizer.sample_parameters(1, 4, 6)
    image = torch.arange(72, dtype=torch.uint8).reshape(1, 4, 6, 3)
    randomizer(image, parameters)
    parameters.matrix[0, 0, 2] = 2 / 6
    shifted = torch.zeros_like(image)
    shifted[:, :, :-1] = image[:, :, 1:]
    assert torch.equal(randomizer(image, parameters), shifted)
    parameters.canvas_matrices[0, 0, 0, 2] = -2 / 6
    shifted[:, :, 0] = 0
    output, mask = randomizer.apply(image, parameters, return_mask=True)
    assert torch.equal(output, shifted)
    assert not mask[:, :, 0].any() and not mask[:, :, -1].any()
    parameters.matrix = torch.eye(3)[None]
    parameters.canvas_matrices = torch.eye(3)[None, None].repeat(1, 2, 1, 1)
    red = torch.tensor([255, 0, 0], dtype=torch.uint8).expand_as(image).clone()
    assert torch.equal(randomizer(red, parameters), red)
    parameters.hue_radians[0] = torch.pi
    expected = torch.tensor([0, 170, 170], dtype=torch.uint8).expand_as(image)
    assert torch.equal(randomizer(red, parameters), expected)
    restored = ImageParameters.from_state_dict(parameters.state_dict())
    assert torch.equal(randomizer(red, restored), expected)
    copied = copy.deepcopy(randomizer)
    assert torch.equal(copied(red, copy.deepcopy(restored)), expected)


def test_inference_image_parameters_and_drift_remain_fresh_after_mutation():
    """Inference tensors have no version counter, and drift is independent of episode geometry."""
    randomizer = ImageRandomizer(ImageTransformConfig(mode="none"), SmudgeConfig(enabled=False))
    with torch.inference_mode():
        parameters = randomizer.sample_parameters(1, 4, 6)
        parameters.smudge_alpha[0, 1, 2] = 1
        parameters.smudge_tint[0, 0] = 1
        offset = torch.zeros((1, 2))
        image = torch.full((1, 4, 6, 3), 255, dtype=torch.uint8)
        expected = image.clone()
        expected[0, 1, 2] = torch.tensor([255, 0, 0], dtype=torch.uint8)
        assert torch.equal(randomizer.apply(image, parameters, smudge_offset=offset), expected)
        offset[0, 0] = 1 / 6
        expected[0, 1, 2] = 255
        expected[0, 1, 3] = torch.tensor([255, 0, 0], dtype=torch.uint8)
        assert torch.equal(randomizer.apply(image, parameters, smudge_offset=offset), expected)
        parameters.matrix[0, 0, 2] = 2 / 6
        expected[:, :, -1] = 0
        assert torch.equal(randomizer.apply(image, parameters, smudge_offset=offset), expected)


@pytest.mark.parametrize("field", ["matrix", "canvas_matrices", "hue_radians"])
def test_image_records_modified_through_numpy_alias_refresh_cached_transforms(field):
    randomizer = ImageRandomizer(ImageTransformConfig(mode="none"), SmudgeConfig(enabled=False))
    parameters = randomizer.sample_parameters(1, 4, 6)
    image = torch.arange(72, dtype=torch.uint8).reshape(1, 4, 6, 3)
    if field == "hue_radians":
        image = torch.tensor([255, 0, 0], dtype=torch.uint8).expand_as(image).clone()
    randomizer(image, parameters)
    tensor = getattr(parameters, field)
    version = tensor._version
    alias = tensor.numpy()
    if field == "matrix":
        alias[0, 0, 2] = 2 / 6
        expected = torch.zeros_like(image)
        expected[:, :, :-1] = image[:, :, 1:]
    elif field == "canvas_matrices":
        alias[0, 0, 0, 2] = 2 / 6
        expected = image.clone()
        expected[:, :, -1] = 0
    else:
        alias[0] = torch.pi
        expected = torch.tensor([0, 170, 170], dtype=torch.uint8).expand_as(image)
    assert tensor._version == version
    assert torch.equal(randomizer(image, parameters), expected)


@pytest.mark.parametrize(
    "warm_dtype,target_dtype",
    [(torch.bfloat16, None), (None, torch.bfloat16), (torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)],
)
def test_image_cache_preserves_frozen_pixels_across_autocast_context_changes(warm_dtype, target_dtype):
    randomizer = ImageRandomizer(ImageTransformConfig(mode="all"), SmudgeConfig(spot_count=(1, 1), streak_count=(1, 1)))
    parameters = randomizer.sample_parameters(1, 3, 4, seed=7)
    image = torch.arange(36, dtype=torch.uint8).reshape(1, 3, 4, 3) * 7
    offset = torch.zeros((1, 2))
    # Literal byte fixtures captured from the frozen implementation in each target context.
    expected = {
        None: [
            [[0, 0, 0], [59, 63, 70], [0, 0, 0], [0, 0, 0]],
            [[110, 114, 121], [118, 122, 129], [126, 130, 137], [136, 138, 144]],
            [[0, 0, 0], [0, 0, 0], [185, 189, 196], [0, 0, 0]],
        ],
        torch.bfloat16: [
            [[0, 0, 0], [59, 63, 69], [0, 0, 0], [0, 0, 0]],
            [[110, 114, 121], [118, 122, 129], [126, 130, 137], [136, 138, 144]],
            [[0, 0, 0], [0, 0, 0], [185, 190, 196], [0, 0, 0]],
        ],
        torch.float16: [
            [[0, 0, 0], [59, 63, 70], [0, 0, 0], [0, 0, 0]],
            [[110, 115, 121], [118, 122, 129], [126, 130, 137], [136, 137, 144]],
            [[0, 0, 0], [0, 0, 0], [185, 189, 196], [0, 0, 0]],
        ],
    }
    with torch.autocast("cpu", dtype=warm_dtype or torch.bfloat16, enabled=warm_dtype is not None):
        randomizer.apply(image, parameters, smudge_offset=offset)
    with torch.autocast("cpu", dtype=target_dtype or torch.bfloat16, enabled=target_dtype is not None):
        output = randomizer.apply(image, parameters, smudge_offset=offset)
    assert output[0].tolist() == expected[target_dtype]


def test_saturation_dtype_promotion_refreshes_hue_constants_at_the_original_evaluation_point():
    randomizer = ImageRandomizer(ImageTransformConfig(mode="none"), SmudgeConfig(enabled=False))
    parameters = randomizer.sample_parameters(1, 4, 6)
    parameters.hue_radians.fill_(torch.pi / 2)
    image = torch.tensor([255, 0, 0], dtype=torch.uint8).expand(1, 4, 6, 3).clone()
    # A quarter turn around the neutral RGB axis maps red to (1/3, 1/3+1/sqrt(3), 1/3-1/sqrt(3)).
    expected = torch.tensor([85, 232, 0], dtype=torch.uint8).expand_as(image)
    assert torch.equal(randomizer(image, parameters), expected)
    parameters.saturation = parameters.saturation.double()
    assert torch.equal(randomizer(image, parameters), expected)
    restored = ImageParameters.from_state_dict(parameters.state_dict())
    assert torch.equal(randomizer(image, restored), expected)
    restored.saturation = restored.saturation.float()
    assert torch.equal(randomizer(image, restored), expected)
