# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic RGB augmentation with reusable episode records and strict black padding."""

import math
from dataclasses import dataclass, fields, replace

import torch
import torch.nn.functional as F

from .config import ImageTransformConfig, SmudgeConfig


def _tensor_cache_key(tensor: torch.Tensor) -> tuple | None:
    # Inference tensors have no version counter, so their derived data cannot be reused safely.
    if tensor.is_inference():
        return None
    return (id(tensor), tensor._version, tensor.shape, tensor.stride(), tensor.dtype, tensor.device)


def _autocast_cache_key(device: torch.device) -> tuple:
    return (device.type, torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))


@dataclass
class ImageParameters:
    """Owned tensors sufficient to replay augmentation without sampling any additional noise.

    ``matrix`` maps normalized output pixel coordinates to source coordinates; compose it
    with camera intrinsics when projecting landmarks through the augmentation. The RGB
    effects are image-space approximations of lens contamination, not optical simulation.
    """

    height: int
    width: int
    enabled: torch.Tensor
    matrix: torch.Tensor
    canvas_matrices: torch.Tensor
    brightness: torch.Tensor
    contrast: torch.Tensor
    saturation: torch.Tensor
    hue_radians: torch.Tensor
    smudge_alpha: torch.Tensor
    smudge_tint: torch.Tensor
    haze: torch.Tensor
    local_contrast: torch.Tensor
    blur: torch.Tensor

    def state_dict(self) -> dict:
        return {
            f.name: value.clone() if isinstance(value := getattr(self, f.name), torch.Tensor) else value
            for f in fields(self)
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: str | torch.device = "cpu") -> "ImageParameters":
        return cls(
            **{
                key: value.to(device).clone() if isinstance(value, torch.Tensor) else value
                for key, value in state.items()
            }
        )


class ImageRandomizer:
    """Apply deterministic uint8 NHWC RGB transforms; image shape and input remain unchanged."""

    def __init__(self, config: ImageTransformConfig | None = None, smudge: SmudgeConfig | None = None):
        self.config = config or ImageTransformConfig()
        self.smudge = smudge or SmudgeConfig()
        if self.config.mode not in ("none", "one", "all"):
            raise ValueError("Image mode must be none, one or all")
        # One entry per kind bounds memory across episode replacements and image resolutions.
        self._geometry_cache = None
        self._hue_cache = None
        self._luminance_cache = None
        self._hue_constants_cache = None
        self._smudge_grid_cache = None

    def sample_from_snapshot(
        self, snapshot: dict, height: int, width: int, device: str | torch.device | None = None
    ) -> ImageParameters:
        """Construct independent image/smudge records from authoritative per-env episode state.

        Image transforms consume only image.seed. Lens patterns consume only smudge.seed,
        with the exact sampled spot/streak counts and haze. Use manager.image_config and
        manager.smudge_config in the constructor to honor disabled-group profiles.
        """
        image_seed = snapshot["image"]["seed"]
        device = torch.device(device or image_seed.device)
        smudge_state = snapshot["smudge"]
        image_only = ImageRandomizer(self.config, replace(self.smudge, enabled=False))
        records = []
        smudge_fields = ("smudge_alpha", "smudge_tint", "haze", "local_contrast", "blur")
        for i in range(len(image_seed)):
            record = image_only.sample_parameters(1, height, width, seed=int(image_seed[i]), device=device)
            spots = int(smudge_state["spot_count"][i])
            streaks = int(smudge_state["streak_count"][i])
            haze = float(smudge_state["haze"][i])
            profile = replace(
                self.smudge, spot_count=(spots, spots), streak_count=(streaks, streaks), haze=(haze, haze)
            )
            lens = ImageRandomizer(ImageTransformConfig(mode="none"), profile).sample_parameters(
                1, height, width, seed=int(smudge_state["seed"][i]), device=device
            )
            for name in smudge_fields:
                setattr(record, name, getattr(lens, name))
            records.append(record)
        if not records:
            raise ValueError("Snapshot must contain at least one environment")
        values = {}
        for item in fields(ImageParameters):
            first = getattr(records[0], item.name)
            values[item.name] = (
                torch.cat([getattr(record, item.name) for record in records])
                if isinstance(first, torch.Tensor)
                else first
            )
        return ImageParameters(**values)

    def sample_parameters(
        self, batch_size: int, height: int, width: int, seed: int = 0, device: str | torch.device = "cpu"
    ) -> ImageParameters:
        """Sample once per episode/frame and retain the record for PPO observation replay."""
        if batch_size < 1 or min(height, width) < 1:
            raise ValueError("Batch size and image dimensions must be positive")
        device = torch.device(device)
        rng = torch.Generator(device=device).manual_seed(seed)

        def uniform(bounds, shape):
            if bounds[1] < bounds[0]:
                raise ValueError("Randomization range is reversed")
            return torch.rand(shape, device=device, generator=rng) * (bounds[1] - bounds[0]) + bounds[0]

        cfg, smudge = self.config, self.smudge
        available = torch.tensor(
            [cfg.shear_enabled, cfg.crop_enabled, cfg.rotation_enabled, cfg.color_enabled], device=device
        )
        enabled = available.expand(batch_size, 4).clone()
        if cfg.mode == "none":
            enabled.zero_()
        elif cfg.mode == "one":
            weights = torch.tensor(cfg.transform_weights, device=device) * available
            if torch.any(weights < 0) or not torch.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError("one mode requires positive enabled transform weights")
            choices = torch.multinomial(weights, batch_size, replacement=True, generator=rng)
            enabled.zero_().scatter_(1, choices[:, None], True)
        shear = torch.tan(uniform(cfg.shear_degrees, (batch_size,)) * math.pi / 180) * enabled[:, 0]
        crop = torch.where(enabled[:, 1], uniform(cfg.crop_scale, (batch_size,)), 1.0)
        if torch.any((crop <= 0) | (crop > 1)):
            raise ValueError("crop_scale must be in (0,1]")
        offset = (torch.rand((batch_size, 2), device=device, generator=rng) * 2 - 1) * (1 - crop[:, None])
        angle = uniform(cfg.rotation_degrees, (batch_size,)) * math.pi / 180 * enabled[:, 2]
        cosine, sine = angle.cos(), angle.sin()
        # Pixel-aspect correction preserves Euclidean rotation in non-square image coordinates.
        rotation = torch.eye(3, device=device).repeat(batch_size, 1, 1)
        rotation[:, 0, 0], rotation[:, 1, 1] = cosine, cosine
        rotation[:, 0, 1], rotation[:, 1, 0] = -sine * height / width, sine * width / height
        shearing = torch.eye(3, device=device).repeat(batch_size, 1, 1)
        shearing[:, 0, 1] = shear * height / width
        cropping = torch.eye(3, device=device).repeat(batch_size, 1, 1)
        cropping[:, 0, 0], cropping[:, 1, 1] = crop, crop
        cropping[:, :2, 2] = offset
        matrix = cropping @ shearing @ rotation

        def color_factor(bounds):
            return torch.where(enabled[:, 3], uniform(bounds, (batch_size,)), 1.0)

        brightness = color_factor(cfg.brightness)
        contrast = color_factor(cfg.contrast)
        saturation = color_factor(cfg.saturation)
        hue = uniform(cfg.hue_radians, (batch_size,)) * enabled[:, 3]
        alpha = torch.zeros((batch_size, height, width), device=device)
        tint = torch.zeros((batch_size, 3), device=device)
        haze = torch.zeros(batch_size, device=device)
        local_contrast = torch.ones(batch_size, device=device)
        blur = torch.zeros(batch_size, device=device)
        if smudge.enabled:
            y, x = torch.meshgrid(
                (torch.arange(height, device=device) + 0.5) / height,
                (torch.arange(width, device=device) + 0.5) / width,
                indexing="ij",
            )
            for streak, bounds in ((False, smudge.spot_count), (True, smudge.streak_count)):
                counts = torch.randint(bounds[0], bounds[1] + 1, (batch_size,), device=device, generator=rng)
                for j in range(bounds[1]):
                    centers = torch.rand((batch_size, 2), device=device, generator=rng)
                    radius = uniform(smudge.radius, (batch_size, 1, 1))
                    anisotropy = uniform(smudge.anisotropy, (batch_size, 1, 1))
                    theta = uniform((0.0, math.pi), (batch_size, 1, 1))
                    dx, dy = x - centers[:, 0, None, None], y - centers[:, 1, None, None]
                    u, v = dx * theta.cos() + dy * theta.sin(), -dx * theta.sin() + dy * theta.cos()
                    rx = radius * (3.0 if streak else 1.0)
                    ry = radius * anisotropy * (0.15 if streak else 1.0)
                    spot = torch.exp(-0.5 * ((u / rx.clamp_min(1e-6)) ** 2 + (v / ry.clamp_min(1e-6)) ** 2))
                    spot *= uniform(smudge.opacity, (batch_size, 1, 1)) * (counts > j)[:, None, None]
                    alpha = 1 - (1 - alpha) * (1 - spot)
            tint = uniform(smudge.tint, (batch_size, 3))
            haze = uniform(smudge.haze, (batch_size,))
            local_contrast = uniform(smudge.local_contrast, (batch_size,))
            blur = uniform(smudge.blur_pixels, (batch_size,)).clamp(0, 2) / 2
        return ImageParameters(
            height,
            width,
            enabled,
            matrix,
            torch.stack((rotation, shearing @ rotation), dim=1),
            brightness,
            contrast,
            saturation,
            hue,
            alpha,
            tint,
            haze,
            local_contrast,
            blur,
        )

    def apply(
        self,
        images: torch.Tensor,
        parameters: ImageParameters,
        return_mask: bool = False,
        smudge_offset: torch.Tensor | None = None,
    ):
        """Return pixels and optional valid mask. Retain explicit Bx2 normalized smudge offsets for replay."""
        p = parameters
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Expected uint8 NHWC RGB images")
        if images.shape[:3] != (p.enabled.shape[0], p.height, p.width) or images.device != p.matrix.device:
            raise ValueError("Image shape/device differs from sampled parameters")
        source = images.permute(0, 3, 1, 2).float() / 255
        autocast_key = _autocast_cache_key(images.device)
        matrix_key, canvas_key = _tensor_cache_key(p.matrix), _tensor_cache_key(p.canvas_matrices)
        geometry_key = (source.shape, p.height, p.width, matrix_key, canvas_key, autocast_key)
        cacheable = matrix_key is not None and canvas_key is not None
        if (
            not cacheable
            or self._geometry_cache is None
            or self._geometry_cache[0] != geometry_key
            or not torch.equal(p.matrix, self._geometry_cache[4][0])
            or not torch.equal(p.canvas_matrices, self._geometry_cache[4][1])
        ):
            grid = F.affine_grid(p.matrix[:, :2], source.shape, align_corners=False)
            valid = (grid[..., 0].abs() <= 1 - 1 / p.width + 1e-6) & (grid[..., 1].abs() <= 1 - 1 / p.height + 1e-6)
            # Crop can map an invalid rotated/sheared canvas pixel back into the original source.
            # Preserve each intermediate canvas domain as well as the fused source domain.
            for canvas in p.canvas_matrices.unbind(1):
                canvas_grid = F.affine_grid(canvas[:, :2], source.shape, align_corners=False)
                valid &= (canvas_grid[..., 0].abs() <= 1 - 1 / p.width + 1e-6) & (
                    canvas_grid[..., 1].abs() <= 1 - 1 / p.height + 1e-6
                )
            # Snapshots catch writes through external aliases, which do not advance mutation versions.
            self._geometry_cache = (
                (
                    geometry_key,
                    (p.matrix, p.canvas_matrices),
                    grid,
                    valid,
                    (p.matrix.detach().clone(), p.canvas_matrices.detach().clone()),
                )
                if cacheable
                else None
            )
        else:
            _, _, grid, valid, _ = self._geometry_cache
        pixels = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        alpha = p.smudge_alpha[:, None]
        if smudge_offset is not None:
            if smudge_offset.shape != (images.shape[0], 2) or smudge_offset.device != images.device:
                raise ValueError("Smudge offsets must have shape [B,2] on the image device")
            smudge_grid_key = (alpha.shape, images.device, torch.get_default_dtype(), autocast_key)
            if self._smudge_grid_cache is None or self._smudge_grid_cache[0] != smudge_grid_key:
                identity = torch.eye(3, device=images.device).expand(images.shape[0], 3, 3)
                smudge_grid = F.affine_grid(identity[:, :2], alpha.shape, align_corners=False)
                self._smudge_grid_cache = (smudge_grid_key, smudge_grid)
            smudge_grid = self._smudge_grid_cache[1].clone()
            smudge_grid -= 2 * smudge_offset[:, None, None]
            alpha = F.grid_sample(alpha, smudge_grid, align_corners=False, padding_mode="zeros")
        blurry = F.avg_pool2d(F.pad(pixels, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        pixels = pixels + (blurry - pixels) * alpha * p.blur[:, None, None, None]
        local_mean = F.avg_pool2d(F.pad(pixels, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        pixels = pixels + (local_mean - pixels) * alpha * (1 - p.local_contrast[:, None, None, None])
        pixels = pixels * (1 - alpha) + p.smudge_tint[:, :, None, None] * alpha
        pixels = (
            pixels * (1 - p.haze[:, None, None, None]) + p.smudge_tint[:, :, None, None] * p.haze[:, None, None, None]
        )
        luminance_key = (pixels.device, pixels.dtype, torch.get_default_dtype(), autocast_key)
        if self._luminance_cache is None or self._luminance_cache[0] != luminance_key:
            luminance_weights = pixels.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
            self._luminance_cache = (luminance_key, luminance_weights)
        luminance = (pixels * self._luminance_cache[1]).sum(1, keepdim=True)
        pixels = luminance + (pixels - luminance) * p.saturation[:, None, None, None]
        # Saturation may promote pixels, so hue constants use the resulting dtype.
        constants_key = (pixels.device, pixels.dtype, torch.get_default_dtype(), autocast_key)
        if self._hue_constants_cache is None or self._hue_constants_cache[0] != constants_key:
            axis = pixels.new_tensor([1, 1, 1]) / math.sqrt(3)
            skew = pixels.new_tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]) / math.sqrt(3)
            identity = torch.eye(3, device=pixels.device)
            self._hue_constants_cache = (
                constants_key,
                axis[:, None] @ axis[None, :],
                skew,
                identity,
            )
        _, axis_outer, skew, identity = self._hue_constants_cache
        # Rotation around the neutral RGB axis changes hue while leaving neutral gray unchanged.
        hue_key = _tensor_cache_key(p.hue_radians)
        hue_cache_key = (constants_key, hue_key)
        if (
            hue_key is None
            or self._hue_cache is None
            or self._hue_cache[0] != hue_cache_key
            or not torch.equal(p.hue_radians, self._hue_cache[3])
        ):
            cos, sin = p.hue_radians.cos()[:, None, None], p.hue_radians.sin()[:, None, None]
            hue_matrix = cos * identity + (1 - cos) * axis_outer + sin * skew
            self._hue_cache = (
                (hue_cache_key, p.hue_radians, hue_matrix, p.hue_radians.detach().clone())
                if hue_key is not None
                else None
            )
        else:
            hue_matrix = self._hue_cache[2]
        pixels = torch.einsum("bij,bjhw->bihw", hue_matrix, pixels)
        mean = pixels.mean((1, 2, 3), keepdim=True)
        pixels = (pixels - mean) * p.contrast[:, None, None, None] + mean
        pixels *= p.brightness[:, None, None, None]
        # Apply mask last: haze, additive contrast and tint may otherwise color the uncovered border.
        output = (pixels.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1)
        output = output.masked_fill(~valid[..., None], 0)
        return (output, valid.clone()) if return_mask else output

    def __call__(self, images: torch.Tensor, parameters: ImageParameters) -> torch.Tensor:
        return self.apply(images, parameters)
