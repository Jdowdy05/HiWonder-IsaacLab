# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 5.5.1 configuration for native-resolution wrist-RGB asymmetric PPO."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PPOTrainingConfig:
    """PPO uses the consumed RGB tensor directly; no resize or epoch augmentation is performed."""

    actor_mode: Literal["rgb", "state_debug"] = "rgb"
    frame_stack: int = 1
    num_steps_per_env: int = 16
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    hidden_dims: tuple[int, ...] = (256, 128)
    cnn_channels: tuple[int, ...] = (16, 32, 64)
    # The pinned CNN padding formula can be negative for kernel==stride on nondivisible inputs.
    # Kernels >= 2*stride-1 keep every intermediate image dimension valid without resizing.
    cnn_kernel_size: tuple[int, ...] = (8, 7, 3)
    cnn_stride: tuple[int, ...] = (4, 4, 2)
    learning_rate: float = 3e-4
    init_action_std: float = 0.35
    entropy_coef: float = 0.005
    gamma: float = 0.99
    lam: float = 0.95
    save_interval: int = 50
    seed: int = 0
    # RSL-RL autocasts only PPO updates to BF16; parameters and environment inputs stay FP32.
    use_mixed_precision: bool = False

    def __post_init__(self):
        if self.actor_mode not in ("rgb", "state_debug"):
            raise ValueError("actor_mode must be rgb or state_debug")
        if not isinstance(self.frame_stack, int) or self.frame_stack < 1:
            raise ValueError("frame_stack must be a positive integer")
        if min(self.num_steps_per_env, self.num_learning_epochs, self.num_mini_batches, self.save_interval) < 1:
            raise ValueError("PPO rollout/epoch/minibatch/save counts must be positive")
        if not self.hidden_dims or min(self.hidden_dims) < 1:
            raise ValueError("PPO hidden dimensions must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or self.init_action_std <= 0:
            raise ValueError("PPO learning rate/action std must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError("Invalid PPO discount or GAE parameter")
        if not isinstance(self.use_mixed_precision, bool):
            raise ValueError("use_mixed_precision must be a bool")

    def to_runner_config(self) -> dict:
        """Return an owned dictionary accepted by the pinned OnPolicyRunner/PPO constructors."""
        actor = dict(
            class_name="rsl_rl.models:CNNModel" if self.actor_mode == "rgb" else "rsl_rl.models:MLPModel",
            hidden_dims=list(self.hidden_dims),
            activation="elu",
            obs_normalization=False,
            distribution_cfg=dict(
                class_name="rsl_rl.modules.distribution:GaussianDistribution",
                init_std=self.init_action_std,
                std_type="log",
                std_range=(0.02, 1.0),
            ),
        )
        if self.actor_mode == "rgb":
            actor["cnn_cfg"] = dict(
                output_channels=list(self.cnn_channels),
                kernel_size=list(self.cnn_kernel_size),
                stride=list(self.cnn_stride),
                padding="zeros",
                norm="none",
                activation="elu",
                global_pool="none",
                flatten=True,
            )
        return dict(
            seed=self.seed,
            num_steps_per_env=self.num_steps_per_env,
            save_interval=self.save_interval,
            check_for_nan=True,
            logger="tensorboard",
            torch_compile_mode=None,
            obs_groups=dict(actor=["policy_image", "proprio", "command"], critic=["critic"]),
            actor=actor,
            critic=dict(
                class_name="rsl_rl.models:MLPModel",
                hidden_dims=list(self.hidden_dims),
                activation="elu",
                obs_normalization=False,
                distribution_cfg=None,
            ),
            algorithm=dict(
                class_name="rsl_rl.algorithms:PPO",
                num_learning_epochs=self.num_learning_epochs,
                num_mini_batches=self.num_mini_batches,
                learning_rate=self.learning_rate,
                schedule="adaptive",
                desired_kl=0.01,
                clip_param=0.2,
                gamma=self.gamma,
                lam=self.lam,
                value_loss_coef=1.0,
                entropy_coef=self.entropy_coef,
                max_grad_norm=1.0,
                use_clipped_value_loss=True,
                normalize_advantage_per_mini_batch=False,
                use_mixed_precision=self.use_mixed_precision,
            ),
            hiwonder=dict(
                actor_mode=self.actor_mode,
                frame_stack=self.frame_stack,
                image_resize=None,
                epoch_augmentation=False,
                learner_precision="bf16_autocast" if self.use_mixed_precision else "fp32",
            ),
        )
