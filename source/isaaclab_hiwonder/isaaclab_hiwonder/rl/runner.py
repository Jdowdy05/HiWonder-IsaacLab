# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stock PPO rollout/update with atomic environment/sampler/RNG checkpoint boundaries."""

import copy
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


class HiWonderRunner(OnPolicyRunner):
    """Keep RSL-RL's actual consumed observation storage and PPO updates unchanged.

    Checkpoints are taken only after a complete PPO rollout/update. Environment state owns
    its SGS bank/hash/history/active IDs, nuisance streams, temporal/image state and physics
    snapshots. Algorithm, optimizer and RNG states restore exactly; physical replay may
    differ within backend tolerances because native solver warm-start caches are not saved.
    """

    def __init__(
        self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", metadata: dict | None = None
    ):
        self._inside_learn = False
        self.metadata = copy.deepcopy(metadata or {})
        config = copy.deepcopy(train_cfg)
        if config["obs_groups"]["actor"] != ["policy_image", "proprio", "command"] or config["obs_groups"][
            "critic"
        ] != ["critic"]:
            raise ValueError("HiWonder actor input groups must exclude privileged critic state")
        obs = env.get_observations()
        mode = config["hiwonder"]["actor_mode"]
        expected_rank = 4 if mode == "rgb" else 2
        expected_channels = 3 * config["hiwonder"].get("frame_stack", 1)
        if obs["policy_image"].ndim != expected_rank or (
            mode == "rgb" and obs["policy_image"].shape[1] != expected_channels
        ):
            raise ValueError(
                "Actor input must be declared BCHW RGB frame stack or the explicitly selected state_debug vector"
            )
        if env.num_envs * config["num_steps_per_env"] < config["algorithm"]["num_mini_batches"] * 2:
            raise ValueError("PPO requires at least two rollout samples per minibatch")
        super().__init__(env, config, log_dir, device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Train with stock storage; maintain the next iteration index across saved/resumed calls."""
        if num_learning_iterations < 1:
            raise ValueError("num_learning_iterations must be positive")
        self._inside_learn = True
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self._inside_learn = False
        self.current_learning_iteration += 1

    def save(self, path: str, infos: dict | None = None) -> None:
        """Atomically save learner, environment, RNG and provenance at an optimizer boundary."""
        if self.alg.storage.step != 0:
            raise ValueError("Checkpoint only after a complete PPO rollout/update")
        base = self.env.unwrapped
        state = self.alg.save()
        state.update(
            iter=self.current_learning_iteration,
            infos=infos,
            hiwonder=dict(
                version=1,
                next_iteration=self.current_learning_iteration + int(self._inside_learn),
                environment=base.state_dict(),
                runner_config=copy.deepcopy(self.cfg),
                metadata=copy.deepcopy(self.metadata),
                rng=dict(
                    torch_cpu=torch.get_rng_state(),
                    torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    numpy=np.random.get_state(),
                    python=random.getstate(),
                ),
            ),
        )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(state, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self.logger.save_model(str(destination), self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        restore_environment: bool = True,
        restore_rng: bool = True,
    ) -> dict:
        """Resume local trusted checkpoints; policy-only evaluation leaves held-out resets untouched."""
        state = torch.load(path, weights_only=False, map_location=map_location or self.device)
        checkpoint = state.get("hiwonder")
        if checkpoint is None or checkpoint["version"] != 1:
            raise ValueError("Checkpoint lacks compatible HiWonder environment state")
        previous_sources = checkpoint.get("metadata", {}).get("source_sha256", {})
        current_sources = self.metadata.get("source_sha256", {})
        changed = [
            name
            for name in previous_sources.keys() & current_sources.keys()
            if previous_sources[name] != current_sources[name]
        ]
        if restore_environment and changed:
            warnings.warn(
                "Source changed since checkpoint: "
                + ", ".join(sorted(changed))
                + ". Physical bank compatibility is checked separately; "
                "revalidate changed reward/force/outcome semantics.",
                RuntimeWarning,
                stacklevel=2,
            )
        if restore_environment and self.alg.storage.step != 0:
            raise ValueError("Cannot replace environment during a partial PPO rollout")
        if restore_environment:
            # Environment validates bank, nuisance schema and geometry before accepting its state.
            self.env.unwrapped.load_state_dict(checkpoint["environment"])
        load_iteration = self.alg.load(state, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = checkpoint["next_iteration"]
        if restore_rng:
            rng = checkpoint["rng"]
            torch.set_rng_state(rng["torch_cpu"].cpu())
            if rng["torch_cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["torch_cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        return state["infos"]

    def load_policy(self, path: str, map_location: str | None = None) -> None:
        """Load acting weights without training sampler, optimizer, iteration or RNG restoration."""
        self.load(
            path,
            load_cfg=dict(actor=True, critic=False, optimizer=False, iteration=False, rnd=False),
            map_location=map_location,
            restore_environment=False,
            restore_rng=False,
        )

    def initialize_weights(self, path: str, map_location: str | None = None) -> None:
        """Initialize both models on a fresh cohort, keeping its optimizer, sampler, iteration and RNG."""
        self.load(
            path,
            load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False),
            map_location=map_location,
            restore_environment=False,
            restore_rng=False,
        )
