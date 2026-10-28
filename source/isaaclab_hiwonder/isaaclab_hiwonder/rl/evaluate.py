# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Uniform held-out policy evaluation with actual task-defined terminal outcomes."""

import math
from collections import defaultdict

import torch


@torch.inference_mode()
def evaluate_held_out(env, policy, episodes: int = 100, max_steps: int | None = None) -> dict:
    """Measure deterministic actor success on a separately constructed uniform held-out environment.

    This function never loads a training sampler or substitutes reward/proximity for success.
    Native auto-reset terminal outcomes are paired with configuration IDs captured before
    stepping. Each fixed structural cohort has a preallocated episode quota; faster
    terminations cannot increase that cohort's weight in the uniform-bank score.
    """
    base = env.unwrapped
    if base.cfg.hiwonder_sampling != "uniform" or base.cfg.hiwonder_split != "heldout":
        raise ValueError("Evaluation requires sampling=uniform, split=heldout and an independent seed")
    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive")
    bank_size = base.sampler.num_configs
    if bank_size < env.num_envs:
        raise ValueError("Evaluation bank must contain every structural cohort")
    cohort_sizes = [(bank_size + env.num_envs - 1 - i) // env.num_envs for i in range(env.num_envs)]
    weights = [size / bank_size for size in cohort_sizes]
    shares = [episodes * weight for weight in weights]
    quotas = [math.floor(share) for share in shares]
    remainder_order = sorted(range(env.num_envs), key=lambda i: (-(shares[i] - quotas[i]), i))
    for i in remainder_order[: episodes - sum(quotas)]:
        quotas[i] += 1
    if max_steps is None:
        max_steps = 2 * max(quotas) * int(env.max_episode_length)
    if max_steps < 1:
        raise ValueError("Evaluation step budget must be positive")
    policy.eval()
    obs = env.get_observations()
    outcomes = []
    by_config = defaultdict(list)
    by_family = defaultdict(list)
    summed_rewards = torch.zeros(env.num_envs, device=env.device)
    cohort_outcomes = [[] for _ in range(env.num_envs)]
    cohort_returns = [[] for _ in range(env.num_envs)]
    steps = 0
    for steps in range(1, max_steps + 1):
        previous_ids = base.config_ids.clone()
        previous_families = base.reset_family_ids.clone()
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions.to(env.device))
        summed_rewards += rewards
        ended = (dones > 0).nonzero(as_tuple=False).flatten()
        if not len(ended):
            continue
        if "episode_success" in extras:
            success = torch.as_tensor(extras["episode_success"], device=env.device)[ended].bool()
        elif "completed_env_ids" in extras and "completed_successes" in extras:
            ids = torch.as_tensor(extras["completed_env_ids"], device=env.device)
            values = torch.as_tensor(extras["completed_successes"], device=env.device).bool()
            mapping = {int(i): bool(value) for i, value in zip(ids.tolist(), values.tolist())}
            if not all(int(i) in mapping for i in ended):
                raise ValueError("Missing task-defined terminal success for completed evaluation episodes")
            success = torch.tensor([mapping[int(i)] for i in ended], device=env.device)
        else:
            raise ValueError("Environment must provide task-defined terminal successes for evaluation")
        for env_id, passed in zip(ended.tolist(), success.tolist()):
            if len(cohort_outcomes[env_id]) < quotas[env_id]:
                outcomes.append(passed)
                cohort_outcomes[env_id].append(passed)
                by_config[int(previous_ids[env_id])].append(passed)
                by_family[base.reset_family_names[int(previous_families[env_id])]].append(passed)
                value = float(summed_rewards[env_id])
                cohort_returns[env_id].append(value)
        summed_rewards[ended] = 0
        if len(outcomes) >= episodes:
            break
    successes = sum(outcomes)
    complete = len(outcomes) == episodes and all(cohort_outcomes)
    success_rate = (
        sum(weight * sum(values) / len(values) for weight, values in zip(weights, cohort_outcomes))
        if complete
        else None
    )
    mean_return = (
        sum(weight * sum(values) / len(values) for weight, values in zip(weights, cohort_returns)) if complete else None
    )
    return dict(
        sampling="uniform",
        split="heldout",
        aggregation="uniform_configuration_bank",
        requested_episodes=episodes,
        completed_episodes=len(outcomes),
        success_count=successes,
        success_rate=success_rate,
        mean_return=mean_return,
        sample_success_rate=successes / len(outcomes) if outcomes else None,
        cohort_coverage=sum(weight for weight, values in zip(weights, cohort_outcomes) if values),
        configuration_coverage=len(by_config) / base.sampler.num_configs,
        steps=steps,
        budget_exhausted=len(outcomes) < episodes,
        by_cohort={
            str(i): dict(
                bank_configurations=cohort_sizes[i],
                weight=weights[i],
                requested_episodes=quotas[i],
                completed_episodes=len(values),
                successes=sum(values),
                success_rate=sum(values) / len(values) if values else None,
                mean_return=sum(cohort_returns[i]) / len(values) if values else None,
            )
            for i, values in enumerate(cohort_outcomes)
        },
        by_configuration={
            str(key): dict(episodes=len(values), successes=sum(values), success_rate=sum(values) / len(values))
            for key, values in sorted(by_config.items())
        },
        by_reset_family={
            key: dict(episodes=len(values), successes=sum(values), success_rate=sum(values) / len(values))
            for key, values in sorted(by_family.items())
        },
    )
