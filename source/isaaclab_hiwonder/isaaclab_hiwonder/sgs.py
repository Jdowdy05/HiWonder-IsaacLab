# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success Guided Sampling, arXiv:2610.12465v1 sections 3.2 and B.2."""

import math

import torch


def cohort_probabilities(weights: torch.Tensor, num_cohorts: int) -> torch.Tensor:
    """Actual equal-cohort allocation for IDs bound to ``id % num_cohorts`` geometry slices."""
    if weights.ndim != 1 or not 1 <= num_cohorts <= weights.numel():
        raise ValueError("Cohort weights/count must describe a nonempty bank")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("Cohort weights must be finite and nonnegative")
    missing = (-weights.numel()) % num_cohorts
    padded = (
        torch.cat((weights, torch.zeros(missing, device=weights.device, dtype=weights.dtype))) if missing else weights
    )
    masses = padded.reshape(-1, num_cohorts).sum(0)
    if torch.any(masses <= 0):
        raise ValueError("Every cohort requires positive weight")
    cohorts = torch.arange(weights.numel(), device=weights.device) % num_cohorts
    return weights / (masses[cohorts] * num_cohorts)


class SuccessGuidedSampler:
    """Fixed-bank sampler with newest-H Boolean histories and deterministic duplicate appends.

    Call ``update`` once per completed episode, before assigning replacement IDs. Interrupted
    episodes are omitted by the caller. Bank definitions and active environment IDs belong to
    the environment checkpoint, alongside this sampler's state.
    """

    def __init__(
        self,
        num_configs: int,
        history_length: int = 100,
        target: float = 0.5,
        concentration: float = 1.0,
        epsilon: float = 1e-4,
        temperature: float = 2.0,
        seed: int = 0,
        device: str = "cpu",
    ):
        if num_configs < 1 or history_length < 1:
            raise ValueError("Bank and history lengths must be positive")
        if not all(math.isfinite(x) for x in (target, concentration, epsilon, temperature)):
            raise ValueError("Sampler hyperparameters must be finite")
        if not 0 <= target <= 1 or concentration < 0 or epsilon <= 0:
            raise ValueError("Invalid target, concentration or epsilon")
        self.num_configs, self.history_length = num_configs, history_length
        self.target, self.concentration = target, concentration
        self.epsilon, self.temperature = epsilon, temperature
        self.device = torch.device(device)
        self.history = torch.zeros((num_configs, history_length), dtype=torch.bool, device=self.device)
        self.counts = torch.zeros(num_configs, dtype=torch.long, device=self.device)
        self.pointers = torch.zeros_like(self.counts)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._probability_cache = None

    def update(self, ids, successes) -> None:
        """Append all outcomes in supplied order, including simultaneous repeated IDs."""
        raw_ids = torch.as_tensor(ids, device=self.device)
        outcomes = torch.as_tensor(successes, device=self.device)
        if raw_ids.ndim != 1 or outcomes.shape != raw_ids.shape:
            raise ValueError("ids and successes must be matching one-dimensional batches")
        if raw_ids.numel() == 0:
            return
        if raw_ids.dtype == torch.bool or not torch.all(raw_ids == raw_ids.long()):
            raise ValueError("Configuration IDs must be integers")
        ids = raw_ids.long()
        if torch.any((ids < 0) | (ids >= self.num_configs)):
            raise IndexError("Configuration ID outside bank")
        if not torch.all((outcomes == 0) | (outcomes == 1)):
            raise ValueError("Completed outcomes must be Boolean")
        order = torch.argsort(ids, stable=True)
        sorted_ids = ids[order]
        unique, batch_counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        starts = torch.cumsum(batch_counts, 0) - batch_counts
        offsets = torch.arange(ids.numel(), device=self.device) - torch.repeat_interleave(starts, batch_counts)
        # Drop overwritten outcomes BEFORE indexed writes: all retained (row,column) pairs are unique.
        keep = offsets >= torch.repeat_interleave((batch_counts - self.history_length).clamp_min(0), batch_counts)
        columns = (self.pointers[sorted_ids] + offsets) % self.history_length
        self.history[sorted_ids[keep], columns[keep]] = outcomes[order][keep].bool()
        self.pointers[unique] = (self.pointers[unique] + batch_counts) % self.history_length
        self.counts[unique] = (self.counts[unique] + batch_counts).clamp_max(self.history_length)

    def success_rates(self) -> torch.Tensor:
        """Return valid-outcome means; empty histories use the target success prior."""
        means = self.history.sum(1).double() / self.counts.clamp_min(1)
        return torch.where(self.counts > 0, means, self.target)

    def probabilities(self) -> torch.Tensor:
        """Return normalized appendix-floor, temperature-adjusted beta kernel weights."""
        # Public histories/counts can change without update(); inference tensors lack mutation versions.
        cache_key = None
        if not self.history.is_inference() and not self.counts.is_inference():
            cache_key = (
                id(self.history),
                self.history._version,
                self.history.shape,
                self.history.stride(),
                self.history.dtype,
                self.history.device,
                id(self.counts),
                self.counts._version,
                self.counts.shape,
                self.counts.stride(),
                self.counts.dtype,
                self.counts.device,
                self.target,
                self.concentration,
                self.epsilon,
                self.temperature,
            )
        if (
            cache_key is not None
            and self._probability_cache is not None
            and self._probability_cache[0] == cache_key
            and torch.equal(self.history, self._probability_cache[3][0])
            and torch.equal(self.counts, self._probability_cache[3][1])
        ):
            return self._probability_cache[2].clone()
        p = self.success_rates()
        score = (p + self.epsilon).pow(self.concentration * self.target)
        score *= (1 - p + self.epsilon).pow(self.concentration * (1 - self.target))
        logits = (score.clamp_min(self.epsilon) + self.epsilon).log() / max(self.temperature, 1.0)
        probabilities = torch.softmax(logits, dim=0)
        # Owned snapshots detect alias writes that bypass versions; returned storage stays independent.
        self._probability_cache = (
            (
                cache_key,
                (self.history, self.counts),
                probabilities,
                (self.history.detach().clone(), self.counts.detach().clone()),
            )
            if cache_key is not None
            else None
        )
        return probabilities.clone()

    def sample(self, count: int) -> torch.Tensor:
        """Draw bank IDs with replacement using the checkpointed generator."""
        if count < 0:
            raise ValueError("Sample count cannot be negative")
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.multinomial(self.probabilities(), count, replacement=True, generator=self.generator)

    def state_dict(self) -> dict:
        """Take an owned snapshot, unaffected by later updates or sampling."""
        return dict(
            version=1,
            num_configs=self.num_configs,
            history_length=self.history_length,
            target=self.target,
            concentration=self.concentration,
            epsilon=self.epsilon,
            temperature=self.temperature,
            history=self.history.clone(),
            counts=self.counts.clone(),
            pointers=self.pointers.clone(),
            rng_state=self.generator.get_state().clone(),
        )

    def load_state_dict(self, state: dict) -> None:
        """Restore history, hyperparameters and generator; reject incompatible bank shapes."""
        if (
            state["version"] != 1
            or state["num_configs"] != self.num_configs
            or state["history_length"] != self.history_length
        ):
            raise ValueError("Sampler checkpoint schema/bank mismatch")
        history = state["history"].to(self.device).clone()
        counts = state["counts"].to(self.device).clone()
        pointers = state["pointers"].to(self.device).clone()
        if (
            history.shape != self.history.shape
            or counts.shape != self.counts.shape
            or pointers.shape != self.pointers.shape
        ):
            raise ValueError("Corrupt sampler state shapes")
        if torch.any((counts < 0) | (counts > self.history_length)) or torch.any(
            (pointers < 0) | (pointers >= self.history_length)
        ):
            raise ValueError("Corrupt sampler ring state")
        checked = SuccessGuidedSampler(
            self.num_configs,
            self.history_length,
            state["target"],
            state["concentration"],
            state["epsilon"],
            state["temperature"],
            device=str(self.device),
        )
        checked.generator.set_state(state["rng_state"].cpu())
        self.history, self.counts, self.pointers = history, counts, pointers
        self.target, self.concentration = checked.target, checked.concentration
        self.epsilon, self.temperature = checked.epsilon, checked.temperature
        self.generator = checked.generator
        self._probability_cache = None
