# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch
from isaaclab_hiwonder.sgs import SuccessGuidedSampler


def test_numeric_distribution_uses_valid_counts_and_appendix_floor():
    sampler = SuccessGuidedSampler(4, concentration=12, temperature=0.2)
    torch.testing.assert_close(sampler.probabilities(), torch.full((4,), 0.25, dtype=torch.float64))
    sampler.update([0, 1, 1, 2], [False, False, True, True])
    # Independent scalar reference: the unseen fourth row uses target p=.5.
    weights = [max((p + 1e-4) ** 6 * (1 - p + 1e-4) ** 6, 1e-4) + 1e-4 for p in [0, 0.5, 1, 0.5]]
    expected = torch.tensor([w / sum(weights) for w in weights], dtype=torch.float64)
    torch.testing.assert_close(sampler.probabilities(), expected)
    assert sampler.success_rates().tolist() == [0, 0.5, 1, 0.5]


def test_duplicate_terminal_ids_keep_all_outcomes_across_wraparound_and_resume():
    sampler = SuccessGuidedSampler(2, history_length=3, seed=71)
    sampler.update([1, 0, 1, 1, 1, 1], [False, True, True, False, True, True])
    assert sampler.success_rates().tolist() == [1, pytest.approx(2 / 3)]
    state = sampler.state_dict()
    clone = SuccessGuidedSampler(2, history_length=3)
    clone.load_state_dict(state)
    assert torch.equal(sampler.sample(300), clone.sample(300))
    sampler.update([1], [False])
    assert sampler.success_rates()[1].item() == pytest.approx(2 / 3)
    assert clone.success_rates()[1].item() == pytest.approx(2 / 3)
    sampler.update([1], [False])
    assert sampler.success_rates()[1].item() == pytest.approx(1 / 3)
    assert state["counts"].tolist() == [1, 3]


@pytest.mark.parametrize("ids,successes", [([-1], [True]), ([2], [False]), ([0], [float("nan")]), ([0], [0.5])])
def test_invalid_terminal_batch_is_rejected_without_history_mutation(ids, successes):
    sampler = SuccessGuidedSampler(2)
    with pytest.raises((ValueError, IndexError)):
        sampler.update(ids, successes)
    assert sampler.state_dict()["counts"].sum().item() == 0


def test_actual_cohort_distribution_normalizes_each_fixed_geometry_slice():
    from isaaclab_hiwonder.sgs import cohort_probabilities

    weights = torch.tensor([0.1, 0.2, 0.1, 0.5, 0.1], dtype=torch.float64)
    actual = cohort_probabilities(weights, 2)
    expected = torch.tensor([1 / 6, 1 / 7, 1 / 6, 5 / 14, 1 / 6], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    assert actual[::2].sum().item() == pytest.approx(0.5)
    assert actual[1::2].sum().item() == pytest.approx(0.5)


def test_probability_replay_avoids_recomputing_the_distribution_and_returns_owned_values():
    """Repeated diagnostics and sampling should reuse unchanged success-kernel work."""
    sampler = SuccessGuidedSampler(3, history_length=3, concentration=12)
    sampler.update([0, 1, 1], [False, False, True])
    first = sampler.probabilities()
    first.zero_()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True) as profile:
        replay = sampler.probabilities()
    # Independent scalar weights for p=[0,.5,.5], default temperature=2.
    weights = [(max((p + 1e-4) ** 6 * (1 - p + 1e-4) ** 6, 1e-4) + 1e-4) ** 0.5 for p in [0, 0.5, 0.5]]
    torch.testing.assert_close(replay, torch.tensor([w / sum(weights) for w in weights], dtype=torch.float64))
    assert not any(event.key == "aten::_softmax" for event in profile.key_averages())


@pytest.mark.parametrize(
    "change,expected",
    [
        ("history", [0.43463134644690093, 0.43463134644690093, 0.1307373071061981]),
        ("counts", [0.6714391790638375, 0.20221875896067604, 0.1263420619754863]),
        ("target", [0.10374608588574587, 0.3449000288837969, 0.5513538852304573]),
        ("concentration", [0.36534647535237935, 0.3173267623238103, 0.3173267623238103]),
        ("epsilon", [0.4230023420383288, 0.3083767743668521, 0.268620883594819]),
        ("temperature", [0.4118152691605644, 0.3522005601732215, 0.23598417066621422]),
        ("replace_history", [1 / 3, 1 / 3, 1 / 3]),
        ("replace_counts", [0.6697435558502621, 0.21515569165100112, 0.11510075249873679]),
        ("numpy_history", [0.43463134644690093, 0.43463134644690093, 0.1307373071061981]),
        ("numpy_counts", [0.6714391790638375, 0.20221875896067604, 0.1263420619754863]),
    ],
)
def test_probability_cache_observes_each_public_dependency_independently(change, expected):
    # Literal outputs captured from the frozen implementation before optimization.
    sampler = SuccessGuidedSampler(3, history_length=4, target=0.25, concentration=4, epsilon=0.01, temperature=1)
    sampler.update([0] * 4 + [1] * 4 + [2] * 4, [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    sampler.probabilities()
    if change == "history":
        sampler.history[0, 1] = True
    elif change == "counts":
        sampler.counts[1] = 3
    elif change == "replace_history":
        sampler.history = torch.zeros_like(sampler.history)
    elif change == "replace_counts":
        sampler.counts = torch.full_like(sampler.counts, 3)
    elif change == "numpy_history":
        sampler.history.numpy()[0, 1] = True
    elif change == "numpy_counts":
        sampler.counts.numpy()[1] = 3
    else:
        setattr(sampler, change, {"target": 0.75, "concentration": 8, "epsilon": 0.1, "temperature": 3}[change])
    torch.testing.assert_close(sampler.probabilities(), torch.tensor(expected, dtype=torch.float64))


def test_probability_state_load_and_inference_mutation_refresh_owned_distribution():
    sampler = SuccessGuidedSampler(3, history_length=3, concentration=12, temperature=1)
    sampler.probabilities()
    sampler.history[0, 0] = True
    sampler.counts[0] = 1
    sampler.target = 0.25
    sampler.concentration = 8
    sampler.epsilon = 0.01
    sampler.temperature = 3
    weights = [(max((p + 0.01) ** 2 * (1 - p + 0.01) ** 6, 0.01) + 0.01) ** (1 / 3) for p in [1, 0.25, 0.25]]
    expected = torch.tensor([w / sum(weights) for w in weights], dtype=torch.float64)
    torch.testing.assert_close(sampler.probabilities(), expected)
    sampler.history = torch.zeros_like(sampler.history)
    sampler.counts = torch.zeros_like(sampler.counts)
    torch.testing.assert_close(sampler.probabilities(), torch.full((3,), 1 / 3, dtype=torch.float64))
    state = sampler.state_dict()
    state["history"][0, 0] = True
    state["counts"][0] = 1
    with torch.inference_mode():
        sampler.load_state_dict(state)
        torch.testing.assert_close(sampler.probabilities(), expected)
        sampler.history.zero_()
        sampler.counts.zero_()
        torch.testing.assert_close(sampler.probabilities(), torch.full((3,), 1 / 3, dtype=torch.float64))
