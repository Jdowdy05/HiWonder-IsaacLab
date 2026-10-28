# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from isaaclab_hiwonder.sim.observations import pack_privileged_observations


def test_named_privileged_schema_preserves_joint_dynamics_and_rejects_truncation():
    parts = [
        ("q", torch.zeros(2, 6)),
        ("actuator/stiffness", torch.full((2, 6), 25.0)),
        ("base/xy", torch.tensor([[0.1, 0.2], [-0.1, 0.3]])),
    ]
    value, layout = pack_privileged_observations(parts, capacity=16)
    assert layout == {"q": (0, 6), "actuator/stiffness": (6, 12), "base/xy": (12, 14)}
    assert value[0].tolist() == pytest.approx([0] * 6 + [25] * 6 + [0.1, 0.2, 0, 0])
    assert value[1, 12:14].tolist() == pytest.approx([-0.1, 0.3])
    with pytest.raises(ValueError, match="capacity"):
        pack_privileged_observations(parts + [("object/pose", torch.ones(2, 7))], capacity=16)


def test_host_rows_preserve_distinct_layouts_values_and_observation_ownership():
    from isaaclab_hiwonder.sim import observations

    assert callable(getattr(observations, "pack_privileged_rows", None)), "Missing batched host-row packing"
    joints = torch.tensor([1.0, 2.0])
    rows = [[("q", joints), ("contact", 3)], [("goal", [4.0, 5.0, 6.0])]]
    first, layouts = observations.pack_privileged_rows(rows, capacity=5)
    assert first.tolist() == [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]
    assert layouts == [{"q": (0, 2), "contact": (2, 3)}, {"goal": (0, 3)}]
    joints[0] = 9
    second, _ = observations.pack_privileged_rows(rows, capacity=5)
    assert first[0, 0] == 1 and second[0, 0] == 9
    second.fill_(-1)
    assert joints.tolist() == [9, 2] and first[1].tolist() == [4, 5, 6, 0, 0]


def test_host_rows_reject_device_fields_and_never_silently_truncate():
    from isaaclab_hiwonder.sim import observations

    assert callable(getattr(observations, "pack_privileged_rows", None)), "Missing batched host-row packing"
    with pytest.raises(ValueError, match="CPU"):
        observations.pack_privileged_rows([[("q", torch.empty(2, device="meta"))]])
    with pytest.raises(ValueError, match="capacity"):
        observations.pack_privileged_rows([[("goal", [1, 2, 3])]], capacity=2)
    with pytest.raises(ValueError, match="Duplicate"):
        observations.pack_privileged_rows([[("x", 1), ("x", 2)]])
    with pytest.raises(ValueError, match="nonempty"):
        observations.pack_privileged_rows([])


def test_host_packing_ignores_global_default_device():
    from isaaclab_hiwonder.sim.observations import pack_privileged_rows

    with torch.device("meta"):
        value, _ = pack_privileged_rows([[("goal", [1, 2])]], capacity=3, device="cpu")
    assert value.device.type == "cpu" and value.tolist() == [[1, 2, 0]]


def test_host_row_output_dtype_keeps_original_float32_field_rounding():
    from isaaclab_hiwonder.sim.observations import pack_privileged_rows

    value, _ = pack_privileged_rows([[("goal", [16777217.0])]], capacity=2, dtype=torch.float64)
    assert value.dtype == torch.float64
    assert value.tolist() == [[16777216.0, 0.0]]
