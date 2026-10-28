# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named privileged-vector packing, with an exported schema and no silent truncation."""

import torch


def pack_privileged_rows(
    rows: list[list[tuple[str, object]]],
    capacity: int = 256,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, list[dict[str, tuple[int, int]]]]:
    """Pack independent host rows before one device transfer, with owned output storage.

    Scene/task summaries already live on the host. Device fields must be gathered
    explicitly by the caller instead of hiding a synchronization per small field.
    Rows may have different named layouts; the established capacity guard applies
    to each complete row before any result is published.
    """
    if not rows:
        raise ValueError("Privileged rows must be nonempty")
    packed, layouts = [], []
    for row in rows:
        parts = []
        for name, value in row:
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise ValueError("Privileged row fields must be on the CPU")
            parts.append((name, torch.as_tensor(value, dtype=torch.float32, device="cpu").reshape(1, -1)))
        tensor, layout = pack_privileged_observations(parts, capacity)
        packed.append(tensor)
        layouts.append(layout)
    return torch.cat(packed, dim=0).to(device=device, dtype=dtype), layouts


def pack_privileged_observations(
    parts: list[tuple[str, torch.Tensor]], capacity: int = 256
) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """Pack ordered batched fields and return their explicit half-open index ranges."""
    if not parts or capacity < 1:
        raise ValueError("Privileged observations require fields and positive capacity")
    first = parts[0][1]
    batch = first.shape[0]
    layout = {}
    flattened = []
    offset = 0
    for name, values in parts:
        if name in layout:
            raise ValueError(f"Duplicate privileged field {name}")
        if values.shape[0] != batch or values.device != first.device:
            raise ValueError("Privileged field batch/device mismatch")
        flat = values.reshape(batch, -1)
        end = offset + flat.shape[1]
        layout[name] = (offset, end)
        offset = end
        flattened.append(flat)
    if offset > capacity:
        raise ValueError(
            f"Privileged fields need {offset} values, exceeding capacity {capacity}; increase declared critic width"
        )
    result = torch.zeros((batch, capacity), device=first.device, dtype=torch.float32)
    result[:, :offset] = torch.cat(flattened, -1)
    return result, layout
