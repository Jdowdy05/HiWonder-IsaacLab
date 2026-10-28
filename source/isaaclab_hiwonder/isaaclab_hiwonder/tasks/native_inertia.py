# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve geometric rod inertia across Newton's absolute inertia regularizer.

Only explicitly mapped task rods are restored. Validation uses double precision;
the native model remains float32. Install before SolverVBD construction so its
body-property caches start from the restored tensors.
"""

import re

import numpy as np


def capture_cable_inertias(builder, expected_segments):
    """Capture physically valid source tensors before ModelBuilder.finalize."""
    records = {}
    for root, count in expected_segments.items():
        matches = [
            (index, label)
            for index, label in enumerate(builder.body_label)
            if label.startswith(root + "/") and re.search(r"_edge_body_\d+$", label)
        ]
        if len(matches) != count:
            raise ValueError(f"Expected {count} native rod bodies below {root}, found {len(matches)}")
        for index, label in matches:
            inertia = np.asarray(builder.body_inertia[index], dtype=np.float64).reshape(3, 3).copy()
            mass = float(builder.body_mass[index])
            eigenvalues = np.linalg.eigvalsh(inertia)
            if (
                not np.isfinite(inertia).all()
                or not np.isfinite(mass)
                or mass <= 0
                or not np.allclose(inertia, inertia.T, rtol=1e-7, atol=1e-20)
                or np.any(eigenvalues <= 0)
                or eigenvalues[2] > (eigenvalues[0] + eigenvalues[1]) * (1 + 1e-6)
            ):
                raise ValueError(f"Invalid authored physical inertia for {label}")
            if label in records:
                raise ValueError(f"Duplicate native rod mapping: {label}")
            records[label] = dict(mass=mass, inertia=inertia)
    return records


def restore_cable_inertias(model, records):
    """Restore only recorded rods through public model arrays; return an audit."""
    import warp as wp

    if not records:
        return {"body_count": 0, "bodies": []}
    labels = {label: index for index, label in enumerate(model.body_label)}
    inertia = model.body_inertia.numpy().copy()
    inverse = model.body_inv_inertia.numpy().copy()
    mass = model.body_mass.numpy().copy()
    inverse_mass = model.body_inv_mass.numpy()
    report = []
    for label, record in records.items():
        if label not in labels:
            raise ValueError(f"Finalized model lost native rod {label}")
        index = labels[label]
        original = record["inertia"]
        if not np.isclose(mass[index], record["mass"], rtol=2e-6, atol=0):
            raise ValueError(f"Native rod mass changed during finalization: {label}")
        regularized = np.linalg.eigvalsh(inertia[index].astype(np.float64))
        inertia[index] = original.astype(np.float32)
        inverse[index] = np.linalg.inv(original).astype(np.float32) if inverse_mass[index] > 0 else 0
        if inverse_mass[index] > 0:
            np.testing.assert_allclose(inertia[index].astype(np.float64) @ inverse[index], np.eye(3), atol=2e-6)
        report.append(
            dict(
                body=label,
                mass_kg=float(mass[index]),
                authored_eigenvalues_kg_m2=np.linalg.eigvalsh(original).tolist(),
                regularized_eigenvalues_kg_m2=regularized.tolist(),
                restored_eigenvalues_kg_m2=np.linalg.eigvalsh(inertia[index].astype(np.float64)).tolist(),
            )
        )
    wp.copy(model.body_inertia, wp.array(inertia, dtype=model.body_inertia.dtype, device=model.device))
    wp.copy(model.body_inv_inertia, wp.array(inverse, dtype=model.body_inv_inertia.dtype, device=model.device))
    np.testing.assert_array_equal(model.body_mass.numpy(), mass)
    restored = model.body_inertia.numpy()
    for label, record in records.items():
        np.testing.assert_array_equal(restored[labels[label]], record["inertia"].astype(np.float32))
    return {
        "status": "AUTHORED_GEOMETRIC_ROD_INERTIA_RESTORED_NOT_HARDWARE_CALIBRATED",
        "body_count": len(report),
        "masses_preserved": True,
        "bodies": report,
    }
