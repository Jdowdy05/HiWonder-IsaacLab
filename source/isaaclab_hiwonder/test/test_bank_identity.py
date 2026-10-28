# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

from isaaclab_hiwonder.sim.checkpoint import task_definition_hash
from isaaclab_hiwonder.tasks import build_task


def test_bank_identity_changes_when_native_material_or_rest_mesh_changes():
    task = build_task("barrel_jack", seed=7)
    original = task_definition_hash(task)
    changed = copy.deepcopy(task)
    changed.deformables[0].cable_material.twist_modulus_pa *= 2
    assert task_definition_hash(changed) != original
    changed = copy.deepcopy(task)
    changed.deformables[0].vertices[1, 1] += 0.001
    assert task_definition_hash(changed) != original
    assert task_definition_hash(copy.deepcopy(task)) == original


def test_accepted_structural_cohort_hash_tracks_baked_geometry_but_excludes_fresh_motor_noise():
    from isaaclab_hiwonder.randomization import RandomizationManager
    from isaaclab_hiwonder.sim.checkpoint import structural_snapshot_hash

    snapshot = RandomizationManager(1, seed=4).snapshot()
    original = structural_snapshot_hash(snapshot)
    snapshot["actuator"]["stiffness"] += 1
    assert structural_snapshot_hash(snapshot) == original
    snapshot["table"]["width"] += 0.1
    assert structural_snapshot_hash(snapshot) != original
