# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific geometry, mechanics, reset configurations and measured outcomes."""

from .api import ActorState, BodySpec, CableMaterialSpec, DeformableSpec, JointSpec, SceneState, ShapeSpec, TaskDefinition, TaskResult
from .catalog import GARMENTS, KNOT_VARIANTS, TASK_NAMES, build_task
from .outcomes import TaskEpisode
