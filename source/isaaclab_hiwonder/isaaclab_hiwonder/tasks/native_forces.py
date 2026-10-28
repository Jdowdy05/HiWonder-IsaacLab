# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graph-safe Newton substep force hooks for persistent plastic creases.

Use one ``NewtonTaskForceHook`` for all environments. Install during PHYSICS_READY
before solver initialization/graph capture. Bind by actual backend deformable
ranges, not private asset handles. Reset its plastic buffers
for the same environment subset as the scene. Rendering/contact/self-contact are
owned by native Newton/VBD; these kernels add missing constitutive forces only.
"""

import numpy as np
import warp as wp


@wp.func
def _angle(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    e = wp.normalize(b - a)
    n1 = wp.normalize(wp.cross(b - a, c - a))
    n2 = wp.normalize(wp.cross(d - a, b - a))
    return wp.atan2(wp.dot(wp.cross(n1, n2), e), wp.dot(n1, n2))


@wp.kernel
def plastic_crease_forces(
    indices: wp.array2d(dtype=wp.int32),
    parameters: wp.array(dtype=wp.vec3),
    position: wp.array(dtype=wp.vec3),
    rest: wp.array(dtype=wp.float32),
    accumulated: wp.array(dtype=wp.float32),
    force: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    origin = position[indices[i, 0]]
    a = wp.vec3(0.0)
    b = position[indices[i, 1]] - origin
    c = position[indices[i, 2]] - origin
    d = position[indices[i, 3]] - origin
    angle = _angle(a, b, c, d)
    stiffness = parameters[i][0]
    yield_angle = parameters[i][1]
    hardening = parameters[i][2]
    elastic = wp.atan2(wp.sin(angle - rest[i]), wp.cos(angle - rest[i]))
    excess = wp.abs(elastic) - yield_angle - hardening * accumulated[i]
    if excess > 0.0:
        increment = excess / (1.0 + hardening)
        rest[i] = rest[i] + wp.sign(elastic) * increment
        rest[i] = wp.atan2(wp.sin(rest[i]), wp.cos(rest[i]))
        accumulated[i] = accumulated[i] + increment
    torque = -stiffness * wp.atan2(wp.sin(angle - rest[i]), wp.cos(angle - rest[i]))
    # Energy-gradient reference, preserving all rotational/translational reactions.
    eps = float(1.0e-6)
    for vertex in range(4):
        gradient = wp.vec3(0.0)
        for dim in range(3):
            shift = wp.vec3(0.0)
            shift[dim] = eps
            plus = float(0.0)
            minus = float(0.0)
            if vertex == 0:
                plus = _angle(a + shift, b, c, d)
                minus = _angle(a - shift, b, c, d)
            elif vertex == 1:
                plus = _angle(a, b + shift, c, d)
                minus = _angle(a, b - shift, c, d)
            elif vertex == 2:
                plus = _angle(a, b, c + shift, d)
                minus = _angle(a, b, c - shift, d)
            else:
                plus = _angle(a, b, c, d + shift)
                minus = _angle(a, b, c, d - shift)
            gradient[dim] = wp.atan2(wp.sin(plus - minus), wp.cos(plus - minus)) / (2.0 * eps)
        wp.atomic_add(force, indices[i, vertex], torque * gradient)


@wp.kernel
def reset_plastic_subset(
    env_ids: wp.array(dtype=wp.int32),
    selected: wp.array(dtype=wp.bool),
    rest: wp.array(dtype=wp.float32),
    accumulated: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if selected[env_ids[i]]:
        rest[i] = 0.0
        accumulated[i] = 0.0


class NewtonTaskForceHook:
    """Native callback with persistent device-side plastic state.

    ``surfaces`` contains ``(env_id, spec, particle_offset)`` from
    manager.backend.deformable_ranges. Cable frame bonds are native constraints
    authored by NativeTaskBinding before parsing.
    """

    def __init__(self, model, surfaces=()):
        self.model = model
        self.device = model.device
        indices = []
        parameters = []
        env_ids = []
        self.surface_ranges = {}
        for env_id, spec, offset in surfaces:
            if spec.kind != "plastic_shell":
                continue
            start = len(indices)
            hinges = np.asarray(spec.parameters["hinges"], dtype=np.int32)
            indices.extend((hinges + offset).tolist())
            parameters.extend(
                [(spec.parameters["hinge_stiffness"], spec.parameters["yield_angle"], 0.02)] * len(hinges)
            )
            env_ids.extend([env_id] * len(hinges))
            self.surface_ranges[env_id, spec.name] = (start, len(indices))
        self.indices = wp.array(np.asarray(indices, dtype=np.int32).reshape(-1, 4), dtype=wp.int32, device=self.device)
        self.parameters = wp.array(parameters, dtype=wp.vec3, device=self.device)
        self.env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        self.rest = wp.zeros(len(indices), device=self.device)
        self.accumulated = wp.zeros(len(indices), device=self.device)

    def __call__(self, state):
        if len(self.indices):
            wp.launch(
                plastic_crease_forces,
                dim=len(self.indices),
                inputs=[self.indices, self.parameters, state.particle_q, self.rest, self.accumulated, state.particle_f],
                device=self.device,
            )

    def register(self, manager):
        manager.register_state_force_callback(self)

    def reset(self, environment_mask):
        selected = wp.array(environment_mask, dtype=wp.bool, device=self.device)
        if len(self.indices):
            wp.launch(
                reset_plastic_subset,
                dim=len(self.indices),
                inputs=[self.env_ids, selected, self.rest, self.accumulated],
                device=self.device,
            )

    def crease_angles(self, env_id, name="sheet"):
        start, end = self.surface_ranges[env_id, name]
        return self.rest.numpy()[start:end]
