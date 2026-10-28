# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Supported VBD primitives implement the scalar robot motor model.

VBD 1.6 ignores native effort/speed limits, joint friction and armature. This
adapter uses stabilized PD feedforward torques, a no-load speed torque envelope,
smooth Coulomb friction, and a generalized inertial drive. The latter adds
0.5*A*(q_next-q-dt*qd)^2/dt^2 without changing link mass or inertia. Compensation
preserves that energy when VBD clamps its drive target to a position limit.
The speed envelope limits motor-driven motion; external loads can backdrive it.
These are explicit servo approximations, not identified HX-30HM dynamics.
"""

import re

import newton
import numpy as np
import warp as wp

from ..assets.registration import JOINT_NAMES
from .api import rotation_matrix


@wp.kernel
def _motor_step(
    q_indices: wp.array(dtype=int),
    v_indices: wp.array(dtype=int),
    t_indices: wp.array(dtype=int),
    parameters: wp.array2d(dtype=float),
    inertia_floor: wp.array(dtype=float),
    targets: wp.array(dtype=float),
    feedforward: wp.array(dtype=float),
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    lower: wp.array(dtype=float),
    upper: wp.array(dtype=float),
    dt: float,
    ke: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    target_q: wp.array(dtype=float),
    target_qd: wp.array(dtype=float),
    force: wp.array(dtype=float),
    applied: wp.array(dtype=float),
):
    i = wp.tid()
    qi, vi, ti = q_indices[i], v_indices[i], t_indices[i]
    position, velocity = q[qi], qd[vi]
    effort, vmax = parameters[i, 2], wp.max(parameters[i, 3], 1.0e-8)
    kp, damping = parameters[i, 0], parameters[i, 1]
    inertia = wp.max(parameters[i, 5] + inertia_floor[i], 1.0e-10)
    # Backward-Euler scalar rotor prediction avoids explicit high-damping
    # chatter at tiny jaw inertias. Coupled/contact loads remain with VBD.
    denominator = 1.0 + (damping * dt + kp * dt * dt) / inertia
    torque = (kp * (targets[i] - position) - (damping + kp * dt) * velocity + feedforward[i]) / denominator
    torque = wp.clamp(torque, -effort, effort)
    # Braking remains available; only torque accelerating towards no-load speed
    # is reduced. A zero speed setting disables active motor driving.
    if torque * velocity >= 0.0:
        allowance = effort * wp.max(0.0, 1.0 - wp.abs(velocity) / vmax)
        torque = wp.clamp(torque, -allowance, allowance)
    if parameters[i, 3] <= 0.0:
        torque = 0.0
    friction = parameters[i, 4] * wp.tanh(velocity / 0.001)
    stopping = inertia * wp.abs(velocity) / dt
    friction = wp.clamp(friction, -stopping, stopping)
    inertia_ke = parameters[i, 5] / (dt * dt)
    predicted = position + dt * velocity
    clipped = wp.clamp(predicted, lower[vi], upper[vi])
    ke[vi] = inertia_ke
    kd[vi] = 0.0
    target_q[ti] = clipped
    target_qd[vi] = 0.0
    force[vi] = torque - friction + inertia_ke * (predicted - clipped)
    applied[i] = torque


@wp.kernel
def _set_parameters(rows: wp.array(dtype=int), values: wp.array2d(dtype=float), destination: wp.array2d(dtype=float)):
    i, j = wp.tid()
    destination[rows[i], j] = values[i, j]


class VBDMotorAdapter:
    """One row per controlled REVOLUTE joint; properties are [kp,kd,tau,v,mu,A]."""

    def __init__(self, model, joints):
        self.model = model
        joints = np.asarray(joints, dtype=np.int32).reshape(-1)
        if not len(joints) or np.any(model.joint_type.numpy()[joints] != int(newton.JointType.REVOLUTE)):
            raise ValueError("Native robot motors require nonempty scalar revolute joint indices")
        self.joints = joints
        self.q_indices = wp.array(model.joint_q_start.numpy()[joints], dtype=int, device=model.device)
        self.v_indices = wp.array(model.joint_qd_start.numpy()[joints], dtype=int, device=model.device)
        self.t_indices = wp.array(model.joint_target_q_start.numpy()[joints], dtype=int, device=model.device)
        self.parameters = wp.zeros((len(joints), 6), dtype=float, device=model.device)
        # A lower inertia bound from the two bodies' COM angular inertias.
        # Translational leverage and the rest of the chain can only add inertia.
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        poses, frames = model.body_q.numpy(), model.joint_X_p.numpy()
        inverse = model.body_inv_inertia.numpy()
        axes, dofs = model.joint_axis.numpy(), model.joint_qd_start.numpy()
        floors = []
        for joint in joints:
            rotation = rotation_matrix(frames[joint, 3:])
            if parent[joint] >= 0:
                rotation = rotation_matrix(poses[parent[joint], 3:]) @ rotation
            axis = rotation @ axes[dofs[joint]]
            inverse_inertia = 0.0
            for body in (parent[joint], child[joint]):
                if body >= 0:
                    local = rotation_matrix(poses[body, 3:]).T @ axis
                    inverse_inertia += local @ inverse[body] @ local
            floors.append(1 / max(inverse_inertia, 1e-8))
        self.inertia_floor = wp.array(floors, dtype=float, device=model.device)
        self.targets = wp.zeros(len(joints), dtype=float, device=model.device)
        self.feedforward = wp.zeros(len(joints), dtype=float, device=model.device)
        self.applied_effort = wp.zeros(len(joints), dtype=float, device=model.device)
        dofs = model.joint_qd_start.numpy()[joints]
        self.set_parameters(
            np.stack(
                [
                    model.joint_target_ke.numpy()[dofs],
                    model.joint_target_kd.numpy()[dofs],
                    model.joint_effort_limit.numpy()[dofs],
                    model.joint_velocity_limit.numpy()[dofs],
                    model.joint_friction.numpy()[dofs],
                    model.joint_armature.numpy()[dofs],
                ],
                axis=1,
            )
        )
        self.set_targets(model.joint_q.numpy()[model.joint_q_start.numpy()[joints]])

    @classmethod
    def robot(cls, model):
        """Map all six CAD motors per environment without assuming model DOF order."""
        selected = {}
        for joint, label in enumerate(model.joint_label):
            match = re.search(r"/env_(\d+)/Room/Robot/", label)
            name = label.rsplit("/", 1)[-1]
            if match and name in JOINT_NAMES:
                selected[int(match.group(1)), name] = joint
        if not selected:
            return None
        environments = sorted({key[0] for key in selected})
        if environments != list(range(len(environments))) or len(selected) != len(environments) * len(JOINT_NAMES):
            raise ValueError("Native robot joint mapping is incomplete")
        return cls(model, [selected[i, name] for i in environments for name in JOINT_NAMES])

    def set_parameters(self, values, rows=None):
        values = np.asarray(values, dtype=np.float32).reshape(-1, 6)
        rows = np.arange(len(self.joints), dtype=np.int32) if rows is None else np.asarray(rows, dtype=np.int32)
        if len(rows) != len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Motor parameters must be finite nonnegative rows matching selected joints")
        if len(set(rows.tolist())) != len(rows) or np.any(rows < 0) or np.any(rows >= len(self.joints)):
            raise ValueError("Motor parameter rows must be unique valid indices")
        wp.launch(
            _set_parameters,
            dim=values.shape,
            inputs=[wp.array(rows, dtype=int, device=self.model.device), wp.array(values, device=self.model.device)],
            outputs=[self.parameters],
            device=self.model.device,
        )

    def set_targets(self, values, feedforward=None):
        def copy(destination, source):
            if hasattr(source, "detach"):
                source = wp.from_torch(source.detach().contiguous().reshape(-1))
                if len(source) != len(destination):
                    raise ValueError("Native target count differs from motor count")
                wp.copy(destination, source)
            else:
                source = np.asarray(source, dtype=np.float32).reshape(-1)
                if len(source) != len(destination) or not np.isfinite(source).all():
                    raise ValueError("Native targets must be finite and match motor count")
                destination.assign(source)

        copy(self.targets, values)
        if feedforward is None:
            self.feedforward.zero_()
        else:
            copy(self.feedforward, feedforward)

    def prepare(self, state, control, dt):
        if dt <= 0:
            raise ValueError("Native motor timestep must be positive")
        wp.launch(
            _motor_step,
            dim=len(self.joints),
            inputs=[
                self.q_indices,
                self.v_indices,
                self.t_indices,
                self.parameters,
                self.inertia_floor,
                self.targets,
                self.feedforward,
                state.joint_q,
                state.joint_qd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                dt,
            ],
            outputs=[
                self.model.joint_target_ke,
                self.model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                control.joint_f,
                self.applied_effort,
            ],
            device=self.model.device,
        )
