# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge Newton 1.6 VBD's public force collector to SDK contact sensors.

SolverVBD exposes collect_rigid_contact_forces but inherits an unimplemented
update_contacts. This local subclass supplies that interface using actual solved
contact state. No dependency files are patched and no contact forces are guessed.
The adapter also publishes measured generalized coordinates from VBD's maximal
body state for the SDK's joint sensors, actuator feedback and FREE-root views.
"""

import newton
import numpy as np
import warp as wp
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from newton.solvers import SolverVBD

from .vbd_motors import VBDMotorAdapter


@wp.kernel
def _snapshot_contact_history(
    previous: wp.array(dtype=wp.transform),
    current: wp.array(dtype=wp.transform),
    world: wp.array(dtype=wp.int32),
    rebaseline: wp.array(dtype=wp.bool),
    world_count: int,
    output: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    index = world[i]
    if index < 0:
        index = world_count
    output[i] = previous[i]
    if rebaseline[index]:
        output[i] = current[i]


class ContactReportingVBD(SolverVBD):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.contact_previous_pose = wp.clone(model.body_q)
        self.contact_rebaseline = wp.ones(model.world_count + 1, dtype=wp.bool, device=model.device)
        self.contact_dt = 1 / 240
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        types = model.joint_type.numpy()
        membership = np.full(model.body_count, -1, dtype=np.int32)
        adjacency = {}
        for parent, child in zip(parents, children):
            adjacency.setdefault(int(parent), []).append(int(child))
        for joint, (parent, child, kind) in enumerate(zip(parents, children, types)):
            if parent != -1 or kind != int(newton.JointType.FIXED):
                continue
            pending = [int(child)]
            seen = set()
            while pending:
                body = pending.pop()
                if body in seen:
                    continue
                seen.add(body)
                membership[body] = joint
                pending.extend(adjacency.get(body, []))
        self.fixed_root_membership = wp.array(membership, dtype=wp.int32, device=model.device)
        self.rest_before_root_reset = wp.clone(model.body_q)
        self.robot_motors = VBDMotorAdapter.robot(model)

    def sync_fixed_root_rest(self, world_mask=None):
        """Move reference poses with reset anchors, preserving internal rest geometry."""
        if not self.model.body_count:
            return
        wp.copy(self.rest_before_root_reset, self.model.body_q)
        wp.launch(
            _move_fixed_root_reference,
            dim=self.model.body_count,
            inputs=[
                self.fixed_root_membership,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.rest_before_root_reset,
                self.model.body_world,
                world_mask,
                self.model.world_count,
                self.model.body_q,
            ],
            device=self.device,
        )
        self.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES | newton.ModelFlags.JOINT_PROPERTIES)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask=world_mask, flags=flags)
        # VBD's public reset contract rebaselines rigid history regardless of flags.
        if world_mask is None:
            self.contact_rebaseline.fill_(True)
        else:
            # Multiple indexed resets may precede the next step; retain their union.
            wp.launch(
                _merge_reset_mask,
                dim=len(self.contact_rebaseline),
                inputs=[world_mask, self.contact_rebaseline],
                device=self.device,
            )

    def step(self, state_in, state_out, control, contacts, dt):
        if self.robot_motors is not None:
            self.robot_motors.prepare(state_in, control, dt)
        if self.model.body_count:
            wp.launch(
                _snapshot_contact_history,
                dim=self.model.body_count,
                inputs=[
                    self.body_q_prev,
                    state_in.body_q,
                    self.model.body_world,
                    self.contact_rebaseline,
                    self.model.world_count,
                    self.contact_previous_pose,
                ],
                device=self.device,
            )
        self.contact_dt = dt
        super().step(state_in, state_out, control, contacts, dt)
        # Publish generalized state from VBD's solved maximal body coordinates.
        # SDK actuator feedback and FREE-body root views consume these arrays.
        if self.model.joint_count:
            newton.eval_ik(self.model, state_out, state_out.joint_q, state_out.joint_qd)
        self.contact_rebaseline.zero_()

    def update_contacts(self, contacts, state=None):
        if state is None:
            raise ValueError("VBD contact force reporting requires the post-step state")
        self.collect_rigid_contact_forces(state.body_q, self.contact_previous_pose, contacts, self.contact_dt)


@wp.kernel
def _merge_reset_mask(source: wp.array(dtype=wp.bool), target: wp.array(dtype=wp.bool)):
    i = wp.tid()
    target[i] = target[i] or source[i]


class NewtonHiWonderVBDManager(NewtonVBDManager):
    @classmethod
    def get_robot_motors(cls):
        """Return this local solver's graph-safe, generalized motor adapter."""
        return cls._solver.robot_motors

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        return ContactReportingVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg))

    @classmethod
    def _eval_fk_impl(cls, world_reset_mask, fk_mask):
        cls._solver.sync_fixed_root_rest(world_reset_mask)
        super()._eval_fk_impl(world_reset_mask, fk_mask)


@wp.kernel
def _move_fixed_root_reference(
    membership: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    parent_frames: wp.array(dtype=wp.transform),
    child_frames: wp.array(dtype=wp.transform),
    previous: wp.array(dtype=wp.transform),
    body_world: wp.array(dtype=wp.int32),
    selected: wp.array(dtype=wp.bool),
    world_count: int,
    reference: wp.array(dtype=wp.transform),
):
    body = wp.tid()
    joint = membership[body]
    world = body_world[body]
    if world < 0:
        world = world_count
    active = bool(True)
    if selected:
        active = selected[world]
    if joint >= 0 and active:
        root = joint_child[joint]
        desired = wp.transform_multiply(parent_frames[joint], wp.transform_inverse(child_frames[joint]))
        delta = wp.transform_multiply(desired, wp.transform_inverse(previous[root]))
        reference[body] = wp.transform_multiply(delta, previous[body])
