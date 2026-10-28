# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Free six-DOF cap with a disengaging, reciprocal screw constraint.

This is an equivalent contact/constraint model, not a mesh-resolved thread. It
does not change body pose or joint enable flags. Engagement is triggered by the
measured thread-entry geometry; after runout no cap constraint wrench remains.
"""

import numpy as np

from .api import rotation_matrix
from .mechanics import screw_reaction


class ScrewCap:
    def __init__(self, parameters):
        self.parameters = parameters
        self.angle = 0.0
        self.previous_yaw = 0.0
        self.engaged = parameters["operation"] == "open"
        self.unthreaded = False
        self.broken = False
        self.grasp_intervals = 0
        self.grasp_time = 0.0
        self.release_time = 0.1
        self.interval_counted = False

    def observe(self, state, dt=0.0):
        cap, bottle = state.actors["cap"], state.actors["bottle"]
        bottle_rotation = rotation_matrix(bottle.orientation)
        cap_rotation = rotation_matrix(cap.orientation)
        relative_rotation = bottle_rotation.T @ cap_rotation
        yaw = float(np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0]))
        delta = float(np.arctan2(np.sin(yaw - self.previous_yaw), np.cos(yaw - self.previous_yaw)))
        self.angle += delta
        self.previous_yaw = yaw
        local = bottle_rotation.T @ (np.asarray(cap.position) - bottle.position) - self.parameters["seat_offset"]
        velocity = bottle_rotation.T @ (np.asarray(cap.linear_velocity) - bottle.linear_velocity)
        angular = bottle_rotation.T @ (np.asarray(cap.angular_velocity) - bottle.angular_velocity)
        tilt = float(np.arccos(np.clip(relative_rotation[2, 2], -1, 1)))
        radial = float(np.linalg.norm(local[:2]))
        if not self.engaged and radial < 0.0015 and tilt < 0.1 and 0.0085 < local[2] < 0.0105 and abs(yaw) < 0.25:
            self.engaged = True
            self.angle = 2 * np.pi * self.parameters["thread_turns"] + yaw
        if self.engaged:
            residual = local[2] - self.parameters["pitch"] * self.angle / (2 * np.pi)
            if radial > 0.015 or abs(residual) > 0.012 or tilt > 0.6:
                self.broken = True
            outward = velocity[2] > 0.002 or angular[2] > 0.1 or self.parameters["operation"] == "open"
            if (
                local[2] > self.parameters["open_height"]
                and self.angle > 2 * np.pi * self.parameters["thread_turns"] - 0.25
                and outward
            ):
                self.engaged = False
                self.unthreaded = True
        grasp = self.engaged and state.touching("cap", "left_finger") and state.touching("cap", "right_finger")
        if dt > 0:
            if grasp:
                self.grasp_time += dt
                if self.grasp_time >= 0.1 and not self.interval_counted:
                    self.grasp_intervals += 1
                    self.interval_counted = True
                self.release_time = 0.0
            else:
                self.release_time += dt
                if self.release_time >= 0.05:
                    self.grasp_time = 0.0
                    self.interval_counted = False
        state.mechanisms.update(
            cap_angle=self.angle,
            cap_height=float(local[2]),
            cap_radial_error=radial,
            cap_tilt=tilt,
            cap_engaged=float(self.engaged),
            cap_unthreaded=float(self.unthreaded),
            cap_broken=float(self.broken),
            cap_grasp_intervals=float(self.grasp_intervals),
        )
        return local, velocity, angular, bottle_rotation, cap_rotation

    def forces(self, state, dt=0.0):
        local, velocity, angular, bottle_rotation, cap_rotation = self.observe(state, dt)
        if not self.engaged or self.broken:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        force_z, torque_z = screw_reaction(self.angle, local[2], angular[2], velocity[2], self.parameters["pitch"])
        force_local = -1500 * local - 2 * velocity
        force_local[2] = force_z
        # A physical closed stop prevents over-tightening beyond the seated angle.
        torque_z += max(-self.angle, 0) * 0.05 - 0.001 * angular[2]
        force = bottle_rotation @ force_local
        tilt_torque = 0.08 * np.cross(cap_rotation[:, 2], bottle_rotation[:, 2])
        tilt_torque -= bottle_rotation @ np.r_[0.002 * angular[:2], 0.0]
        torque = tilt_torque + bottle_rotation[:, 2] * torque_z
        scale = min(1.0, 8.0 / max(np.linalg.norm(force), 1e-12), 0.15 / max(np.linalg.norm(torque), 1e-12))
        force *= scale
        torque *= scale
        # Equal opposite reaction at the cap origin conserves total moment.
        arm = np.asarray(state.actors["cap"].position) - state.actors["bottle"].position
        return force, torque, -force, -torque - np.cross(arm, force)

    def state_dict(self):
        return {
            name: getattr(self, name)
            for name in (
                "angle",
                "previous_yaw",
                "engaged",
                "unthreaded",
                "broken",
                "grasp_intervals",
                "grasp_time",
                "release_time",
                "interval_counted",
            )
        }

    def load_state_dict(self, value):
        for name in self.state_dict():
            setattr(self, name, value[name])
