# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-derived task outcomes with continuous dwell, release and contact evidence."""

import numpy as np

from .api import SceneState, TaskDefinition, TaskResult, rotation_matrix
from .mechanics import close_rope, knot_determinant, ring_crossings


class TaskEpisode:
    """Per-environment outcome history. Construct afresh on each real reset."""

    def __init__(self, task: TaskDefinition):
        self.task = task
        self.elapsed = 0.0
        self.held = 0.0
        self.stage = 0
        self.stage_held = 0.0
        self.previous_drive = None
        self.previous_gear = None
        self.gear_rotation = 0.0
        self.mesh_rotation = 0.0
        self.mesh_error = 0.0
        self.previous_rope_positions = None
        self.roll_only_violated = False

    def evaluate(self, state: SceneState, dt: float, action_cost: float = 0.0) -> TaskResult:  # noqa: C901
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Outcome timestep must be finite and positive")
        self.elapsed += dt
        task, p, metrics = self.task, self.task.parameters, {}
        dynamic_names = [b.name for b in task.bodies if not b.static and b.shapes]
        finite = all(
            np.all(
                np.isfinite(
                    np.concatenate(
                        [
                            np.asarray(s.position),
                            np.asarray(s.orientation),
                            np.asarray(s.linear_velocity),
                            np.asarray(s.angular_velocity),
                        ]
                    )
                )
            )
            for s in state.actors.values()
        )
        finite = finite and all(np.all(np.isfinite(v)) for v in state.deformable_positions.values())
        finite = finite and all(np.all(np.isfinite(v)) for v in state.deformable_velocities.values())
        escaped = any(
            np.asarray(state.actors[n].position)[2] < -0.03
            or np.linalg.norm(np.asarray(state.actors[n].position)[:2]) > 0.48
            for n in dynamic_names
        )
        escaped = escaped or any(
            np.any(nodes[:, 2] < -0.03) or np.any(np.linalg.norm(nodes[:, :2], axis=1) > 0.48)
            for nodes in state.deformable_positions.values()
        )
        if not finite:
            return TaskResult(-1, False, True, True, False, {"failure": -1}, {"nonfinite": 1})

        def steady(label):
            actor = state.actors[label]
            return np.linalg.norm(actor.linear_velocity) < 0.015 and np.linalg.norm(actor.angular_velocity) < 0.2

        def released(label):
            return not any(state.touching(label, jaw) for jaw in ("gripper", "left_finger", "right_finger"))

        def seated(label, target, tolerance=0.001, angle=0.08, keyed=False):
            actor = state.actors[label]
            distance = float(np.linalg.norm(np.asarray(actor.position) - target))
            rot = rotation_matrix(actor.orientation)
            angular = (
                float(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
                if keyed
                else float(np.arccos(np.clip(rot[2, 2], -1, 1)))
            )
            metrics[f"{label}/position_error"] = distance
            metrics[f"{label}/angle_error"] = angular
            return distance < tolerance and angular < angle

        success = False
        progress = 0.0
        if task.name in ("peg_insert", "connector_insert", "barrel_jack"):
            label = p["moving"]
            geometry = seated(label, p["seat"], p["position_tolerance"], p["angle_tolerance"], p["keyed"])
            success = geometry and state.touching(label, p["fixture"]) and released(label) and steady(label)
            progress = np.exp(-metrics[f"{label}/position_error"] / 0.035)
            if task.name == "barrel_jack":
                nodes = state.deformable_positions["cable"]
                attachment = task.deformables[0].attachments[0]
                target = (
                    np.asarray(attachment["parent_frame"][:3]) @ rotation_matrix(state.actors[label].orientation).T
                    + state.actors[label].position
                )
                error = float(np.linalg.norm(nodes[0] - target))
                metrics["attachment_error"] = error
                # Native cable material frames, measured from the rod's first rigid segment.
                segment = state.actors.get("cable/segment_0")
                orientation_error = np.inf
                if segment is not None:
                    expected = rotation_matrix(state.actors[label].orientation) @ rotation_matrix(
                        p["attachment_orientation"]
                    )
                    relative = expected.T @ rotation_matrix(segment.orientation)
                    orientation_error = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
                metrics["attachment_angle_error"] = orientation_error
                success = success and error < 0.002 and orientation_error < 0.1

        elif task.name == "cube_stack":
            conditions = []
            for i, label in enumerate(p["order"]):
                target = np.asarray(p["base"]) + (0, 0, i * p["size"])
                geometry = seated(label, target, 0.003, 0.12)
                support = "table" if i == 0 else p["order"][i - 1]
                conditions.append(geometry and state.touching(label, support) and steady(label) and released(label))
            progress = float(np.mean(conditions))
            success = all(conditions)

        elif task.name == "nut_pick_place":
            contained = []
            for label in p["objects"]:
                pos = np.asarray(state.actors[label].position)
                inside = np.all(np.abs(pos[:2] - p["center"]) < np.asarray(p["half_extent"]) - 0.006)
                contained.append(
                    inside
                    and p["floor"] < pos[2] < p["rim"]
                    and steady(label)
                    and released(label)
                    and state.touching(label, "container")
                )
            metrics["contained_count"] = sum(contained)
            progress = float(np.mean(contained))
            success = all(contained)

        elif task.name == "washer_ring":
            actor = state.actors["washer"]
            distance = float(np.linalg.norm(np.asarray(actor.position) - p["seat"]))
            axis = rotation_matrix(actor.orientation)[:, 2]
            alignment = abs(float(np.dot(axis, p["ring_tangent"])))
            # At the designated bottom seat the hoop centerline must pass through the washer aperture.
            success = (
                distance < 0.0015
                and alignment > 0.99
                and state.touching("washer", "ring")
                and steady("washer")
                and released("washer")
            )
            metrics.update(seat_error=distance, tangent_alignment=alignment)
            progress = np.exp(-distance / 0.03) * alignment

        elif task.name == "shape_sort":
            conditions = []
            for index, (label, target) in enumerate(p["targets"].items()):
                rotation = rotation_matrix(state.actors[label].orientation)
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                symmetry = p["symmetry_orders"][index]
                yaw_error = abs(np.arctan2(np.sin(symmetry * yaw), np.cos(symmetry * yaw))) / symmetry
                conditions.append(
                    seated(label, target, 0.0015, 0.09)
                    and yaw_error < 0.09
                    and state.touching(label, f"slot_{index}")
                    and steady(label)
                    and released(label)
                )
            progress = float(np.mean(conditions))
            success = all(conditions)

        elif task.name == "bottle_cap":
            m = state.mechanisms
            angle = m["cap_angle"]
            enough_grasps = m["cap_grasp_intervals"] >= p["minimum_grasp_intervals"]
            residual = m["cap_height"] - p["pitch"] * angle / (2 * np.pi)
            metrics.update(m)
            metrics["screw_residual"] = residual
            escaped = escaped or bool(m["cap_broken"])
            if p["operation"] == "open":
                away = m["cap_radial_error"] > 0.045
                success = bool(m["cap_unthreaded"]) and not m["cap_engaged"] and away and enough_grasps
                success = success and released("cap") and steady("cap") and state.touching("cap", "table")
                progress = min(max(angle / (2 * np.pi * p["thread_turns"]), 0), 1) * 0.6 + 0.4 * float(away)
            else:
                success = bool(m["cap_engaged"]) and abs(angle) < 0.15 and abs(m["cap_height"]) < 0.001
                success = success and m["cap_radial_error"] < 0.001 and m["cap_tilt"] < 0.08 and enough_grasps
                success = success and released("cap") and steady("cap") and state.touching("cap", "bottle")
                progress = np.exp(-abs(angle) / 6) * float(m["cap_engaged"])

        elif task.name in ("knob", "switch", "latch", "key_lock"):
            q = float(state.joint_positions[p["joint"]])
            qd = float(state.joint_velocities[p["joint"]])
            error = abs(q - p["target"])
            success = error < p["tolerance"] and abs(qd) < (0.01 if task.name in ("latch", "bottle_cap") else 0.08)
            label = {"knob": "knob", "switch": "switch", "latch": "bolt", "bottle_cap": "cap", "key_lock": "key"}[
                task.name
            ]
            success = success and released(label)
            progress = float(np.exp(-error / (0.3 if task.name in ("knob", "switch", "key_lock") else 0.01)))
            metrics.update(joint_error=error, joint_position=q)
            if task.name == "latch":
                bolt = state.actors["bolt"]
                keeper = state.actors["keeper"]
                rotation = rotation_matrix(bolt.orientation)
                tip = np.asarray(bolt.position) + rotation[:, 0] * 0.0225
                transverse = np.abs(np.asarray(bolt.position)[1:] - np.asarray(keeper.position)[1:])
                # An unloaded bolt can sit inside a clearance-fit keeper without
                # normal force. Actual penetration beyond the keeper's far face,
                # within its aperture, establishes engagement.
                if p["target_state"] == "engaged":
                    success = success and tip[0] > p["keeper_x"] + 0.003 and np.all(transverse < [0.0015, 0.001])
                else:
                    success = success and tip[0] < p["keeper_x"] - 0.004
                success = success and rotation[0, 0] > 0.995 and steady("bolt")
            elif task.name == "key_lock":
                # The key must physically rotate with the cylinder, and the cam must retract the bolt.
                key = state.actors["key"]
                key_angle = np.arctan2(rotation_matrix(key.orientation)[1, 0], rotation_matrix(key.orientation)[0, 0])
                inserted = np.linalg.norm(np.asarray(key.position) - p["seat"]) < 0.0015
                success = success and inserted and abs(key_angle - q) < 0.1 and state.touching("key", "cylinder")
                success = success and state.joint_positions["bolt_travel"] < -0.012

        elif task.name in ("rope_thread", "cable_untangle"):
            nodes = state.deformable_positions["rope"]
            velocity = state.deformable_velocities["rope"]
            speed = float(np.max(np.linalg.norm(velocity, axis=1)))
            # Stationarity is observed at the controller's frame rate. Retain
            # raw instantaneous speed for diagnostics, but reject actual nodal
            # displacement above the same 20 mm/s threshold. A fresh episode
            # has no measured interval and cannot claim stationary history.
            ready = self.previous_rope_positions is not None
            stationary = False
            if ready:
                control_speed = float(np.max(np.linalg.norm(nodes - self.previous_rope_positions, axis=1)) / dt)
                metrics["max_control_frame_particle_speed"] = control_speed
                stationary = control_speed < 0.02 and np.isfinite(speed)
            self.previous_rope_positions = nodes.copy()
            metrics["stationarity_history_ready"] = float(ready)
            length = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())
            initial_length = float(np.linalg.norm(np.diff(task.deformables[0].vertices, axis=0), axis=1).sum())
            intact = 0.9 < length / initial_length < 1.1
            if task.name == "rope_thread" and task.variant == "thread":
                counts = [ring_crossings(nodes, r["center"], r["normal"], r["radius"]) for r in p["rings"]]
                metrics.update({f"ring_{i}/signed_passages": float(c) for i, c in enumerate(counts)})
                success = all(c == 1 for c in counts) and nodes[0, 0] < 0.20 and nodes[-1, 0] > 0.30
                progress = float(np.mean([c == 1 for c in counts]))
            else:
                endpoint_error = float(np.max(np.linalg.norm(nodes[[0, -1]] - p["endpoint_goals"], axis=1)))
                metrics["endpoint_error"] = endpoint_error
                if task.name == "cable_untangle":
                    straightness = float(np.linalg.norm(nodes[-1] - nodes[0]) / max(length, 1e-9))
                    metrics["straightness"] = straightness
                    # A near-straight, unstretched chain cannot retain a loop or tight self knot.
                    success = straightness > p["straightness"] and endpoint_error < 0.02
                    progress = straightness
                else:
                    determinant = knot_determinant(close_rope(nodes, p["closure_apex"])) if endpoint_error < 0.02 else 0
                    metrics["knot_determinant"] = float(determinant)
                    success = determinant == p["knot_determinant"] and endpoint_error < 0.02
                    progress = 0.5 * float(determinant == p["knot_determinant"]) + 0.5 * np.exp(-endpoint_error / 0.05)
            metrics.update(max_particle_speed=speed, stretch_ratio=length / initial_length)
            success = success and intact and stationary and released("rope")

        elif task.name in ("cloth_fold", "paper_airplane"):
            nodes = state.deformable_positions["sheet"]
            velocity = state.deformable_velocities["sheet"]
            goal = (
                p["fold_stages"][min(self.stage, len(p["fold_stages"]) - 1)]
                if task.name == "paper_airplane"
                else p["target_vertices"]
            )
            error = float(np.sqrt(np.mean(np.sum((nodes - goal) ** 2, axis=1))))
            edges = task.deformables[0].edges
            current = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
            rest = np.linalg.norm(p["rest_vertices"][edges[:, 0]] - p["rest_vertices"][edges[:, 1]], axis=1)
            strain = float(np.max(np.abs(current / rest - 1)))
            stationary = np.max(np.linalg.norm(velocity, axis=1)) < 0.02
            folded = error < p["fold_tolerance"] and strain < 0.2 and stationary and released("sheet")
            if task.name == "paper_airplane":
                rest_angles = state.crease_rest_angles["sheet"]
                required = p["crease_masks"][self.stage]
                crease_fraction = float(np.mean(np.abs(rest_angles[required]) > 0.3)) if np.any(required) else 0.0
                metrics["crease_fraction"] = crease_fraction
                folded = (
                    error < p["fold_tolerance"]
                    and strain < 0.2
                    and stationary
                    and crease_fraction >= p["crease_fraction"]
                )
                self.stage_held = self.stage_held + dt if folded else 0.0
                final_stage = len(p["fold_stages"]) - 1
                if self.stage_held >= task.hold_seconds and self.stage < final_stage:
                    self.stage += 1
                    self.held = 0
                    self.stage_held = 0
                    folded = False
                success = folded and self.stage == final_stage and released("sheet")
                metrics["fold_stage"] = float(self.stage)
            else:
                success = folded
            progress = np.exp(-error / 0.04)
            metrics.update(fold_rms_error=error, max_edge_strain=strain)

        elif task.name == "gear_assembly":
            geometry = seated("gear", p["seat"], 0.001, 0.07)
            drive = state.joint_positions["drive_angle"]
            rot = rotation_matrix(state.actors["gear"].orientation)
            angle = np.arctan2(rot[1, 0], rot[0, 0])
            phase = abs(
                np.arctan2(np.sin(p["teeth"] * (angle + drive) - np.pi), np.cos(p["teeth"] * (angle + drive) - np.pi))
            )
            if self.previous_drive is not None:
                delta = drive - self.previous_drive
                gear_delta = np.arctan2(np.sin(angle - self.previous_gear), np.cos(angle - self.previous_gear))
                if geometry and state.touching("gear", "drive_gear"):
                    self.mesh_rotation += abs(delta)
                    self.gear_rotation += abs(gear_delta)
                    self.mesh_error += abs(delta + gear_delta)
                elif not geometry:
                    self.mesh_rotation = self.gear_rotation = self.mesh_error = 0
                elif self.mesh_rotation > 0:
                    self.mesh_error += abs(delta + gear_delta)
            self.previous_drive, self.previous_gear = drive, angle
            transmitted = self.mesh_rotation > 0.20 and self.gear_rotation > 0.18 and self.mesh_error < 0.05
            success = geometry and phase < p["phase_tolerance"] and transmitted and steady("gear") and released("gear")
            metrics.update(tooth_phase_error=phase, transmitted_rotation=self.gear_rotation, mesh_error=self.mesh_error)
            progress = 0.5 * float(geometry) + 0.5 * min(self.gear_rotation / 0.2, 1)

        elif task.name == "circuit_restore":
            enabled = {}
            for i, target in enumerate(p["terminals"]):
                label = f"connector_{i}"
                enabled[label] = (
                    seated(label, target, 0.001, 0.08, True)
                    and state.touching(label, f"terminal_{i}")
                    and released(label)
                    and steady(label)
                )
            enabled["switch"] = (
                state.joint_positions["switch_angle"] > 0.30
                and abs(state.joint_velocities["switch_angle"]) < 0.08
                and released("switch")
            )
            reachable = {"source"}
            for _ in p["graph"]:
                for a, b, component in p["graph"]:
                    if enabled[component] and (a in reachable or b in reachable):
                        reachable.update((a, b))
            success = "load" in reachable
            progress = float(np.mean(list(enabled.values())))
            metrics.update({f"continuity/{name}": float(value) for name, value in enabled.items()})

        elif task.name == "catch_ball":
            ball = state.actors["ball"]
            grasp = state.touching("ball", "left_finger") and state.touching("ball", "right_finger")
            error = np.linalg.norm(np.asarray(ball.position) - p["capture_center"])
            success = grasp and steady("ball") and error < p["capture_radius"]
            progress = np.exp(-error / 0.05)
            metrics.update(capture_error=float(error), two_finger_contact=float(grasp))

        elif task.name == "roll_ball":
            ball = state.actors["ball"]
            distance = np.abs(np.asarray(ball.position) - p["center"])
            # Center clearance includes radius so rim balancing cannot count as pocket entry.
            inside = bool(np.all(distance[:2] < p["pocket_half_extent"] - p["ball_radius"]))
            success = (
                inside
                and distance[2] < 0.003
                and state.touching("ball", "pocket")
                and released("ball")
                and steady("ball")
            )
            progress = np.exp(-np.linalg.norm(distance) / 0.08)
            metrics["pocket_error"] = float(np.linalg.norm(distance))
            two_jaw_grasp = state.touching("ball", "left_finger") and state.touching("ball", "right_finger")
            supported = state.touching("ball", "table") or state.touching("ball", "pocket")
            carried = (
                not supported and not released("ball") and ball.position[2] > p["ball_radius"] + p["support_clearance"]
            )
            # Permit the 10mm entrance ramp with 4mm contact clearance, while a
            # sphere clearing the 25mm rim violates the roll-only operation.
            lifted = ball.position[2] > p["max_roll_center_height"]
            self.roll_only_violated = self.roll_only_violated or two_jaw_grasp or carried or lifted
            escaped = escaped or self.roll_only_violated
            metrics["roll_only_violated"] = float(self.roll_only_violated)

        self.held = self.held + dt if success and not escaped else 0.0
        done = self.held >= task.hold_seconds
        failure = bool(escaped)
        components = {
            "progress": float(progress),
            "success": float(done),
            "failure": -float(failure),
            "action": -float(action_cost),
        }
        reward = (
            components["success"] + 0.1 * components["progress"] + components["failure"] + 0.001 * components["action"]
        )
        metrics.update(hold_seconds=self.held, elapsed_seconds=self.elapsed)
        return TaskResult(
            reward,
            done,
            failure,
            done or failure,
            self.elapsed >= task.horizon_seconds and not (done or failure),
            components,
            metrics,
        )
