# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured VBD dynamics, including its otherwise unsupported motor properties."""

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_hiwonder.tasks.vbd_motors import VBDMotorAdapter


def rotor(armature=0.0, upper=2.0, inertia=0.01):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    # A preceding FREE joint deliberately separates coordinate and DOF indices.
    builder.add_body(xform=wp.transform((2, 0, 0), wp.quat_identity()), mass=1, inertia=np.eye(3) * 0.01)
    body = builder.add_link(mass=1, inertia=np.eye(3) * inertia)
    joint = builder.add_joint_revolute(
        -1, body, axis=(0, 0, 1), armature=armature, limit_lower=-2, limit_upper=upper, limit_ke=1000, limit_kd=0
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverVBD(model, iterations=20, rigid_compliant_alm=True)
    return model, solver, joint


def advance(model, solver, motor, current=None, steps=192, dt=1 / 1920):
    current = current or model.state()
    following, control = model.state(), model.control()
    for _ in range(steps):
        current.clear_forces()
        motor.prepare(current, control, dt)
        solver.step(current, following, control, None, dt)
        newton.eval_ik(model, following, following.joint_q, following.joint_qd)
        current, following = following, current
    return current


@pytest.mark.parametrize("armature", [0.0, 0.01, 0.1])
def test_rotor_acceleration_includes_randomized_armature(armature):
    model, solver, joint = rotor(armature)
    motor = VBDMotorAdapter(model, [joint])
    motor.set_parameters([[0, 0, 10, 1e9, 0, armature]])
    motor.set_targets([0], feedforward=[0.1])
    state = advance(model, solver, motor)
    speed = state.joint_qd.numpy()[model.joint_qd_start.numpy()[joint]]
    assert speed == pytest.approx(0.1 * 0.1 / (0.01 + armature), rel=0.003)
    np.testing.assert_allclose(state.joint_q.numpy()[:7], model.joint_q.numpy()[:7], atol=1e-7)


def test_motor_effort_and_speed_envelope_bound_actual_acceleration():
    model, solver, joint = rotor()
    motor = VBDMotorAdapter(model, [joint])
    motor.set_parameters([[100, 0, 0.02, 0.1, 0, 0]])
    motor.set_targets([1])
    state = advance(model, solver, motor, steps=3840)
    speed = float(state.joint_qd.numpy()[model.joint_qd_start.numpy()[joint]])
    assert 0.095 < speed < 0.1001
    assert abs(float(motor.applied_effort.numpy()[0])) <= 0.020001


def test_motor_coulomb_friction_changes_measured_free_rotor_decay():
    final = []
    for friction in (0, 0.02):
        model, solver, joint = rotor()
        current = model.state()
        velocity = current.joint_qd.numpy()
        velocity[model.joint_qd_start.numpy()[joint]] = 0.2
        current.joint_qd.assign(velocity)
        newton.eval_fk(model, current.joint_q, current.joint_qd, current)
        motor = VBDMotorAdapter(model, [joint])
        motor.set_parameters([[0, 0, 1, 10, friction, 0]])
        state = advance(model, solver, motor, current, steps=96)
        final.append(float(state.joint_qd.numpy()[model.joint_qd_start.numpy()[joint]]))
    assert final[0] == pytest.approx(0.2, abs=1e-4)
    assert final[1] == pytest.approx(0.1, abs=0.002)


def test_inertial_drive_preserves_acceleration_across_a_joint_limit():
    model, solver, joint = rotor(0.01, upper=0.1)
    q_index = model.joint_q_start.numpy()[joint]
    v_index = model.joint_qd_start.numpy()[joint]
    current = model.state()
    coordinates, velocity = current.joint_q.numpy(), current.joint_qd.numpy()
    coordinates[q_index], velocity[v_index] = 0.0999, 1.0
    current.joint_q.assign(coordinates)
    current.joint_qd.assign(velocity)
    newton.eval_fk(model, current.joint_q, current.joint_qd, current)
    motor = VBDMotorAdapter(model, [joint])
    motor.set_parameters([[0, 0, 1, 1e9, 0, 0.01]])
    dt = 1 / 1920
    state = advance(model, solver, motor, current, steps=1, dt=dt)
    expected = ((0.01 + 0.01) * (0.0999 + dt) + 1000 * dt**2 * 0.1) / (0.02 + 1000 * dt**2)
    assert state.joint_q.numpy()[q_index] == pytest.approx(expected, abs=3e-7)


def test_tiny_jaw_hold_does_not_enter_saturated_damping_limit_cycle():
    model, solver, joint = rotor(1e-4, inertia=1e-5)
    current = model.state()
    index = model.joint_q_start.numpy()[joint]
    positions = current.joint_q.numpy()
    positions[index] = 0.01
    current.joint_q.assign(positions)
    newton.eval_fk(model, current.joint_q, current.joint_qd, current)
    motor = VBDMotorAdapter(model, [joint])
    motor.set_parameters([[25, 2.5, 1, 2.5, 0, 1e-4]])
    motor.set_targets([0])
    state = advance(model, solver, motor, current, steps=960)
    assert abs(float(state.joint_q.numpy()[index])) < 0.0002
    assert abs(float(state.joint_qd.numpy()[model.joint_qd_start.numpy()[joint]])) < 0.002


def test_tiny_joint_passive_friction_stops_without_reversing_motion():
    model, solver, joint = rotor(1e-4, inertia=1e-5)
    current = model.state()
    index = model.joint_qd_start.numpy()[joint]
    velocity = current.joint_qd.numpy()
    velocity[index] = 0.001
    current.joint_qd.assign(velocity)
    newton.eval_fk(model, current.joint_q, current.joint_qd, current)
    motor = VBDMotorAdapter(model, [joint])
    motor.set_parameters([[0, 0, 1, 2.5, 0.1, 1e-4]])
    state = advance(model, solver, motor, current, steps=8)
    speed = float(state.joint_qd.numpy()[index])
    assert -1e-6 < speed < 1e-5
