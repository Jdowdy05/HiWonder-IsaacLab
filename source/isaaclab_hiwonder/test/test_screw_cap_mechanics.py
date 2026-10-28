# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Free-body cap thread, history and outcome contracts; not robot policy proof."""

import numpy as np
from isaaclab_hiwonder.tasks import ActorState, SceneState, TaskEpisode, build_task
from isaaclab_hiwonder.tasks.backend import TaskMechanics, native_deformable_config


def _position(state, angle, height):
    state.actors["cap"] = ActorState((0.25, 0.03, 0.048 + height), (0, 0, np.sin(angle / 2), np.cos(angle / 2)))


def _regrasp(mechanics, state):
    state.contacts = {("cap", "left_finger"), ("cap", "right_finger")}
    mechanics.forces(state, 0.12)
    state.contacts = set()
    mechanics.forces(state, 0.06)
    state.contacts = {("cap", "left_finger"), ("cap", "right_finger")}
    mechanics.forces(state, 0.12)


def test_thread_force_has_reciprocal_axial_torque_and_total_wrench():
    task = build_task("bottle_cap", "open")
    assert not task.joints
    state = SceneState(actors=dict(task.initial))
    _position(state, 0, 0.001)
    forces = TaskMechanics(task).forces(state)
    assert forces.body_forces["cap"][2] < -1
    assert forces.body_torques["cap"][2] > 0
    np.testing.assert_allclose(forces.body_forces["cap"] + forces.body_forces["bottle"], 0, atol=1e-12)
    assert abs(forces.body_forces["cap"][2] * 0.003 / (2 * np.pi) + forces.body_torques["cap"][2]) < 1e-12


def test_unscrewed_cap_is_free_in_all_six_dofs_and_requires_real_regrasp_intervals():
    task = build_task("bottle_cap", "open")
    state = SceneState(actors=dict(task.initial))
    mechanics = TaskMechanics(task)
    _regrasp(mechanics, state)
    for angle in np.linspace(0, 7 * np.pi, 100):
        _position(state, angle, 0.003 * angle / (2 * np.pi))
        mechanics.forces(state, 0.01)
    state.actors["cap"] = ActorState((0.16, -0.06, 0.011), (0.2, 0.3, 0.4, 0.8426149773))
    state.contacts = {("cap", "table")}
    forces = mechanics.forces(state, 0.1)
    np.testing.assert_array_equal(forces.body_forces["cap"], np.zeros(3))
    np.testing.assert_array_equal(forces.body_torques["cap"], np.zeros(3))
    assert state.mechanisms["cap_grasp_intervals"] == 2
    assert TaskEpisode(task).evaluate(state, 0.6).success
    # A history lacking sufficient actual grasp intervals must not complete.
    mechanics.cap.grasp_intervals = 1
    mechanics.cap.observe(state)
    assert not TaskEpisode(task).evaluate(state, 0.6).success


def test_free_cap_can_engage_close_and_resume_unwrapped_rotation():
    task = build_task("bottle_cap", "close")
    state = SceneState(actors=dict(task.initial))
    mechanics = TaskMechanics(task)
    assert not mechanics.cap.engaged
    _position(state, 0, 0.0095)
    mechanics.forces(state, 0.01)
    assert mechanics.cap.engaged
    _regrasp(mechanics, state)
    for angle in np.linspace(6 * np.pi, 0, 100):
        _position(state, angle, 0.003 * angle / (2 * np.pi))
        mechanics.forces(state, 0.01)
        if angle < np.pi:
            snapshot = mechanics.cap.state_dict()
            restored = TaskMechanics(task)
            restored.cap.load_state_dict(snapshot)
            np.testing.assert_allclose(
                restored.forces(state).body_forces["cap"], mechanics.forces(state).body_forces["cap"]
            )
    state.contacts = {("cap", "bottle")}
    mechanics.forces(state, 0.1)
    assert TaskEpisode(task).evaluate(state, 0.6).success


def test_native_surface_spawner_discovers_descendant_garment_mesh(tmp_path):
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    task = build_task("cloth_fold", "shirt")
    cfg = native_deformable_config(task.deformables[0], "/World/Sheet", tmp_path)
    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        cfg.spawn.func("/World/Sheet", cfg.spawn)
    meshes = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Sheet")) if prim.IsA(UsdGeom.Mesh)]
    assert meshes
    assert any(len(UsdGeom.Mesh(prim).GetPointsAttr().Get()) == len(task.deformables[0].vertices) for prim in meshes)


def test_actual_vbd_free_cap_rotation_produces_axial_thread_motion():
    import newton
    import warp as wp

    task = build_task("bottle_cap", "open")
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    bottle = builder.add_body(xform=wp.transform(task.initial["bottle"].position, wp.quat_identity()), mass=0)
    cap = builder.add_body(xform=wp.transform(task.initial["cap"].position, wp.quat_identity()), mass=0.03)
    builder.add_shape_cylinder(cap, radius=0.023, half_height=0.01)
    builder.color()
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_compliant_alm=True)
    current, following = model.state(), model.state()
    control = model.control()
    mechanics = TaskMechanics(task)
    for _ in range(500):
        pose = current.body_q.numpy().copy()
        velocity = current.body_qd.numpy().copy()
        state = SceneState(
            actors={
                name: ActorState(pose[index, :3], pose[index, 3:], velocity[index, :3], velocity[index, 3:])
                for name, index in (("bottle", bottle), ("cap", cap))
            }
        )
        forces = mechanics.forces(state, 0.0001)
        wrench = np.zeros((2, 6), np.float32)
        wrench[cap, :3] = forces.body_forces["cap"]
        wrench[cap, 3:] = forces.body_torques["cap"] + [0, 0, 0.02]
        current.clear_forces()
        wp.copy(current.body_f, wp.array(wrench, dtype=wp.spatial_vector, device="cpu"))
        solver.step(current, following, control, None, 0.0001)
        current, following = following, current
    final = current.body_q.numpy()
    assert np.all(np.isfinite(final))
    assert mechanics.cap.angle > 0.1
    assert final[cap, 2] - 0.048 > 0.00003
