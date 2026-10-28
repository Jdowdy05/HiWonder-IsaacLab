# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actual Newton/Warp arrays exercise force application; no simulated robot proof."""

import newton
import numpy as np
import warp as wp
from isaaclab_hiwonder.tasks.api import DeformableSpec
from isaaclab_hiwonder.tasks.mechanics import dihedral
from isaaclab_hiwonder.tasks.native_forces import NewtonTaskForceHook


def test_native_paper_hook_preserves_plastic_state_and_resets_selected_environment():
    builder = newton.ModelBuilder()
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0.008]], dtype=np.float32)
    for offset in (0, 0.1):
        builder.add_cloth_mesh(
            pos=wp.vec3(offset, 0, 0),
            rot=wp.quat_identity(),
            scale=1,
            vel=wp.vec3(),
            vertices=points.tolist(),
            indices=[0, 1, 2, 1, 0, 3],
            density=0.1,
            edge_ke=0,
        )
    model = builder.finalize(device="cpu")
    state = model.state()
    spec = DeformableSpec(
        "sheet",
        "plastic_shell",
        points,
        np.empty((0, 2), int),
        np.array([[0, 1, 2], [1, 0, 3]]),
        0.0001,
        800,
        100,
        0,
        0,
        parameters={"hinges": [[0, 1, 2, 3]], "hinge_stiffness": 0.015, "yield_angle": 0.15},
    )
    hook = NewtonTaskForceHook(model, surfaces=[(0, spec, 0), (1, spec, 4)])
    hook(state)
    before = hook.rest.numpy().copy()
    assert np.all(np.abs(before) > 0.4)
    assert np.linalg.norm(state.particle_f.numpy()) > 0.01
    hook.reset([True, False])
    np.testing.assert_allclose(hook.rest.numpy(), [0, before[1]], atol=1e-7)


def test_vbd_integrates_plastic_crease_force_without_elastic_flattening():
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0.008]], np.float32)
    builder.add_cloth_mesh(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        scale=1,
        vel=wp.vec3(),
        vertices=points.tolist(),
        indices=[0, 1, 2, 1, 0, 3],
        density=0.1,
        tri_ke=100,
        tri_ka=100,
        tri_kd=0.01,
        edge_ke=0,
        particle_radius=0.00006,
    )
    builder.particle_mass[:3] = [0.0, 0.0, 0.0]
    builder.color()
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverVBD(model, iterations=5, particle_enable_self_contact=False)
    current, following = model.state(), model.state()
    spec = DeformableSpec(
        "sheet",
        "plastic_shell",
        points,
        np.empty((0, 2), int),
        np.array([[0, 1, 2], [1, 0, 3]]),
        0.00006,
        800,
        100,
        0,
        0,
        parameters={"hinges": [[0, 1, 2, 3]], "hinge_stiffness": 2e-5, "yield_angle": 0.15},
    )
    hook = NewtonTaskForceHook(model, surfaces=[(0, spec, 0)])
    control = model.control()
    for _ in range(100):
        current.clear_forces()
        hook(current)
        solver.step(current, following, control, None, 0.0001)
        current, following = following, current
    actual = current.particle_q.numpy()
    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(actual[3] - points[3]) > 1e-6
    assert abs(dihedral(actual)) > 0.3
    assert abs(hook.rest.numpy()[0]) > 0.4


def test_native_barrel_bond_holds_translation_and_twist_in_two_worlds(tmp_path):
    from isaaclab_hiwonder.tasks import build_task
    from isaaclab_hiwonder.tasks.api import rotation_matrix
    from isaaclab_hiwonder.tasks.backend import author_rigid_usd, native_deformable_config
    from isaaclab_hiwonder.tasks.native_backend import author_native_frame_bonds, physics_config
    from isaaclab_hiwonder.tasks.native_inertia import capture_cable_inertias, restore_cable_inertias

    from pxr import Usd

    import isaaclab.sim as sim_utils

    definition = build_task("barrel_jack")
    spec = definition.deformables[0]
    spec.vertices = spec.vertices[:3].copy()
    spec.edges = np.array([[0, 1], [1, 2]])
    path = tmp_path / "bond.usda"
    author_rigid_usd(definition, path)
    stage = Usd.Stage.Open(str(path))
    cable_path = "/Task/Deformables/cable"
    cfg = native_deformable_config(spec, cable_path, tmp_path)
    with sim_utils.use_stage(stage):
        cfg.spawn.func(cable_path, cfg.spawn)
    author_native_frame_bonds(stage, [(0, definition, "/Task", {"cable": cable_path})])
    source = newton.ModelBuilder(gravity=(0, 0, 0))
    source.add_usd(stage)
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for world in range(2):
        builder.add_world(source, xform=wp.transform((world, 0, 0), wp.quat_identity()), label_prefix=f"world{world}")
    builder.color()
    roots = {label.split("/geometry/")[0]: 2 for label in builder.body_label if label.endswith("_edge_body_0")}
    assert len(roots) == 2
    authored_inertias = capture_cable_inertias(builder, roots)
    model = builder.finalize(device="cpu")
    regularized = model.body_inertia.numpy().copy()
    regularized_inverse = model.body_inv_inertia.numpy().copy()
    masses = model.body_mass.numpy().copy()
    receipt = restore_cable_inertias(model, authored_inertias)
    assert receipt["body_count"] == 4
    assert (
        max(
            record["regularized_eigenvalues_kg_m2"][0] / record["restored_eigenvalues_kg_m2"][0]
            for record in receipt["bodies"]
        )
        > 1000
    )
    np.testing.assert_array_equal(model.body_mass.numpy(), masses)
    other = [i for i, label in enumerate(model.body_label) if label not in authored_inertias]
    np.testing.assert_array_equal(model.body_inertia.numpy()[other], regularized[other])
    np.testing.assert_array_equal(model.body_inv_inertia.numpy()[other], regularized_inverse[other])
    bonds = [i for i, label in enumerate(model.joint_label) if "CableBond_" in label]
    assert len(bonds) == 6
    np.testing.assert_array_equal(model.joint_world.numpy()[bonds], [0, 0, 0, 1, 1, 1])
    parents = [i for i, label in enumerate(model.body_label) if label.endswith("/Task/plug")]
    children = [i for i, label in enumerate(model.body_label) if label.endswith("_edge_body_0")]
    assert len(parents) == len(children) == 2
    initial = model.body_q.numpy().copy()
    current, following = model.state(), model.state()
    solver = newton.solvers.SolverVBD(model, iterations=20, rigid_compliant_alm=True)
    reference, reference_next = model.state(), model.state()
    reference_solver = newton.solvers.SolverVBD(model, iterations=20, rigid_compliant_alm=True)
    control = model.control()
    dt = 1 / 240 / physics_config(definition).num_substeps
    external = np.zeros((model.body_count, 6), np.float32)
    external[children[0]] = [0, 0.1, 0, 0.0001, 0.0001, 0.0001]
    load = wp.array(external, dtype=wp.spatial_vector, device="cpu")
    for _ in range(300):
        current.clear_forces()
        wp.copy(current.body_f, load)
        solver.step(current, following, control, None, dt)
        current, following = following, current
        reference.clear_forces()
        reference_solver.step(reference, reference_next, control, None, dt)
        reference, reference_next = reference_next, reference
    final = current.body_q.numpy()
    frame = np.asarray(spec.attachments[0]["parent_frame"])
    length = np.linalg.norm(spec.vertices[1] - spec.vertices[0])
    for parent, child in zip(parents, children):
        a, b = final[parent], final[child]
        anchor = a[:3] + rotation_matrix(a[3:]) @ frame[:3]
        endpoint = b[:3] + rotation_matrix(b[3:]) @ [0, 0, -length / 2]
        relative = (rotation_matrix(a[3:]) @ rotation_matrix(frame[3:])).T @ rotation_matrix(b[3:])
        assert np.linalg.norm(anchor - endpoint) < 0.001
        assert np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)) < 0.05
    assert np.linalg.norm(final[parents[0], :3] - initial[parents[0], :3]) > 0.0001
    np.testing.assert_array_equal(final[[parents[1], children[1]]], reference.body_q.numpy()[[parents[1], children[1]]])
    reset_poses, reset_velocities = final.copy(), current.body_qd.numpy().copy()
    selected_bodies = model.body_world.numpy() == 0
    reset_poses[selected_bodies] = initial[selected_bodies]
    reset_velocities[selected_bodies] = 0
    wp.copy(current.body_q, wp.array(reset_poses, dtype=wp.transform, device="cpu"))
    wp.copy(current.body_qd, wp.array(reset_velocities, dtype=wp.spatial_vector, device="cpu"))
    solver.reset(current, world_mask=wp.array([True, False, False], dtype=wp.bool, device="cpu"))
    for _ in range(30):
        current.clear_forces()
        reference.clear_forces()
        solver.step(current, following, control, None, dt)
        reference_solver.step(reference, reference_next, control, None, dt)
        current, following = following, current
        reference, reference_next = reference_next, reference
    np.testing.assert_array_equal(
        current.body_q.numpy()[[parents[1], children[1]]], reference.body_q.numpy()[[parents[1], children[1]]]
    )
    for index, label in enumerate(model.body_label):
        if label in authored_inertias:
            np.testing.assert_array_equal(
                model.body_inertia.numpy()[index], authored_inertias[label]["inertia"].astype(np.float32)
            )


def test_native_inertia_restoration_rejects_nonphysical_authored_tensor():
    import pytest
    from isaaclab_hiwonder.tasks.native_inertia import capture_cable_inertias

    builder = newton.ModelBuilder()
    builder.add_body(label="/Cable/mesh_edge_body_0", mass=0.001, inertia=np.diag([1e-10, 1e-10, 3e-10]))
    with pytest.raises(ValueError, match="Invalid authored physical inertia"):
        capture_cable_inertias(builder, {"/Cable": 1})


def test_vbd_contact_adapter_reports_measured_support_force_after_reset():
    from isaaclab_hiwonder.tasks.vbd_contact import ContactReportingVBD

    builder = newton.ModelBuilder()
    builder.begin_world()
    body = builder.add_body(xform=wp.transform((0, 0, 0.01), wp.quat_identity()), mass=0.1)
    builder.add_shape_sphere(body, radius=0.01)
    builder.end_world()
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device="cpu")
    solver = ContactReportingVBD(model, iterations=20, rigid_compliant_alm=True)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    current, following = model.state(), model.state()
    control = model.control()
    samples = []
    for index in range(300):
        if index == 100:
            solver.reset(current, world_mask=wp.array([True, False], dtype=wp.bool, device="cpu"), flags=0)
        current.clear_forces()
        pipeline.collide(current, contacts)
        solver.step(current, following, control, contacts, 0.001)
        current, following = following, current
        solver.update_contacts(contacts, current)
        if index > 250:
            count = int(contacts.rigid_contact_count.numpy()[0])
            ids0 = contacts.rigid_contact_shape0.numpy()[:count]
            ids1 = contacts.rigid_contact_shape1.numpy()[:count]
            values = contacts.rigid_contact_force.numpy()[:count]
            shape_body = model.shape_body.numpy()
            samples.append(
                sum(
                    float(f[2]) * (1 if shape_body[b] == body else -1)
                    for a, b, f in zip(ids0, ids1, values)
                    if shape_body[a] == body or shape_body[b] == body
                )
            )
    weight = float(model.body_mass.numpy()[body]) * 9.81
    assert np.isfinite(samples).all()
    assert abs(np.median(samples) - weight) < 0.15 * weight


def test_vbd_publishes_measured_joint_coordinates_from_moving_bodies():
    from isaaclab_hiwonder.tasks.api import rotation_matrix
    from isaaclab_hiwonder.tasks.vbd_contact import ContactReportingVBD

    builder = newton.ModelBuilder()
    body = builder.add_link(xform=wp.transform((0.1, 0, 0.2), wp.quat_identity()), mass=0.1)
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0, 1, 0),
        parent_xform=wp.transform((0, 0, 0.2), wp.quat_identity()),
        child_xform=wp.transform((-0.1, 0, 0), wp.quat_identity()),
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device="cpu")
    solver = ContactReportingVBD(model, iterations=20, rigid_compliant_alm=True)
    current, following = model.state(), model.state()
    control = model.control()
    for _ in range(100):
        current.clear_forces()
        solver.step(current, following, control, None, 0.001)
        current, following = following, current
    rotation = rotation_matrix(current.body_q.numpy()[body, 3:])
    measured = np.arctan2(-rotation[2, 0], rotation[0, 0])
    assert abs(measured) > 0.1
    assert abs(float(current.joint_q.numpy()[0]) - measured) < 0.001
    assert abs(float(current.joint_qd.numpy()[0])) > 0.1


def test_vbd_fixed_root_reset_moves_rest_reference_without_repinning_live_body():
    from isaaclab_hiwonder.tasks.api import rotation_matrix
    from isaaclab_hiwonder.tasks.vbd_contact import ContactReportingVBD

    builder = newton.ModelBuilder()
    root = builder.add_link(xform=wp.transform((0, 0, 0.2), wp.quat_identity()), mass=0.1)
    builder.add_shape_box(root, hx=0.02, hy=0.02, hz=0.02)
    joint = builder.add_joint_fixed(-1, root, parent_xform=wp.transform((0, 0, 0.2), wp.quat_identity()))
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device="cpu")
    solver = ContactReportingVBD(model, iterations=20, rigid_compliant_alm=True)
    current, following = model.state(), model.state()
    target = wp.transform((0.1, 0.05, 0.2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.3))
    wp.copy(model.joint_X_p, wp.array([target], dtype=wp.transform, device="cpu"))
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    solver.reset(current, flags=0)
    solver.sync_fixed_root_rest()
    newton.eval_fk(model, current.joint_q, current.joint_qd, current)
    control = model.control()
    for _ in range(150):
        current.clear_forces()
        solver.step(current, following, control, None, 0.001)
        current, following = following, current
    actual = current.body_q.numpy()[root]
    rotation = rotation_matrix(actual[3:])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    assert abs(yaw - 0.3) < 0.001
    np.testing.assert_allclose(actual[:3], [0.1, 0.05, 0.2], atol=0.001)


def test_native_short_rope_removes_material_overlap_but_keeps_nonlocal_contact(tmp_path):
    from isaaclab_hiwonder.tasks import build_task
    from isaaclab_hiwonder.tasks.backend import native_deformable_config
    from isaaclab_hiwonder.tasks.native_backend import filter_cable_material_overlap
    from isaaclab_hiwonder.tasks.native_inertia import capture_cable_inertias, restore_cable_inertias

    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    spec = build_task("rope_thread", variant="thread").deformables[0]
    # Exercise the finer discretization that originally exposed end-cap overlap.
    spec.vertices = np.column_stack(
        [np.linspace(spec.vertices[0, axis], spec.vertices[-1, axis], 81) for axis in range(3)]
    )
    spec.edges = np.column_stack((np.arange(80), np.arange(1, 81)))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = "/Task/rope"
    cfg = native_deformable_config(spec, root, tmp_path)
    with sim_utils.use_stage(stage):
        cfg.spawn.func(root, cfg.spawn)
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 2e4
    builder.default_shape_cfg.kd = 100
    builder.default_shape_cfg.gap = 0.001
    builder.add_usd(stage)
    filter_cable_material_overlap(builder, spec, root)
    captured = capture_cable_inertias(builder, {root: len(spec.edges)})
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device="cpu")
    restore_cable_inertias(model, captured)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap")
    contacts = pipeline.contacts()
    current, following = model.state(), model.state()
    pipeline.collide(current, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    bodies = model.shape_body.numpy()
    first = bodies[contacts.rigid_contact_shape0.numpy()[:count]]
    second = bodies[contacts.rigid_contact_shape1.numpy()[:count]]
    assert not np.any((first >= 0) & (second >= 0)), "A straight cable must not collide with its own material"
    solver = newton.solvers.SolverVBD(model, iterations=20, rigid_compliant_alm=True)
    control = model.control()
    for _ in range(256):
        current.clear_forces()
        pipeline.collide(current, contacts)
        solver.step(current, following, control, contacts, 1 / 1920)
        current, following = following, current
        assert np.isfinite(current.body_q.numpy()).all()
    poses = current.body_q.numpy()
    assert np.max(abs(poses[:, 2] - spec.radius)) < 0.001
    # A distant material segment touching the first must remain a real contact.
    ids = sorted(
        (i for i, label in enumerate(model.body_label) if "_edge_body_" in label),
        key=lambda i: int(model.body_label[i].rsplit("_", 1)[-1]),
    )
    poses[ids[-1]] = poses[ids[0]]
    wp.copy(current.body_q, wp.array(poses, dtype=wp.transform, device="cpu"))
    pipeline.collide(current, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    pairs = zip(
        bodies[contacts.rigid_contact_shape0.numpy()[:count]],
        bodies[contacts.rigid_contact_shape1.numpy()[:count]],
    )
    assert any(set(pair) == {ids[0], ids[-1]} for pair in pairs)


def test_native_production_rope_can_satisfy_unchanged_stationary_dwell(tmp_path):
    from isaaclab_hiwonder.tasks import build_task
    from isaaclab_hiwonder.tasks.backend import native_deformable_config
    from isaaclab_hiwonder.tasks.native_backend import filter_cable_material_overlap, physics_config
    from isaaclab_hiwonder.tasks.native_inertia import capture_cable_inertias, restore_cable_inertias
    from isaaclab_hiwonder.tasks.runtime import cable_nodes_from_segments

    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    definition = build_task("rope_thread", variant="thread")
    spec = definition.deformables[0]
    spec.vertices = spec.vertices + [0, 0, 0.705]
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    path = "/Task/rope"
    cfg = native_deformable_config(spec, path, tmp_path)
    with sim_utils.use_stage(stage):
        cfg.spawn.func(path, cfg.spawn)
    physics = physics_config(definition)
    builder = newton.ModelBuilder()
    for name in ("ke", "kd", "mu", "gap"):
        setattr(builder.default_shape_cfg, name, getattr(physics.default_shape_cfg, name))
    builder.add_usd(stage)
    filter_cable_material_overlap(builder, spec, path)
    builder.add_shape_box(-1, xform=wp.transform((0.23, 0, 0.685), wp.quat_identity()), hx=0.4, hy=0.4, hz=0.02)
    captured = capture_cable_inertias(builder, {path: len(spec.edges)})
    builder.color()
    model = builder.finalize(device="cpu")
    restore_cable_inertias(model, captured)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=physics.solver_cfg.iterations,
        rigid_compliant_alm=physics.solver_cfg.rigid_compliant_alm,
        rigid_body_contact_buffer_size=physics.solver_cfg.rigid_body_contact_buffer_size,
    )
    current, following = model.state(), model.state()
    control = model.control()
    ids = sorted(
        (i for i, label in enumerate(model.body_label) if "_edge_body_" in label),
        key=lambda i: int(model.body_label[i].rsplit("_", 1)[-1]),
    )
    lengths = np.linalg.norm(np.diff(spec.vertices, axis=0), axis=1)
    frequency = 240 * physics.num_substeps
    speeds = []
    for step in range(2 * frequency):
        current.clear_forces()
        pipeline.collide(current, contacts)
        solver.step(current, following, control, contacts, 1 / frequency)
        current, following = following, current
        if step >= 1.5 * frequency:
            _, velocity = cable_nodes_from_segments(current.body_q.numpy()[ids], current.body_qd.numpy()[ids], lengths)
            speeds.append(float(np.linalg.norm(velocity, axis=1).max()))
    assert np.isfinite(speeds).all()
    assert max(speeds) < 0.02, "A released resting rope must qualify for the full unchanged stationary window"
