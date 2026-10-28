# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured runtime contracts with real Newton/Warp arrays and mechanism equations."""

import copy
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_hiwonder.tasks import build_task
from isaaclab_hiwonder.tasks.api import ActorState, SceneState
from isaaclab_hiwonder.tasks.backend import TaskMechanics
from isaaclab_hiwonder.tasks.runtime import TaskRuntime
from isaaclab_newton.physics import NewtonManager


class _Scene(dict):
    device = "cpu"

    def __init__(self, assets, origins):
        super().__init__(assets)
        self.env_origins = torch.tensor(origins, dtype=torch.float32)


def _metadata(count):
    return {
        "task_support_offsets_m": np.zeros((count, 3)),
        "task_body_assets": {},
        "task_articulations": {},
        "task_deformables": {},
        "task_contacts": {},
    }


def _native_model(order):
    builder = newton.ModelBuilder()
    positions = {"a": ((1, 2, 4), (11, 3, 6)), "b": ((4, 5, 6), (14, 7, 9))}
    for env_id, name in order:
        builder.add_body(
            xform=wp.transform(positions[name][env_id], wp.quat_identity()),
            com=wp.vec3(0, 0, 0.5),
            label=f"/World/envs/env_{env_id}/Room/Task/{name}",
        )
    model = builder.finalize(device="cpu")
    state = model.state()
    state.body_qd.assign(np.tile([1, 2, 3, 0, 2, 0], (len(order), 1)).astype(np.float32))
    return model, state


@wp.kernel
def _delayed_native_measurement(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    local_com: wp.array(dtype=wp.vec3),
    iterations: int,
    phase: float,
):
    index = wp.tid()
    value = index + 1
    for _ in range(iterations):
        value = value * 1664525 + 1013904223
    x = phase + 1.0 + wp.float32((value >> 8) & 127) / 128.0
    pose[index] = wp.transform(wp.vec3(x, 2.0, 3.0), wp.quat_identity())
    velocity[index] = wp.spatial_vector(1.0, 2.0, 3.0, 0.0, 2.0, 0.0)
    local_com[index] = wp.vec3(0.0, 0.0, 0.5 + phase * 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA producer and consumer stream")
def test_native_states_wait_for_warp_producer_on_nondefault_torch_stream(monkeypatch):
    """A pending Warp write must precede selected Torch reads without a device-wide fence."""
    device = "cuda:0"
    model = SimpleNamespace(
        body_label=[f"/World/envs/env_{i}/Room/Task/actor" for i in (1, 0)],
        body_com=wp.zeros(2, dtype=wp.vec3, device=device),
    )
    state = SimpleNamespace(
        body_q=wp.zeros(2, dtype=wp.transform, device=device),
        body_qd=wp.zeros(2, dtype=wp.spatial_vector, device=device),
    )
    monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(NewtonManager, "get_state_0", lambda: state)
    asset = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Room/Task/actor"),
        data=SimpleNamespace(__backend_name__="newton"),
    )
    scene = _Scene({"actor": asset}, [[0, 0, 0], [0, 0, 0]])
    scene.device = device
    metadata = _metadata(2)
    metadata["task_body_assets"] = {"actor": "actor"}
    runtime = TaskRuntime(scene, metadata, [build_task("cube_stack")] * 2)
    fields = [state.body_q, state.body_qd, model.body_com]
    # Compile and provision mappings before the asynchronous handoff under test.
    wp.launch(_delayed_native_measurement, dim=2, inputs=[*fields, 0, -1.0], device=device)
    wp.synchronize_device(device)
    runtime.states()
    producer = wp.Stream(device)
    consumer = torch.cuda.Stream(device=device)
    previous = None
    for phase in (0.0, 1.0):
        with wp.ScopedStream(producer, sync_enter=False, sync_exit=False), torch.cuda.stream(consumer):
            wp.launch(_delayed_native_measurement, dim=2, inputs=[*fields, 5_000_000, phase], device=device)
            measured = runtime.states()
        # Exact literals were established by the frozen synchronous Warp reader.
        np.testing.assert_array_equal(measured[0].actors["actor"].position, [1.703125 + phase, 2, 3])
        np.testing.assert_array_equal(measured[1].actors["actor"].position, [1.9765625 + phase, 2, 3])
        for item in measured:
            actor = item.actors["actor"]
            assert actor.position.shape == actor.linear_velocity.shape == actor.angular_velocity.shape == (3,)
            assert actor.orientation.shape == (4,)
            np.testing.assert_array_equal(actor.orientation, [0, 0, 0, 1])
            np.testing.assert_array_equal(actor.linear_velocity, [-phase, 2, 3])
            np.testing.assert_array_equal(actor.angular_velocity, [0, 2, 0])
        if previous is not None:
            np.testing.assert_array_equal(previous[0].actors["actor"].position, [1.703125, 2, 3])
            np.testing.assert_array_equal(previous[0].actors["actor"].linear_velocity, [0, 2, 3])
        previous = measured


def test_states_remap_rebuilt_model_and_keep_prior_measurements_owned(monkeypatch):
    """Rebuilt model body ordering must not reuse the prior model's actor indices."""
    active = dict(zip(("model", "state"), _native_model([(1, "b"), (0, "a"), (1, "a"), (0, "b")])))
    monkeypatch.setattr(NewtonManager, "get_model", lambda: active["model"])
    monkeypatch.setattr(NewtonManager, "get_state_0", lambda: active["state"])
    assets = {
        name: SimpleNamespace(
            cfg=SimpleNamespace(prim_path=f"{{ENV_REGEX_NS}}/Room/Task/{name}"),
            data=SimpleNamespace(__backend_name__="newton"),
        )
        for name in ("a", "b")
    }
    metadata = _metadata(2)
    metadata["task_body_assets"] = {name: name for name in assets}
    metadata["task_support_offsets_m"] = [[0, 0, 1], [0, 0, 2]]
    runtime = TaskRuntime(_Scene(assets, [[0, 0, 0], [10, 0, 0]]), metadata, [build_task("cube_stack")] * 2)
    before = runtime.states()
    np.testing.assert_array_equal(before[0].actors["a"].position, [1, 2, 3])
    np.testing.assert_array_equal(before[1].actors["b"].position, [4, 7, 7])
    np.testing.assert_array_equal(before[0].actors["a"].linear_velocity, [0, 2, 3])
    assert before[0].actors["a"].orientation.dtype == np.float32

    active.update(zip(("model", "state"), _native_model([(0, "b"), (1, "a"), (0, "a"), (1, "b")])))
    after = runtime.states()
    np.testing.assert_array_equal(after[0].actors["a"].position, [1, 2, 3])
    np.testing.assert_array_equal(after[1].actors["b"].position, [4, 7, 7])
    measured = active["state"].body_q.numpy()
    measured[:, 0] += 0.25
    active["state"].body_q.assign(measured)
    fresh = runtime.states()
    np.testing.assert_array_equal(fresh[0].actors["a"].position, [1.25, 2, 3])
    np.testing.assert_array_equal(after[0].actors["a"].position, [1, 2, 3])
    before[0].actors["a"].orientation[:] = [0, 0, 1, 0]
    np.testing.assert_array_equal(fresh[0].actors["a"].orientation, [0, 0, 0, 1])


def test_detent_effort_uses_live_joint_measurements_without_unrelated_body_reads():
    """The joint-only mechanism must not depend on unused actor/contact buffers."""

    class JointData:
        __backend_name__ = "physx"
        joint_pos = torch.tensor([[0.1], [-0.2]])
        joint_vel = torch.tensor([[0.2], [-0.1]])

        def __getattr__(self, name):
            raise RuntimeError(f"Unused body buffer requested: {name}")

    commands = []
    asset = SimpleNamespace(
        body_names=["housing", "switch"],
        joint_names=["switch_angle"],
        data=JointData(),
        actuators=SimpleNamespace(
            target_command=SimpleNamespace(set_effort_index=lambda **kwargs: commands.append(kwargs["value"].clone()))
        ),
        write_data_to_sim=lambda: None,
    )
    metadata = _metadata(2)
    metadata["task_articulations"] = {
        "housing": {"asset_key": "switch", "body_names": ["housing", "switch"], "joint_names": ["switch_angle"]}
    }
    runtime = TaskRuntime(
        _Scene({"switch": asset}, [[0, 0, 0], [1, 0, 0]]),
        metadata,
        [copy.deepcopy(build_task("switch")) for _ in range(2)],
    )
    detents = [definition.joints[0].parameters.pop("detent_states") for definition in runtime.definitions]
    runtime.apply_forces(0.0)
    np.testing.assert_array_equal(commands[-1].numpy(), [[0], [0]])
    for definition, values in zip(runtime.definitions, detents):
        definition.joints[0].parameters["detent_states"] = values
    runtime.apply_forces(0.01)
    np.testing.assert_allclose(commands[-1].numpy(), [[0.023054943], [-0.029047838]], rtol=1e-6)
    asset.data.joint_pos[:] = 0
    asset.data.joint_vel[:] = 0
    runtime.apply_forces(0.01)
    np.testing.assert_allclose(commands[-1].numpy(), 0, atol=1e-16)
    np.testing.assert_allclose(runtime.time, [0.02, 0.02], atol=0)


def _rigid_asset(states, name, wrenches):
    class Composer:
        def reset(self):
            pass

        def add_forces_and_torques(self, forces, torques, **kwargs):
            wrenches[name] = (forces.numpy().copy(), torques.numpy().copy(), kwargs["positions"].numpy().copy())

    values = [state.actors[name] for state in states]
    return SimpleNamespace(
        data=SimpleNamespace(
            __backend_name__="physx",
            root_link_pose_w=torch.tensor(
                np.array([np.r_[s.position, s.orientation] for s in values]), dtype=torch.float32
            ),
            root_com_vel_w=torch.tensor(
                np.array([np.r_[s.linear_velocity, s.angular_velocity] for s in values]), dtype=torch.float32
            ),
            body_com_pos_b=torch.zeros((len(states), 1, 3)),
        ),
        permanent_wrench_composer=Composer(),
    )


def test_cap_force_path_matches_reference_wrenches_and_contact_grasp_history():
    definitions = [build_task("bottle_cap") for _ in range(2)]
    measured = [SceneState(actors=copy.deepcopy(definition.initial)) for definition in definitions]
    for state in measured:
        state.actors["cap"].linear_velocity = (0.001, -0.002, 0.003)
        state.actors["cap"].angular_velocity = (0.01, 0.02, 0.03)
    measured[0].contacts = {("cap", "left_finger"), ("cap", "right_finger")}
    measured[1].contacts = {("cap", "left_finger")}
    wrenches = {}
    assets = {name: _rigid_asset(measured, name, wrenches) for name in definitions[0].initial}
    forces = torch.zeros((2, 1, 3, 3))
    forces[0, 0, :, 0] = torch.tensor([1.0, -1.0, 1.0])
    forces[1, 0, 0, 0] = 1.0
    assets["contact"] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces))
    metadata = _metadata(2)
    metadata["task_body_assets"] = {name: name for name in definitions[0].initial}
    metadata["task_contacts"] = {
        "cap": {"sensor_key": "contact", "filter_labels": ["left_finger", "left_finger", "right_finger"]}
    }
    runtime = TaskRuntime(_Scene(assets, [[0, 0, 0], [0, 0, 0]]), metadata, definitions)
    references = [TaskMechanics(copy.deepcopy(definition)) for definition in definitions]
    # Convert the independent SI fixture to the simulator buffer's float32 values.
    for state in measured:
        for actor in state.actors.values():
            actor.position = np.asarray(actor.position, np.float32).astype(float)
            actor.orientation = np.asarray(actor.orientation, np.float32)
            actor.linear_velocity = np.asarray(actor.linear_velocity, np.float32)
            actor.angular_velocity = np.asarray(actor.angular_velocity, np.float32)
    for _ in range(3):
        expected = []
        for mechanics, state in zip(references, measured):
            mechanics.cap.observe(state)
            expected.append(mechanics.forces(state, 0.05))
        runtime.apply_forces(0.05)
        for name in ("cap", "bottle"):
            np.testing.assert_array_equal(
                wrenches[name][0][:, 0], np.array([f.body_forces[name] for f in expected], np.float32)
            )
            np.testing.assert_array_equal(
                wrenches[name][1][:, 0], np.array([f.body_torques[name] for f in expected], np.float32)
            )
    assert [mechanics.cap.grasp_intervals for mechanics in runtime.mechanics] == [1, 0]
    assert [mechanics.cap.state_dict() for mechanics in runtime.mechanics] == [
        mechanics.cap.state_dict() for mechanics in references
    ]
    assert runtime.states()[0].contacts == measured[0].contacts


def test_lock_force_path_preserves_contact_gate_and_reciprocal_joint_efforts():
    definitions = [build_task("key_lock") for _ in range(2)]
    measured = [SceneState(actors=copy.deepcopy(definition.initial)) for definition in definitions]
    q = torch.tensor([[0.3, 0.004], [-0.1, -0.006]])
    qd = torch.tensor([[0.2, -0.03], [-0.3, 0.04]])
    for i, state in enumerate(measured):
        state.actors["key"].position = np.asarray(definitions[i].parameters["seat"], np.float32).astype(float)
        state.joint_positions = dict(zip(("lock_angle", "bolt_travel"), q[i].tolist()))
        state.joint_velocities = dict(zip(("lock_angle", "bolt_travel"), qd[i].tolist()))
    measured[0].contacts.add(("key", "cylinder"))
    commands = []
    assets = {name: _rigid_asset(measured, name, {}) for name in definitions[0].initial}
    assets["mechanism"] = SimpleNamespace(
        body_names=["housing", "cylinder", "bolt"],
        joint_names=["lock_angle", "bolt_travel"],
        data=SimpleNamespace(joint_pos=q, joint_vel=qd),
        actuators=SimpleNamespace(
            target_command=SimpleNamespace(set_effort_index=lambda **kwargs: commands.append(kwargs["value"].clone()))
        ),
        write_data_to_sim=lambda: None,
    )
    matrix = torch.zeros((2, 1, 2, 3))
    matrix[0, 0, :, 0] = torch.tensor([1.0, -1.0])
    assets["contact"] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=matrix))
    metadata = _metadata(2)
    metadata["task_body_assets"] = {name: name for name in ("key", "strike")}
    metadata["task_articulations"] = {
        "housing": {
            "asset_key": "mechanism",
            "body_names": ["housing", "cylinder", "bolt"],
            "joint_names": ["lock_angle", "bolt_travel"],
        }
    }
    metadata["task_contacts"] = {"key": {"sensor_key": "contact", "filter_labels": ["cylinder", "cylinder"]}}
    expected = [TaskMechanics(definition).forces(state, 0.01) for definition, state in zip(definitions, measured)]
    runtime = TaskRuntime(_Scene(assets, [[0, 0, 0], [0, 0, 0]]), metadata, definitions)
    runtime.apply_forces(0.01)
    np.testing.assert_array_equal(
        commands[-1].numpy(),
        np.array(
            [[force.joint_efforts[name] for name in ("lock_angle", "bolt_travel")] for force in expected], np.float32
        ),
    )


@pytest.mark.parametrize("asset_kind", ["physx_root", "native_root", "articulation"])
def test_live_com_writes_update_actor_origin_velocity_without_reset(asset_kind, monkeypatch):
    """A public backend COM write must affect the next measurement immediately."""
    definition = build_task("cube_stack")
    name = next(iter(definition.initial))
    measured = SceneState(actors=copy.deepcopy(definition.initial))
    measured.actors[name].linear_velocity = (1, 2, 3)
    measured.actors[name].angular_velocity = (0, 2, 0)
    asset = _rigid_asset([measured], name, {})
    asset.data.body_com_pos_b[0, 0, 2] = 0.5
    metadata = _metadata(1)
    metadata["task_body_assets"] = {name: name}
    if asset_kind == "native_root":
        model, state = _native_model([(0, "a")])
        monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
        monkeypatch.setattr(NewtonManager, "get_state_0", lambda: state)
        asset.cfg = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Room/Task/a")
        asset.data.__backend_name__ = "newton"

        def write_com():
            model.body_com.assign(np.array([[0, 0, 1]], np.float32))
    else:
        if asset_kind == "articulation":
            asset.body_names = [name]
            asset.joint_names = []
            asset.data.body_link_pose_w = asset.data.root_link_pose_w[:, None]
            asset.data.body_com_vel_w = asset.data.root_com_vel_w[:, None]
            metadata["task_body_assets"] = {}
            metadata["task_articulations"] = {name: {"asset_key": name, "body_names": [name], "joint_names": []}}

        def write_com():
            # An external NumPy view write also leaves Torch's version unchanged.
            asset.data.body_com_pos_b.numpy()[0, 0, 2] = 1

    runtime = TaskRuntime(_Scene({name: asset}, [[0, 0, 0]]), metadata, [definition])
    before = runtime.states()
    np.testing.assert_array_equal(before[0].actors[name].linear_velocity, [0, 2, 3])
    write_com()
    np.testing.assert_array_equal(runtime.states()[0].actors[name].linear_velocity, [-1, 2, 3])
    np.testing.assert_array_equal(before[0].actors[name].linear_velocity, [0, 2, 3])


@pytest.mark.parametrize("asset_kind", ["physx_root", "native_root", "articulation"])
def test_mixed_com_precision_preserves_actor_dtype_and_rounding(asset_kind, monkeypatch):
    pose = torch.tensor([[0.123, 0.234, 0.345, 0, 0, 0, 1]], dtype=torch.float32)
    velocity = torch.tensor([[0.1234567, 0.2345678, 0.3456789, 0.1234567, 0.2345678, 0.3456789]], dtype=torch.float32)
    com = torch.tensor([[[0.123456789123, 0.234567891234, 0.345678912345]]], dtype=torch.float64)
    asset = SimpleNamespace(
        data=SimpleNamespace(
            __backend_name__="physx", root_link_pose_w=pose, root_com_vel_w=velocity, body_com_pos_b=com
        )
    )
    metadata = _metadata(1)
    metadata["task_body_assets"] = {"body": "asset"}
    if asset_kind == "native_root":
        model = SimpleNamespace(body_label=["/World/envs/env_0/Task/body"], body_com=com[:, 0])
        state = SimpleNamespace(body_q=pose, body_qd=velocity)
        monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
        monkeypatch.setattr(NewtonManager, "get_state_0", lambda: state)
        asset.cfg = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Task/body")
        asset.data.__backend_name__ = "newton"
    elif asset_kind == "articulation":
        asset.body_names, asset.joint_names = ["body"], []
        asset.data.body_link_pose_w, asset.data.body_com_vel_w = pose[:, None], velocity[:, None]
        metadata["task_body_assets"] = {}
        metadata["task_articulations"] = {"body": {"asset_key": "asset", "body_names": ["body"], "joint_names": []}}
    runtime = TaskRuntime(_Scene({"asset": asset}, [[0, 0, 0]]), metadata, [build_task("cube_stack")])
    actor = runtime.states()[0].actors["body"]
    assert actor.orientation.dtype == actor.linear_velocity.dtype == actor.angular_velocity.dtype == np.float32
    np.testing.assert_array_equal(
        actor.linear_velocity,
        np.array([0.12345672398805618, 0.2345677763223648, 0.3456788957118988], np.float32),
    )
    np.testing.assert_array_equal(actor.orientation, [0, 0, 0, 1])


def test_joint_measurements_keep_large_integer_values_when_velocity_is_float():
    asset = SimpleNamespace(
        body_names=[],
        joint_names=["measured_joint"],
        data=SimpleNamespace(joint_pos=torch.tensor([[16777217]], dtype=torch.int64), joint_vel=torch.tensor([[0.25]])),
    )
    metadata = _metadata(1)
    metadata["task_articulations"] = {
        "body": {"asset_key": "asset", "body_names": [], "joint_names": ["measured_joint"]}
    }
    runtime = TaskRuntime(_Scene({"asset": asset}, [[0, 0, 0]]), metadata, [build_task("cube_stack")])
    measured = runtime.states()[0]
    assert measured.joint_positions["measured_joint"] == 16777217.0
    assert measured.joint_velocities["measured_joint"] == 0.25


@pytest.mark.parametrize("representation", ["negative_stride", "big_endian", "numpy_adapter"])
def test_root_measurements_preserve_supported_numpy_representations(representation):
    class NumpyAdapter:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    def represent(value):
        value = np.asarray(value, np.float32)
        if representation == "negative_stride":
            return value[..., ::-1].copy()[..., ::-1]
        if representation == "big_endian":
            return value.astype(value.dtype.newbyteorder(">"))
        return NumpyAdapter(value)

    asset = SimpleNamespace(
        data=SimpleNamespace(
            __backend_name__="physx",
            root_link_pose_w=represent([[0, 0, 0, 0, 0, 0, 1]]),
            root_com_vel_w=represent([[1, 2, 3, 0, 2, 0]]),
            body_com_pos_b=represent([[[0, 0, 0.5]]]),
        )
    )
    metadata = _metadata(1)
    metadata["task_body_assets"] = {"body": "asset"}
    runtime = TaskRuntime(_Scene({"asset": asset}, [[0, 0, 0]]), metadata, [build_task("cube_stack")])
    actor = runtime.states()[0].actors["body"]
    np.testing.assert_array_equal(actor.linear_velocity, [0, 2, 3])
    np.testing.assert_array_equal(actor.orientation, [0, 0, 0, 1])


def test_native_contacts_keep_environment_aliases_static_tables_and_updated_particle_ranges(monkeypatch):
    definitions = [build_task("cloth_fold") for _ in range(2)]
    paths = [f"/World/envs/env_{i}/Room" for i in range(2)]
    model = SimpleNamespace(
        body_label=[path + "/Robot/gripper_body" for path in paths] + [paths[0] + "/Task/support"],
        shape_body=wp.array([0, -1, 1, -1, 2], dtype=wp.int32, device="cpu"),
        shape_label=[
            paths[0] + "/Robot/finger",
            paths[0] + "/Tabletop/top",
            paths[1] + "/Robot/finger",
            paths[1] + "/Tabletop/top",
            paths[0] + "/Task/support/geometry",
        ],
    )
    contacts = SimpleNamespace(
        rigid_contact_count=np.array([4]),
        rigid_contact_shape0=np.array([0, 1, 0, 0]),
        rigid_contact_shape1=np.array([1, 3, 2, 4]),
        rigid_contact_force=np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], np.float32),
        soft_contact_count=np.array([3]),
        soft_contact_indices=np.array([[5, 6, 7], [1005, 1006, 1007], [1005, 1006, 1007]]),
        soft_contact_shape=np.array([0, 2, 0]),
    )
    ranges = {path + "/Task/Deformables/sheet": (i * 1000, 315, None) for i, path in enumerate(paths)}
    monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(NewtonManager, "get_contacts", lambda: contacts)
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(deformable_ranges=ranges))
    positions = torch.tensor(np.stack([definition.deformables[0].vertices for definition in definitions]))
    metadata = _metadata(2)
    metadata["task_deformables"] = {"sheet": "sheet"}
    metadata["native_task_entries"] = [
        (i, definition, path + "/Task", {"sheet": path + "/Task/Deformables/sheet"})
        for i, (definition, path) in enumerate(zip(definitions, paths))
    ]
    scene = _Scene(
        {
            "sheet": SimpleNamespace(
                data=SimpleNamespace(nodal_pos_w=positions, nodal_vel_w=torch.zeros_like(positions))
            )
        },
        [[0, 0, 0], [0, 0, 0]],
    )
    runtime = TaskRuntime(scene, metadata, definitions)
    before = runtime.states()
    assert before[0].contacts == {("left_finger", "table"), ("sheet", "left_finger")}
    assert before[1].contacts == {("sheet", "left_finger")}
    ranges[paths[0] + "/Task/Deformables/sheet"] = (50, 315, None)
    after = runtime.states()
    assert after[0].contacts == {("left_finger", "table")}
    assert after[1].contacts == before[1].contacts
    assert before[0].contacts == {("left_finger", "table"), ("sheet", "left_finger")}
    definitions[0].initial["support"] = ActorState((0, 0, 0))
    assert runtime.states()[0].contacts == {("left_finger", "table"), ("left_finger", "support")}


def test_cable_reconstruction_observes_live_rest_vertex_writes(monkeypatch):
    definition = build_task("barrel_jack")
    spec = definition.deformables[0]
    spec.vertices = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float)
    poses = torch.tensor([[[0, 0, 0.5, 0, 0, 0, 1], [0, 0, 1.5, 0, 0, 0, 1]]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_segment_pose_w=poses.clone(), segment_pose_w=poses, segment_velocity_w=torch.zeros((1, 2, 6))
        )
    )
    model = SimpleNamespace(body_label=[], shape_label=[], shape_body=np.empty(0, dtype=int))
    contacts = SimpleNamespace(rigid_contact_count=np.array([0]), soft_contact_count=np.array([0]))
    monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(NewtonManager, "get_contacts", lambda: contacts)
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(deformable_ranges={}))
    metadata = _metadata(1)
    metadata["task_deformables"] = {spec.name: "cable"}
    metadata["native_task_entries"] = [(0, definition, "/World/envs/env_0/Room/Task", {spec.name: "/Cable"})]
    runtime = TaskRuntime(_Scene({"cable": asset}, [[0, 0, 0]]), metadata, [definition])
    runtime.states()  # Warm both native model mapping and the definition's rest geometry.
    before = runtime.states()
    np.testing.assert_array_equal(before[0].deformable_positions[spec.name][:, 2], [0, 1, 2])
    spec.vertices[:, 2] *= 2
    np.testing.assert_array_equal(runtime.states()[0].deformable_positions[spec.name][:, 2], [-0.5, 1, 2.5])
    np.testing.assert_array_equal(before[0].deformable_positions[spec.name][:, 2], [0, 1, 2])
