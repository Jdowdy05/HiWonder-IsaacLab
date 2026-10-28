# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise real tool action allocation against the raw SDK contract, without a simulator."""

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_hiwonder.config import HiWonderEnvConfig
from isaaclab_hiwonder.sim.control import MotorChannel

ROOT = Path(__file__).parents[1]


class FirstStepCaptured(Exception):
    pass


class Scene(dict):
    def __init__(self, count):
        super().__init__(wrist_camera=object())
        self.env_origins = torch.zeros(count, 3)

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


class RawEnvironment:
    """Only expose fields the raw DirectRLEnv/HiWonder contract actually provides."""

    def __init__(self, count, width):
        self.num_envs = count
        self.device = "cpu"
        self.cfg = types.SimpleNamespace(action_space=width)
        self.targets = torch.zeros(count, width)
        self.joint_order = list(range(width))
        self.channel = MotorChannel(count, width, 1 / 30, 0.5, "cpu")
        self.runtime = types.SimpleNamespace(states=lambda: [], time=np.zeros(count))
        self.scene = Scene(count)
        self.sim = types.SimpleNamespace(
            set_camera_view=lambda *args: None,
            is_running=lambda: True,
            get_physics_step_count=lambda: 0,
            forward=lambda: None,
        )
        self.randomization = types.SimpleNamespace(snapshot=lambda: {"table": {"height": torch.ones(count)}})
        self.captured = None

    def reset(self):
        return {}, {}

    def _reset_idx(self, ids):
        pass

    def set_lighting_preview(self, overrides):
        return {}

    def step(self, actions):
        self.captured = actions.clone()
        raise FirstStepCaptured

    def close(self):
        pass


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("robot_model,width", (("stock", 6), ("eka_v6", 9)))
@pytest.mark.parametrize("tool", ("inspect_hiwonder", "check_hiwonder_tasks", "check_hiwonder_runtime"))
def test_tools_send_zero_actions_using_raw_config_without_num_actions(monkeypatch, tmp_path, robot_model, width, tool):
    module = load_tool(tool)
    env = RawEnvironment(3, width)
    assert not hasattr(env, "num_actions")
    with pytest.raises(FirstStepCaptured):
        if tool == "inspect_hiwonder":
            profile = HiWonderEnvConfig(robot_model=robot_model, num_envs=3)
            # Isaac Lab3 uses explicit visualizers and supplies no headless flag.
            args = types.SimpleNamespace(visualizer=["kit"])
            monkeypatch.setattr(module, "_ImagePanel", lambda *args: None)
            server = types.SimpleNamespace(drain_commands=lambda: [], url="http://127.0.0.1:1/")
            module._inspect(env, profile, args, server)
        elif tool == "check_hiwonder_tasks":
            # The actual native snapshot needs a simulator. Keep reset/action orchestration real.
            monkeypatch.setattr(module, "retained_physics_snapshot", lambda env: {})
            module.verify_indexed_reset(env, tmp_path, "peg_insert")
        else:

            def install(name, **members):
                replacement = types.ModuleType(name)
                replacement.__dict__.update(members)
                monkeypatch.setitem(sys.modules, name, replacement)

            @contextmanager
            def launch(*args, **kwargs):
                yield

            install("isaaclab.app", launch_simulation=launch)
            install("isaaclab.sim", SimulationCfg=lambda **kwargs: types.SimpleNamespace(**kwargs))
            install("isaaclab_rl.rsl_rl", RslRlVecEnvWrapper=lambda raw: raw)
            install(
                "isaaclab_hiwonder.sim.env",
                make_cfg=lambda *args, **kwargs: types.SimpleNamespace(sim=types.SimpleNamespace()),
                make_env=lambda **kwargs: env,
            )
            monkeypatch.setattr(
                sys,
                "argv",
                [tool, "--device", "cpu", "--robot-model", robot_model, "--output-directory", str(tmp_path)],
            )
            module.main()
    assert env.captured.shape == (3, width)
    assert env.captured.device.type == "cpu" and env.captured.dtype == torch.float32
    assert torch.count_nonzero(env.captured).item() == 0
