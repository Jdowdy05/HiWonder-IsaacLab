# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sampler policy persistence at the actual train/resume command boundary."""

import importlib.util
import sys
import types
from contextlib import contextmanager, suppress
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from isaaclab_hiwonder.config import HiWonderEnvConfig
from isaaclab_hiwonder.rl.config import PPOTrainingConfig


def _load_cli():
    path = Path(__file__).parents[1] / "tools" / "train_hiwonder.py"
    spec = importlib.util.spec_from_file_location("train_hiwonder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_preserves_uniform_sampling_and_rejects_an_unrecorded_policy_change():
    module = _load_cli()
    saved = {"hiwonder": {"environment": {"sampling": "uniform"}, "metadata": {"arguments": {"sampling": "uniform"}}}}
    assert module.resolve_sampling("train", None, saved) == "uniform"
    assert module.resolve_sampling("train", "uniform", saved) == "uniform"
    with pytest.raises(ValueError, match="sampling"):
        module.resolve_sampling("train", "sgs", saved)
    assert module.resolve_sampling("train", None, None) == "sgs"
    assert module.resolve_sampling("eval", None, saved) == "uniform"
    # Prior checkpoints have the mode in run metadata, before it was added to env state.
    del saved["hiwonder"]["environment"]["sampling"]
    assert module.resolve_sampling("train", None, saved) == "uniform"


@pytest.mark.parametrize(
    "flags, expected", [([], None), (["--mixed-precision"], True), (["--no-mixed-precision"], False)]
)
def test_cli_precision_is_an_optional_explicit_boolean(monkeypatch, flags, expected):
    module = _load_cli()
    launcher = types.ModuleType("isaaclab.app")
    launcher.add_launcher_args = lambda parser: None
    monkeypatch.setitem(sys.modules, "isaaclab.app", launcher)
    monkeypatch.setattr(sys, "argv", ["train_hiwonder.py", "train", *flags])
    assert getattr(module._arguments(), "mixed_precision", "missing") is expected


@pytest.fixture
def cli_configuration(monkeypatch, tmp_path):
    """Execute the command's real configuration path, stopping before any learner or simulator work."""
    module = _load_cli()
    profile = replace(HiWonderEnvConfig(num_envs=4, seed=17, frame_stack=2), control_hz=60)
    profile = replace(profile, camera=replace(profile.camera, width=320, height=240))
    runner_config = PPOTrainingConfig(
        frame_stack=2, num_steps_per_env=4, num_learning_epochs=1, num_mini_batches=2, seed=17
    ).to_runner_config()
    saved = {
        "hiwonder": {
            "version": 1,
            "runner_config": runner_config,
            "environment": {"sampling": "sgs", "sampler": {"num_configs": 32}},
            "metadata": {"profile": asdict(profile), "arguments": {"task": "peg_insert", "sampling": "sgs"}},
        }
    }
    checkpoint = tmp_path / "copied.pt"
    torch.save(saved, checkpoint)
    captured = {}

    def install(name, **members):
        replacement = types.ModuleType(name)
        replacement.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, replacement)

    @contextmanager
    def launch(*args):
        yield

    class Captured(Exception):
        pass

    class Runner:
        def __init__(self, env, config, **kwargs):
            captured["runner_config"] = config
            raise Captured

    def make_cfg(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(sim=types.SimpleNamespace())

    def physics_config(definition):
        captured["physics_definition"] = definition
        return object()

    def provenance(env, destination, runner_config, arguments):
        captured["arguments"] = dict(arguments)
        return {}

    install("isaaclab.app", launch_simulation=launch)
    install("isaaclab.sim", SimulationCfg=lambda **kwargs: types.SimpleNamespace(**kwargs))
    install("isaaclab_hiwonder.tasks.native_backend", physics_config=physics_config)
    install("isaaclab_hiwonder.rl.runner", HiWonderRunner=Runner)
    install("isaaclab_hiwonder.rl.provenance", write_run_provenance=provenance)
    install("isaaclab_rl.rsl_rl", RslRlVecEnvWrapper=lambda env, **kwargs: env)
    install(
        "isaaclab_hiwonder.sim.env",
        make_cfg=make_cfg,
        make_env=lambda *args, **kwargs: types.SimpleNamespace(close=lambda: None),
    )

    def invoke(**overrides):
        values = dict(
            mode="train",
            task=None,
            num_envs=None,
            seed=None,
            actor_mode=None,
            frame_stack=None,
            reset_families=None,
            grasp_library=None,
            profile=None,
            lighting=None,
            sampling=None,
            bank_size=None,
            iterations=1,
            rollout_steps=None,
            epochs=None,
            minibatches=None,
            smoke=False,
            resume=checkpoint,
            initialize_from=None,
            checkpoint=None,
            episodes=1,
            max_eval_steps=None,
            device="cpu",
            output=tmp_path / f"run_{len(list(tmp_path.iterdir()))}",
        )
        values.update(overrides)
        monkeypatch.setattr(module, "_arguments", lambda: types.SimpleNamespace(**values))
        captured.clear()
        with suppress(Captured):
            module.main()
        return captured

    return invoke, profile, checkpoint


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_copied_checkpoint_uses_embedded_camera_history_and_profile(cli_configuration, mode):
    invoke, expected, checkpoint = cli_configuration
    result = invoke(
        mode=mode, resume=checkpoint if mode == "train" else None, checkpoint=checkpoint if mode == "eval" else None
    )
    profile = result["profile"]
    assert profile.frame_stack == expected.frame_stack
    assert profile.camera == expected.camera
    assert profile.control_hz == expected.control_hz
    assert profile.seed == expected.seed + (1_000_003 if mode == "eval" else 0)
    if mode == "train":
        result = invoke(rollout_steps=4, epochs=1, minibatches=2)
        assert result["runner_config"]["num_steps_per_env"] == 4


@pytest.mark.parametrize("override", [dict(rollout_steps=17), dict(epochs=3), dict(minibatches=4), dict(smoke=True)])
def test_exact_resume_rejects_incompatible_explicit_learner_overrides(cli_configuration, override):
    invoke, _, _ = cli_configuration
    with pytest.raises(ValueError, match="[Rr]esume.*(rollout|epoch|minibatch|smoke)"):
        invoke(**override)


def test_fresh_initialization_applies_explicit_learner_counts_and_new_seed(cli_configuration):
    invoke, _, checkpoint = cli_configuration
    result = invoke(resume=None, initialize_from=checkpoint, seed=123, rollout_steps=17, epochs=3, minibatches=4)
    config = result["runner_config"]
    assert config["num_steps_per_env"] == 17
    assert config["algorithm"]["num_learning_epochs"] == 3
    assert config["algorithm"]["num_mini_batches"] == 4
    assert config["seed"] == result["profile"].seed == 123


@pytest.mark.parametrize("saved_precision", [False, True])
def test_exact_resume_preserves_unspecified_precision_and_records_the_resolved_value(
    cli_configuration, saved_precision
):
    invoke, _, checkpoint = cli_configuration
    saved = torch.load(checkpoint, weights_only=False)
    saved["hiwonder"]["runner_config"]["algorithm"]["use_mixed_precision"] = saved_precision
    torch.save(saved, checkpoint)
    result = invoke()
    assert result["runner_config"]["algorithm"]["use_mixed_precision"] is saved_precision
    expected = "bf16_autocast" if saved_precision else "fp32"
    assert result["runner_config"]["hiwonder"].get("learner_precision") == expected
    assert result["arguments"]["mixed_precision"] is saved_precision
    assert result["arguments"]["learner_precision"] == expected


@pytest.mark.parametrize("saved_precision", [False, True])
def test_exact_resume_rejects_precision_changes_but_accepts_the_recorded_choice(cli_configuration, saved_precision):
    invoke, _, checkpoint = cli_configuration
    saved = torch.load(checkpoint, weights_only=False)
    saved["hiwonder"]["runner_config"]["algorithm"]["use_mixed_precision"] = saved_precision
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="Resume.*precision.*initialize-from"):
        invoke(mixed_precision=not saved_precision)
    assert (
        invoke(mixed_precision=saved_precision)["runner_config"]["algorithm"]["use_mixed_precision"] is saved_precision
    )


@pytest.mark.parametrize("requested", [False, True])
def test_fresh_optimizer_initialization_accepts_an_explicit_precision_choice(cli_configuration, requested):
    invoke, _, checkpoint = cli_configuration
    result = invoke(resume=None, initialize_from=checkpoint, mixed_precision=requested)
    assert result["runner_config"]["algorithm"]["use_mixed_precision"] is requested
    assert result["profile"].camera.width == 320 and result["profile"].camera.height == 240


def test_legacy_resume_without_precision_uses_fp32(cli_configuration):
    invoke, _, checkpoint = cli_configuration
    saved = torch.load(checkpoint, weights_only=False)
    del saved["hiwonder"]["runner_config"]["algorithm"]["use_mixed_precision"]
    torch.save(saved, checkpoint)
    assert invoke()["runner_config"]["algorithm"].get("use_mixed_precision") is False
    with pytest.raises(ValueError, match="Resume.*precision"):
        invoke(mixed_precision=True)


def test_fresh_task_initialization_resolves_the_new_tasks_backend(cli_configuration):
    invoke, _, checkpoint = cli_configuration
    saved = torch.load(checkpoint, weights_only=False)
    saved["hiwonder"]["metadata"]["profile"].update(task="cloth_fold:shirt", backend="newton_vbd")
    saved["hiwonder"]["metadata"]["arguments"]["task"] = "cloth_fold:shirt"
    torch.save(saved, checkpoint)
    result = invoke(resume=None, initialize_from=checkpoint, task="peg_insert", seed=123)
    assert result["profile"].backend == "physx"
    saved["hiwonder"]["metadata"]["profile"].update(
        task="knob", backend="physx", task_parameters={"target_angle_rad": 0.8}
    )
    saved["hiwonder"]["metadata"]["arguments"]["task"] = "knob"
    torch.save(saved, checkpoint)
    result = invoke(resume=None, initialize_from=checkpoint, task="peg_insert", seed=123)
    assert result["profile"].task_parameters == {}


def test_explicit_profile_task_precedes_checkpoint_and_keeps_matching_goal_parameters(cli_configuration):
    import json

    invoke, _, checkpoint = cli_configuration
    selected = checkpoint.parent / "knob-profile.json"
    selected.write_text(json.dumps({"task": "knob", "task_parameters": {"target_angle_rad": 0.8}, "frame_stack": 2}))
    result = invoke(resume=None, initialize_from=checkpoint, profile=selected, seed=123)
    assert result["profile"].task == "knob"
    assert result["profile"].task_parameters == {"target_angle_rad": 0.8}
    assert result["physics_definition"].goal.tolist() == pytest.approx([0.8])
    assert result["profile"].backend == "physx"
    result = invoke(resume=None, initialize_from=checkpoint, profile=selected, task="switch", seed=123)
    assert result["profile"].task == "switch" and result["profile"].task_parameters == {}


def test_held_out_cli_rejects_the_checkpoint_training_seed(cli_configuration):
    invoke, profile, checkpoint = cli_configuration
    with pytest.raises(ValueError, match="[Hh]eld.out.*seed"):
        invoke(mode="eval", resume=None, checkpoint=checkpoint, seed=profile.seed)
