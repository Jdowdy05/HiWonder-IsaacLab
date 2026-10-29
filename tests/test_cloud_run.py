import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "cloud/run.py"


def load():
    spec = importlib.util.spec_from_file_location("cloud_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_keeps_native_rgb_and_eka_model_and_limits_run(tmp_path):
    module = load()
    plan = module.plan_run(tmp_path, "smoke", task="peg_insert", precision="bf16")
    assert plan["profile"]["robot_model"] == "eka_v6"
    assert "camera" not in plan["profile"]
    assert plan["profile"]["episode_seconds"] == 0.2
    assert "--mixed-precision" in plan["command"]
    assert "--headless" not in plan["command"] and "--viz" not in plan["command"]
    assert plan["wall_seconds"] == 600 and plan["max_rss_gib"] == 24


def test_normal_training_has_original_horizon_and_sgs_bank(tmp_path):
    module = load()
    plan = module.plan_run(tmp_path, "train", task="cloth_fold:rag", precision=None)
    assert "episode_seconds" not in plan["profile"]
    assert "--no-mixed-precision" in plan["command"]
    assert plan["command"][plan["command"].index("--bank-size") + 1] == "32768"
    assert "cloth_fold:rag" in plan["command"]


def test_resume_preserves_checkpoint_profile_precision_and_dimensions(tmp_path):
    module = load()
    checkpoint = tmp_path / "runs/run/final.pt"
    plan = module.plan_run(tmp_path, "resume", checkpoint=checkpoint)
    assert "--resume" in plan["command"]
    assert "--profile" not in plan["command"]
    assert "--num-envs" not in plan["command"]
    assert "--mixed-precision" not in plan["command"] and "--no-mixed-precision" not in plan["command"]


@pytest.mark.parametrize("field,value", (("envs", 0), ("iterations", 0), ("wall_seconds", -1), ("max_rss_gib", 0)))
def test_invalid_resource_or_iteration_settings_fail(tmp_path, field, value):
    with pytest.raises(ValueError):
        load().plan_run(tmp_path, "pilot", **{field: value})


def test_unknown_task_does_not_escape_output_directory(tmp_path):
    with pytest.raises(ValueError):
        load().plan_run(tmp_path, "train", task="../../escape")


def test_launch_sidecars_do_not_precreate_trainers_exclusive_output_directory(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)

    def guard(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        assert not output.exists(), "Trainer creates its output directory with exist_ok=False"
        output.mkdir(parents=True)
        Path(kwargs["report"]).write_text(json.dumps({"complete": True}))
        return 0

    monkeypatch.setitem(sys.modules, "guard", types.SimpleNamespace(run_guarded=guard))
    monkeypatch.setattr(sys, "argv", ["run.py", "smoke", "--workspace", str(tmp_path)])
    assert module.main() == 0
