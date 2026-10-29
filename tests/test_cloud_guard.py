import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load():
    path = Path(__file__).parents[1] / "cloud/guard.py"
    spec = importlib.util.spec_from_file_location("cloud_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_rejects_invalid_budgets_before_starting_process(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setattr(module, "sys_linux", lambda: True)
    with pytest.raises(ValueError):
        module.run_guarded(
            ["unused"], cwd=tmp_path, log=tmp_path / "log", report=tmp_path / "report", wall_seconds=0, max_rss_gib=1
        )
    assert not (tmp_path / "log").exists()


@pytest.mark.parametrize("version", (1, 2))
def test_remaining_memory_uses_nested_cgroup_ancestor_not_host_free_memory(monkeypatch, version):
    module = load()
    # Load the shared path parser before replacing filesystem reads.
    if hasattr(module, "cgroup_path_function"):
        module.cgroup_path_function()
    gib = 1024**3
    if version == 2:
        mapping = {
            "/proc/self/cgroup": "0::/tenant/job\n",
            "/sys/fs/cgroup/memory.max": "max",
            "/sys/fs/cgroup/memory.current": "0",
            "/sys/fs/cgroup/tenant/memory.max": str(64 * gib),
            "/sys/fs/cgroup/tenant/memory.current": str(60 * gib),
        }
    else:
        mapping = {
            "/proc/self/cgroup": "5:memory:/tenant/job\n",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(2**63 - 1),
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": "0",
            "/sys/fs/cgroup/memory/tenant/memory.limit_in_bytes": str(64 * gib),
            "/sys/fs/cgroup/memory/tenant/memory.usage_in_bytes": str(60 * gib),
        }
    mapping["/proc/meminfo"] = f"MemAvailable: {80 * 1024**2} kB\n"

    def read(path, *args, **kwargs):
        key = str(path).replace("\\", "/")
        if key not in mapping:
            raise FileNotFoundError(key)
        return mapping[key]

    monkeypatch.setattr(Path, "read_text", read)
    assert module.available_memory() == 4 * gib


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owned process-group behavior")
def test_guard_records_natural_completion_without_other_process_kills(tmp_path):
    code = load().run_guarded(
        [sys.executable, "-c", "print('completed')"],
        cwd=tmp_path,
        log=tmp_path / "log",
        report=tmp_path / "status.json",
        wall_seconds=5,
        max_rss_gib=1,
        reserve_gib=0.01,
    )
    result = json.loads((tmp_path / "status.json").read_text())
    assert code == 0 and result["complete"] and result["child_exit_code"] == 0
    assert "completed" in (tmp_path / "log").read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owned process-group behavior")
def test_guard_stops_sleeping_owned_job_at_deadline(tmp_path):
    code = load().run_guarded(
        [sys.executable, "-c", "import time;time.sleep(10)"],
        cwd=tmp_path,
        log=tmp_path / "log",
        report=tmp_path / "status.json",
        wall_seconds=0.1,
        max_rss_gib=1,
        reserve_gib=0.01,
    )
    result = json.loads((tmp_path / "status.json").read_text())
    assert code == 124 and not result["complete"]
    assert result["stop_reason"] == "wall_time_budget"


def test_all_documented_presets_build_real_task_definitions():
    from isaaclab_hiwonder.tasks.catalog import build_task

    presets = json.loads((Path(__file__).parents[1] / "cloud/task_presets.json").read_text())
    assert len(presets) == len(set(presets)) == 30
    for preset in presets:
        name, _, variant = preset.partition(":")
        assert build_task(name, variant or None) is not None
