"""Linux readiness policy from captured snapshots, without opening a GPU or simulator."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def preflight():
    path = Path(__file__).parents[1] / "cloud" / "preflight.py"
    assert path.exists(), "cloud preflight implementation is missing"
    spec = importlib.util.spec_from_file_location("cloud_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot():
    return {
        "system": "Linux",
        "machine": "x86_64",
        "os_release": {"ID": "ubuntu", "VERSION_ID": "24.04"},
        "driver_version": "595.58.03",
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "memory_total_mib": 24564}],
        "host_memory_bytes": 64 * 1024**3,
        "cgroup_memory_limits": [],
        "affinity_cpus": 16,
        "cgroup_cpu_limits": [],
        "workspace_free_bytes": 160 * 1024**3,
        "workspace_ssd": True,
        "container_detected": False,
        "is_root": False,
        "cap_sys_module": False,
        "nvidia_module_present": True,
        "nvidia_driver_capabilities": None,
        "nvidia_icd_paths": ["/usr/share/vulkan/icd.d/nvidia_icd.json"],
        "vulkan": {
            "available": True,
            "nvidia": True,
            "extensions": [
                "VK_KHR_acceleration_structure",
                "VK_KHR_ray_tracing_pipeline",
                "VK_KHR_ray_query",
                "VK_KHR_deferred_host_operations",
                "VK_KHR_buffer_device_address",
                "VK_EXT_descriptor_indexing",
                "VK_KHR_spirv_1_4",
                "VK_KHR_shader_float_controls",
            ],
        },
        "runtime": {
            "versions": {
                "torch": "2.11.0+cu130",
                "isaacsim": "6.2.0",
                "rsl-rl-lib": "5.5.1",
                "newton": "1.6.1",
                "warp-lang": "1.17.0",
            },
            "torch_cuda_available": True,
            "torch_bf16_supported": True,
            "torch_device_name": "NVIDIA GeForce RTX 4090",
            "torch_memory_total_mib": 24564,
        },
        "collection_errors": [],
    }


def test_supported_runtime_defaults_to_fp32_without_bf16_requirement(preflight, snapshot):
    snapshot["runtime"]["torch_bf16_supported"] = False
    before = copy.deepcopy(snapshot)
    report = preflight.evaluate_snapshot(snapshot, phase="runtime")
    assert report["ok"] and report["errors"] == []
    assert report["learner_precision"] == "fp32"
    assert snapshot == before


@pytest.mark.parametrize(
    "phase,free_gib,okay",
    [("bootstrap", 149, False), ("bootstrap", 150, True), ("runtime", 19, False), ("runtime", 20, True)],
)
def test_phase_disk_requirement_uses_available_workspace_space(preflight, snapshot, phase, free_gib, okay):
    snapshot["workspace_free_bytes"] = free_gib * 1024**3
    report = preflight.evaluate_snapshot(snapshot, phase=phase)
    assert report["ok"] is okay
    if not okay:
        assert any("workspace" in error.lower() and "free" in error.lower() for error in report["errors"])


@pytest.mark.parametrize("version,value", [(2, "max"), (1, str(2**63 - 4096)), (1, "-1")])
def test_unlimited_memory_cgroups_do_not_replace_host_ram(preflight, snapshot, version, value):
    snapshot["cgroup_memory_limits"] = [{"version": version, "source": "fixture", "value": value}]
    report = preflight.evaluate_snapshot(snapshot)
    assert report["ok"]
    assert report["measurements"]["effective_memory_gib"] == 64


@pytest.mark.parametrize("version", [1, 2])
def test_memory_cgroup_limit_is_the_effective_ram(preflight, snapshot, version):
    snapshot["cgroup_memory_limits"] = [{"version": version, "source": "fixture", "value": str(48 * 1024**3)}]
    report = preflight.evaluate_snapshot(snapshot)
    assert not report["ok"]
    assert report["measurements"]["effective_memory_gib"] == 48
    assert any("RAM" in error for error in report["errors"])


@pytest.mark.parametrize(
    "quota,period,expected", [(799999, 100000, 7.99999), (800000, 100000, 8.0), (150000, 100000, 1.5)]
)
def test_cpu_quota_never_rounds_a_partial_cpu_up(preflight, snapshot, quota, period, expected):
    snapshot["cgroup_cpu_limits"] = [{"version": 2, "source": "fixture", "quota": str(quota), "period": str(period)}]
    report = preflight.evaluate_snapshot(snapshot)
    assert report["measurements"]["effective_cpus"] == expected
    assert report["ok"] is (expected >= 8)


def test_cpu_affinity_and_ancestor_cgroup_both_constrain_capacity(preflight, snapshot):
    snapshot["affinity_cpus"] = 12
    snapshot["cgroup_cpu_limits"] = [
        {"version": 1, "source": "parent", "quota": "600000", "period": "100000"},
        {"version": 1, "source": "child", "quota": "-1", "period": "100000"},
    ]
    report = preflight.evaluate_snapshot(snapshot)
    assert not report["ok"] and report["measurements"]["effective_cpus"] == 6
    snapshot["cgroup_cpu_limits"] = []
    snapshot["affinity_cpus"] = 7
    assert not preflight.evaluate_snapshot(snapshot)["ok"]


@pytest.mark.parametrize(
    "name,older,okay",
    [
        ("NVIDIA GeForce RTX 3090", False, False),
        ("NVIDIA GeForce RTX 3090", True, True),
        ("NVIDIA RTX A6000", True, False),
        ("NVIDIA A100-SXM4-80GB", True, False),
        ("NVIDIA GeForce RTX 5090", False, True),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", False, True),
        ("NVIDIA RTX 6000 Ada Generation", False, True),
    ],
)
def test_gpu_allowlist_keeps_older_exception_specific_to_3090_class(preflight, snapshot, name, older, okay):
    snapshot["gpus"][0]["name"] = name
    report = preflight.evaluate_snapshot(snapshot, allow_older_gpu=older)
    assert report["ok"] is okay
    if okay and "3090" in name:
        assert any("published" in warning and "minimum" in warning for warning in report["warnings"])


def test_a_supported_name_cannot_hide_insufficient_vram(preflight, snapshot):
    snapshot["gpus"][0]["memory_total_mib"] = 22999
    report = preflight.evaluate_snapshot(snapshot)
    assert not report["ok"] and any("23000" in error for error in report["errors"])


@pytest.mark.parametrize("driver", ["580.65.06", "595.57.99", None, "unknown", "595.58"])
def test_cuda13_floor_is_distinct_from_required_tested_driver(preflight, snapshot, driver):
    snapshot["driver_version"] = driver
    report = preflight.evaluate_snapshot(snapshot)
    assert not report["ok"] and any("driver" in error.lower() for error in report["errors"])
    assert report["requirements"]["tested_driver_minimum"] == "595.58.03"
    assert report["requirements"]["cuda13_driver_floor"] == "580.65.06"


@pytest.mark.parametrize("caps", ["compute,utility", "", "video,display"])
def test_explicit_container_capabilities_must_mount_graphics(preflight, snapshot, caps):
    snapshot["container_detected"] = True
    snapshot["nvidia_driver_capabilities"] = caps
    report = preflight.evaluate_snapshot(snapshot)
    assert not report["ok"] and any("graphics" in error.lower() for error in report["errors"])


def test_container_root_and_loaded_module_never_prove_unlock(preflight, snapshot):
    snapshot.update(container_detected=True, is_root=True, cap_sys_module=True, nvidia_module_present=True)
    snapshot["nvidia_driver_capabilities"] = "all"
    report = preflight.evaluate_snapshot(snapshot)
    assert report["ok"]
    assert report["unlock_status"] == report["host_kernel_control"] == "unverified"
    assert report["measurements"]["is_root"] is True
    assert any("host" in warning and "unverified" in warning for warning in report["warnings"])


def test_bootstrap_can_install_vulkan_tools_but_runtime_requires_nvidia_rtx_extensions(preflight, snapshot):
    snapshot["vulkan"] = {"available": False, "nvidia": None, "extensions": []}
    snapshot["nvidia_icd_paths"] = []
    bootstrap = preflight.evaluate_snapshot(snapshot)
    assert bootstrap["ok"] and bootstrap["warnings"]
    assert not preflight.evaluate_snapshot(snapshot, phase="runtime")["ok"]


@pytest.mark.parametrize("field,value", [("nvidia", False), ("extensions", ["VK_KHR_ray_tracing_pipeline"])])
def test_runtime_rejects_partial_or_non_nvidia_vulkan(preflight, snapshot, field, value):
    snapshot["vulkan"][field] = value
    report = preflight.evaluate_snapshot(snapshot, phase="runtime")
    assert not report["ok"] and any("Vulkan" in error for error in report["errors"])


def test_amp_requires_bf16_only_in_runtime_phase(preflight, snapshot):
    snapshot["runtime"]["torch_bf16_supported"] = False
    assert preflight.evaluate_snapshot(snapshot, amp=True)["ok"]
    report = preflight.evaluate_snapshot(snapshot, phase="runtime", amp=True)
    assert not report["ok"] and any("BF16" in error for error in report["errors"])
    snapshot["runtime"]["torch_bf16_supported"] = True
    assert preflight.evaluate_snapshot(snapshot, phase="runtime", amp=True)["ok"]


@pytest.mark.parametrize(
    "key,value", [("torch_cuda_available", False), ("torch_cuda_available", "yes"), ("versions", {"torch": "2.10.0"})]
)
def test_runtime_requires_cuda_and_all_pinned_distribution_versions(preflight, snapshot, key, value):
    snapshot["runtime"][key] = value
    assert not preflight.evaluate_snapshot(snapshot, phase="runtime")["ok"]


@pytest.mark.parametrize(
    "name,memory",
    [
        ("NVIDIA GeForce RTX 3090", 24564),
        ("NVIDIA A100-SXM4-80GB", 81920),
        ("NVIDIA GeForce RTX 4090", 20000),
        (None, 24564),
    ],
)
def test_runtime_validates_the_gpu_pytorch_selected_not_an_unused_qualified_card(preflight, snapshot, name, memory):
    snapshot["runtime"].update(torch_device_name=name, torch_memory_total_mib=memory)
    assert not preflight.evaluate_snapshot(snapshot, phase="runtime")["ok"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("host_memory_bytes", None),
        ("affinity_cpus", None),
        ("workspace_free_bytes", "lots"),
        ("gpus", []),
        ("gpus", "4090"),
        ("cgroup_memory_limits", [{"value": "broken"}]),
        ("cgroup_cpu_limits", [{"quota": "max", "period": "broken"}]),
    ],
)
def test_unknown_or_malformed_required_measurements_fail_closed(preflight, snapshot, key, value):
    snapshot[key] = value
    assert not preflight.evaluate_snapshot(snapshot)["ok"]


@pytest.mark.parametrize("key", ["cgroup_memory_limits", "cgroup_cpu_limits"])
def test_absent_cgroup_measurements_are_unknown_rather_than_unlimited(preflight, snapshot, key):
    del snapshot[key]
    assert not preflight.evaluate_snapshot(snapshot)["ok"]


@pytest.mark.parametrize(
    "key,value", [("system", "Windows"), ("machine", "aarch64"), ("os_release", {"ID": "debian", "VERSION_ID": "12"})]
)
def test_only_qualified_linux_x86_64_ubuntu_platforms_pass(preflight, snapshot, key, value):
    snapshot[key] = value
    assert not preflight.evaluate_snapshot(snapshot)["ok"]


def test_container_overlay_storage_is_explicitly_unverified_not_reported_as_ssd(preflight, snapshot):
    snapshot["workspace_ssd"] = None
    report = preflight.evaluate_snapshot(snapshot)
    assert report["ok"] and any("provider" in warning and "bandwidth" in warning for warning in report["warnings"])
    snapshot["workspace_ssd"] = False
    assert not preflight.evaluate_snapshot(snapshot)["ok"]


def test_cli_writes_the_same_report_to_stdout_and_requested_json(preflight, snapshot, monkeypatch, tmp_path, capsys):
    captured = []

    def collect(workspace, runtime=False):
        captured.append((workspace, runtime))
        return snapshot

    monkeypatch.setattr(preflight, "collect_snapshot", collect)
    destination = tmp_path / "reports" / "preflight.json"
    monkeypatch.setattr(
        sys, "argv", ["preflight.py", "--phase", "runtime", "--workspace", str(tmp_path), "--json", str(destination)]
    )
    assert preflight.main() == 0
    assert captured == [(tmp_path, True)]
    assert json.loads(capsys.readouterr().out) == json.loads(destination.read_text(encoding="utf-8"))
    snapshot["driver_version"] = "580.65.06"
    assert preflight.main() == 1


@pytest.mark.parametrize("hierarchy,okay", [("0::/tenant/child", True), ("unparseable", False)])
def test_collector_reads_cgroup_ancestors_and_scopes_nvidia_icd_to_vulkan_subprocess(
    preflight, monkeypatch, tmp_path, hierarchy, okay
):
    content = {
        "/etc/os-release": 'ID=ubuntu\nVERSION_ID="24.04"',
        "/proc/meminfo": "MemTotal: 67108864 kB",
        "/proc/self/cgroup": hierarchy,
        "/proc/self/status": "CapEff:\t0000000000010000",
        "/sys/fs/cgroup/tenant/child/memory.max": "max",
        "/sys/fs/cgroup/tenant/memory.max": str(64 * 1024**3),
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/tenant/child/cpu.max": "max 100000",
        "/sys/fs/cgroup/tenant/cpu.max": "800000 100000",
        "/sys/fs/cgroup/cpu.max": "max 100000",
    }
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(preflight.os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setattr(preflight, "_read_text", lambda path: content.get(str(path).replace("\\", "/")))
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(free=160 * 1024**3))
    monkeypatch.setattr(preflight, "_workspace_ssd", lambda path: None)
    monkeypatch.setattr(preflight, "_nvidia_icds", lambda: ["/test/nvidia_icd.json"])
    monkeypatch.setenv("VK_DRIVER_FILES", "original-choice")
    calls = []

    def run(command, env=None, timeout=20):
        calls.append(command)
        if command[0] == "nvidia-smi":
            return 0, "595.58.03, NVIDIA GeForce RTX 4090, 24564", ""
        if command == ["vulkaninfo"]:
            assert env["VK_DRIVER_FILES"] == "/test/nvidia_icd.json"
            return 0, "NVIDIA\n" + "\n".join(preflight.REQUIRED_VULKAN_EXTENSIONS), ""
        pytest.fail("Unexpected external command: " + repr(command))

    monkeypatch.setattr(preflight, "_run", run)
    collected = preflight.collect_snapshot(tmp_path)
    report = preflight.evaluate_snapshot(collected)
    assert report["ok"] is okay
    assert report["measurements"]["effective_cpus"] == (8 if okay else 16)
    assert report["measurements"]["effective_memory_gib"] == 64
    assert collected["cap_sys_module"] is True
    assert "runtime" not in collected
    assert preflight.os.environ["VK_DRIVER_FILES"] == "original-choice"
    assert len(calls) == 2


def test_icd_discovery_checks_json_library_instead_of_filename(preflight, monkeypatch, tmp_path):
    nvidia = tmp_path / "actual-driver.json"
    nvidia.write_text(json.dumps({"ICD": {"library_path": "libGLX_nvidia.so.0"}}), encoding="utf-8")
    misleading = tmp_path / "nvidia_icd.json"
    misleading.write_text(json.dumps({"ICD": {"library_path": "libvulkan_intel.so"}}), encoding="utf-8")
    monkeypatch.setattr(preflight.Path, "glob", lambda path, pattern: [])
    monkeypatch.setenv("VK_DRIVER_FILES", str(nvidia))
    monkeypatch.setenv("VK_ICD_FILENAMES", str(misleading))
    assert preflight._nvidia_icds() == [str(nvidia)]


def test_runtime_probe_rejects_non_json_or_failed_subprocess(preflight, monkeypatch):
    monkeypatch.setattr(preflight, "_run", lambda *args, **kwargs: (0, "unrelated log output", ""))
    assert preflight._runtime_snapshot()["error"]
    monkeypatch.setattr(
        preflight, "_run", lambda *args, **kwargs: (1, 'CLOUD_PREFLIGHT_JSON={"versions": {}}', "probe failed")
    )
    assert preflight._runtime_snapshot()["error"] == "probe failed"


def test_unreadable_text_is_missing_evidence_instead_of_an_uncaught_decode_error(preflight, tmp_path):
    malformed = tmp_path / "non-utf8"
    malformed.write_bytes(b"\xff\xfe\xff")
    assert preflight._read_text(malformed) is None
