#!/usr/bin/env python3
"""Read-only Linux cloud readiness checks; no simulator app or kernel changes.

``evaluate_snapshot`` accepts the dictionary produced by ``collect_snapshot``.
Measurements retain raw cgroup limits so hierarchy and fractional CPU quotas are
auditable. Bootstrap checks hardware; runtime additionally probes the pinned
Python environment and NVIDIA Vulkan extensions. SSD backing hidden by a
container is reported as unknown, never inferred from free disk space.
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

GIB = 1024**3
TESTED_DRIVER_MINIMUM = "595.58.03"
CUDA13_DRIVER_FLOOR = "580.65.06"
PINNED_VERSIONS = {
    "torch": "2.11.0",
    "isaacsim": "6.2.0",
    "rsl-rl-lib": "5.5.1",
    "newton": "1.6.1",
    "warp-lang": "1.17.0",
}
REQUIRED_VULKAN_EXTENSIONS = frozenset(
    (
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_ray_query",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_buffer_device_address",
        "VK_EXT_descriptor_indexing",
        "VK_KHR_spirv_1_4",
        "VK_KHR_shader_float_controls",
    )
)


def _integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return None


def _driver_version(value):
    if isinstance(value, str) and re.fullmatch(r"\d+\.\d+\.\d+", value):
        return tuple(int(part) for part in value.split("."))
    return None


def _version_matches(value, expected):
    if not isinstance(value, str):
        return False
    release = value.split("+", 1)[0]
    if not re.fullmatch(r"\d+(?:\.\d+)+", release):
        return False
    actual = tuple(int(part) for part in release.split("."))
    required = tuple(int(part) for part in expected.split("."))
    width = max(len(actual), len(required))
    return actual + (0,) * (width - len(actual)) == required + (0,) * (width - len(required))


def _effective_memory(snapshot, errors):
    memory = _integer(snapshot.get("host_memory_bytes"))
    if memory is None or memory <= 0:
        errors.append("Host RAM is unknown or malformed")
        memory = None
    limits = snapshot.get("cgroup_memory_limits")
    if not isinstance(limits, list):
        errors.append("Cgroup RAM limits are malformed")
        limits = []
    for entry in limits:
        if not isinstance(entry, dict):
            errors.append("Cgroup RAM limit is malformed")
            continue
        value, version = entry.get("value"), entry.get("version")
        limit = _integer(value)
        if version == 2 and value == "max":
            continue
        if version == 1 and limit is not None and (limit == -1 or limit >= 2**60):
            continue
        if version not in (1, 2) or limit is None or limit <= 0:
            errors.append(f"Cgroup RAM limit is unknown or malformed: {entry.get('source', 'unknown')}")
        elif memory is not None:
            memory = min(memory, limit)
    if memory is not None and memory < 60 * GIB:
        errors.append("Require >= 60 GiB effective RAM after cgroup limits for a 64 GB class host")
    return memory


def _effective_cpus(snapshot, errors):
    affinity = _integer(snapshot.get("affinity_cpus"))
    cpus = float(affinity) if affinity is not None and affinity > 0 else None
    if cpus is None:
        errors.append("CPU affinity capacity is unknown or malformed")
    limits = snapshot.get("cgroup_cpu_limits")
    if not isinstance(limits, list):
        errors.append("Cgroup CPU limits are malformed")
        limits = []
    for entry in limits:
        if not isinstance(entry, dict):
            errors.append("Cgroup CPU limit is malformed")
            continue
        quota, period = entry.get("quota"), _integer(entry.get("period"))
        version = entry.get("version")
        parsed_quota = _integer(quota)
        unlimited = (version == 2 and quota == "max") or (version == 1 and parsed_quota == -1)
        if (
            version not in (1, 2)
            or period is None
            or period <= 0
            or (not unlimited and (parsed_quota is None or parsed_quota <= 0))
        ):
            errors.append(f"Cgroup CPU quota is unknown or malformed: {entry.get('source', 'unknown')}")
        elif not unlimited and cpus is not None:
            cpus = min(cpus, parsed_quota / period)
    if cpus is not None and (not math.isfinite(cpus) or cpus < 8):
        errors.append("Require >= 8 effective CPUs after affinity/cgroup quota; fractional CPUs are not rounded up")
    return cpus


def _check_vulkan(snapshot, phase, errors, warnings):
    vulkan = snapshot.get("vulkan")
    vulkan = vulkan if isinstance(vulkan, dict) else {}
    icds = snapshot.get("nvidia_icd_paths")
    valid_icds = isinstance(icds, list) and bool(icds) and all(isinstance(path, str) and path for path in icds)
    if vulkan.get("available") is not True:
        (errors if phase == "runtime" else warnings).append(
            "Vulkan tooling/probe is unavailable; install vulkan-tools before runtime validation"
        )
    if not valid_icds:
        (errors if phase == "runtime" else warnings).append("NVIDIA Vulkan ICD path is unverified or missing")
    if phase == "runtime" and vulkan.get("available") is True:
        if vulkan.get("nvidia") is not True:
            errors.append("Vulkan must enumerate the NVIDIA ICD/device")
        extensions = vulkan.get("extensions")
        if not isinstance(extensions, list) or not all(isinstance(item, str) for item in extensions):
            errors.append("Vulkan device extensions are unknown or malformed")
        else:
            missing = REQUIRED_VULKAN_EXTENSIONS - set(extensions)
            if missing:
                errors.append("Vulkan RTX extensions missing: " + ", ".join(sorted(missing)))


def _check_gpus(gpus, allow_older_gpu, errors, warnings, scope="GPU"):
    if not isinstance(gpus, list) or not gpus:
        errors.append(f"{scope} name and VRAM measurements are required")
        gpus = []
    eligible = False
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict) or not isinstance(gpu.get("name"), str):
            errors.append(f"{scope} {index} name is unknown or malformed")
            continue
        name = gpu["name"]
        memory = _integer(gpu.get("memory_total_mib"))
        if memory is None or memory < 0:
            errors.append(f"{scope} {index} VRAM is unknown or malformed")
            continue
        modern = bool(
            re.search(r"\bRTX\s+(?:4090|5090)\b", name, re.IGNORECASE)
            or re.search(r"\bRTX\s+(?:PRO\s+)?\d+\b.*\b(?:Ada|Blackwell)\b", name, re.IGNORECASE)
        )
        older = bool(re.search(r"\bRTX\s+3090(?:\s+Ti)?\b", name, re.IGNORECASE))
        if memory >= 23000 and (modern or (older and allow_older_gpu)):
            eligible = True
            if older:
                warning = f"{name} explicitly allowed below the published Isaac Sim GPU minimum; qualify it locally"
                if warning not in warnings:
                    warnings.append(warning)
    if not eligible:
        errors.append(
            f"{scope}: require RTX 4090/5090 or recent Ada/Blackwell RTX Pro with >= 23000 MiB VRAM; "
            "RTX 3090 needs --allow-older-gpu"
        )
    return gpus


def evaluate_snapshot(snapshot, phase="bootstrap", allow_older_gpu=False, amp=False):
    """Return JSON-safe evidence and failures without reading or changing the host."""
    errors, warnings = [], []
    measurements = {}
    if phase not in ("bootstrap", "runtime"):
        errors.append("phase must be bootstrap or runtime")
    if not isinstance(snapshot, dict):
        snapshot = {}
        errors.append("Snapshot must be a dictionary")
    if snapshot.get("system") != "Linux" or snapshot.get("machine") != "x86_64":
        errors.append("Require Linux x86_64")
    release = snapshot.get("os_release")
    if (
        not isinstance(release, dict)
        or release.get("ID") != "ubuntu"
        or release.get("VERSION_ID")
        not in (
            "22.04",
            "24.04",
        )
    ):
        errors.append("Require Ubuntu 22.04 or 24.04")
    driver = _driver_version(snapshot.get("driver_version"))
    if driver is None or driver < _driver_version(TESTED_DRIVER_MINIMUM):
        errors.append(
            f"Require NVIDIA driver >= {TESTED_DRIVER_MINIMUM}; CUDA 13's separate floor "
            f"{CUDA13_DRIVER_FLOOR} does not meet this tested Isaac Sim driver baseline"
        )
    measurements["driver_version"] = snapshot.get("driver_version")

    measurements["gpus"] = _check_gpus(snapshot.get("gpus"), allow_older_gpu, errors, warnings)

    memory = _effective_memory(snapshot, errors)
    measurements["effective_memory_gib"] = memory / GIB if memory is not None else None
    measurements["effective_cpus"] = _effective_cpus(snapshot, errors)

    free = _integer(snapshot.get("workspace_free_bytes"))
    disk_minimum = 20 if phase == "runtime" else 150
    measurements["workspace_free_gib"] = free / GIB if free is not None else None
    if free is None or free < disk_minimum * GIB:
        errors.append(f"Require >= {disk_minimum} GiB free in the workspace filesystem")
    ssd = snapshot.get("workspace_ssd")
    if ssd is False:
        errors.append("Workspace backing storage is rotational; require SSD/NVMe")
    elif ssd is not True:
        warnings.append(
            "Workspace SSD backing is unverified (common with container overlays); "
            "check provider-advertised SSD/NVMe and disk bandwidth"
        )

    container = snapshot.get("container_detected")
    if not isinstance(container, bool):
        errors.append("Container detection is unknown or malformed")
    for key in ("container_detected", "is_root", "cap_sys_module", "nvidia_module_present", "workspace_ssd"):
        measurements[key] = snapshot.get(key)
    if container:
        warnings.append(
            "Container root/module capabilities do not establish host kernel control; RTX unlock is unverified"
        )
    caps = snapshot.get("nvidia_driver_capabilities")
    if caps is not None and (
        not isinstance(caps, str) or not ({"graphics", "all"} & {part.strip() for part in caps.split(",")})
    ):
        errors.append(
            "NVIDIA_DRIVER_CAPABILITIES must include graphics or all when set; "
            "exporting it cannot mount missing host graphics libraries"
        )

    _check_vulkan(snapshot, phase, errors, warnings)

    if phase == "runtime":
        runtime = snapshot.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        versions = runtime.get("versions")
        versions = versions if isinstance(versions, dict) else {}
        for name, expected in PINNED_VERSIONS.items():
            if not _version_matches(versions.get(name), expected):
                errors.append(f"Require pinned {name}=={expected}; observed {versions.get(name)!r}")
        if runtime.get("torch_cuda_available") is not True:
            errors.append("PyTorch CUDA is unavailable or unverified")
        else:
            _check_gpus(
                [{"name": runtime.get("torch_device_name"), "memory_total_mib": runtime.get("torch_memory_total_mib")}],
                allow_older_gpu,
                errors,
                warnings,
                scope="Selected PyTorch GPU",
            )
        if amp and runtime.get("torch_bf16_supported") is not True:
            errors.append("--amp requires verified PyTorch BF16 support")
        if runtime.get("error"):
            errors.append("Runtime probe failed: " + str(runtime["error"]))
    collection_errors = snapshot.get("collection_errors", [])
    if not isinstance(collection_errors, list):
        errors.append("Collection error evidence is malformed")
    else:
        errors.extend(str(error) for error in collection_errors)
    return {
        "ok": not errors,
        "phase": phase,
        "errors": errors,
        "warnings": warnings,
        "learner_precision": "bf16_autocast" if amp else "fp32",
        "precision_scope": "PPO update autocast only; model parameters, physics and native camera tensors remain FP32",
        "unlock_status": "unverified",
        "host_kernel_control": "unverified",
        "measurements": measurements,
        "requirements": {
            "tested_driver_minimum": TESTED_DRIVER_MINIMUM,
            "cuda13_driver_floor": CUDA13_DRIVER_FLOOR,
            "gpu_memory_mib": 23000,
            "effective_memory_gib": 60,
            "effective_cpus": 8,
            "workspace_free_gib": disk_minimum,
            "runtime_versions": dict(PINNED_VERSIONS),
        },
        "snapshot": snapshot,
    }


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _run(command, env=None, timeout=20):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout, env=env)
        return result.returncode, result.stdout, result.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, "", str(error)


def _cgroup_paths(cgroup_text, controller, filename):
    """Read visible cgroup roots and all process ancestors without escaping a mount."""
    candidates = []
    for line in (cgroup_text or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = parts[1].split(",")
        version = 2 if parts[1] == "" else 1
        if version == 1 and controller not in controllers:
            continue
        roots = (
            [Path("/sys/fs/cgroup")]
            if version == 2
            else [
                Path("/sys/fs/cgroup") / controller,
                Path("/sys/fs/cgroup") / parts[1],
            ]
        )
        relative = Path(parts[2].lstrip("/"))
        if ".." in relative.parts:
            relative = Path()
        for root in roots:
            current = root / relative
            while True:
                item = (version, current / (filename[1] if version == 2 else filename[0]))
                if item not in candidates:
                    candidates.append(item)
                if current == root:
                    break
                current = current.parent
    return candidates


def _workspace_ssd(workspace):
    code, source, _ = _run(["findmnt", "-n", "-o", "SOURCE", "--target", str(workspace)])
    source = source.strip().split("[", 1)[0]
    if code != 0 or not source.startswith("/dev/"):
        return None
    code, rotational, _ = _run(["lsblk", "-n", "-r", "-o", "ROTA", source])
    values = rotational.split()
    if code != 0 or not values or any(value not in ("0", "1") for value in values):
        return None
    return all(value == "0" for value in values)


def _nvidia_icds():
    paths = set()
    for root in ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"):
        paths.update(Path(root).glob("*.json"))
    for variable in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES"):
        paths.update(Path(path) for path in os.environ.get(variable, "").split(os.pathsep) if path)
    nvidia = []
    for path in sorted(paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            library = data.get("ICD", {}).get("library_path", "")
            if isinstance(library, str) and "nvidia" in library.lower():
                nvidia.append(str(path))
        except (OSError, ValueError, AttributeError):
            continue
    return nvidia


def _runtime_snapshot():
    script = """import importlib.metadata, json
result = {"versions": {}}
for name in ("torch", "isaacsim", "rsl-rl-lib", "newton", "warp-lang"):
    try:
        result["versions"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["versions"][name] = None
try:
    import torch
    result["torch_cuda_available"] = bool(torch.cuda.is_available())
    result["torch_bf16_supported"] = bool(torch.cuda.is_bf16_supported()) if result["torch_cuda_available"] else False
    if result["torch_cuda_available"]:
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["torch_device_name"] = device.name
        result["torch_memory_total_mib"] = device.total_memory // (1024 * 1024)
except Exception as error:
    result["error"] = str(error)
print("CLOUD_PREFLIGHT_JSON=" + json.dumps(result))
"""
    code, output, error = _run([sys.executable, "-c", script], timeout=30)
    for line in reversed(output.splitlines()):
        if line.startswith("CLOUD_PREFLIGHT_JSON="):
            try:
                result = json.loads(line.split("=", 1)[1])
                if isinstance(result, dict):
                    if code != 0:
                        result["error"] = error or f"Runtime subprocess exited {code}"
                    return result
            except ValueError:
                break
    return {"error": error or "Runtime subprocess returned no valid probe JSON"}


def collect_snapshot(workspace, runtime=False):
    """Collect read-only measurements with bounded commands; never launch Kit/Isaac Sim."""
    workspace = Path(workspace).expanduser().resolve()
    snapshot = {
        "system": platform.system(),
        "machine": platform.machine(),
        "workspace": str(workspace),
        "os_release": {},
        "gpus": [],
        "driver_version": None,
        "host_memory_bytes": None,
        "affinity_cpus": None,
        "cgroup_memory_limits": [],
        "cgroup_cpu_limits": [],
        "workspace_free_bytes": None,
        "workspace_ssd": None,
        "container_detected": None,
        "is_root": os.geteuid() == 0 if hasattr(os, "geteuid") else None,
        "cap_sys_module": None,
        "nvidia_module_present": Path("/sys/module/nvidia").exists(),
        "nvidia_driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
        "nvidia_icd_paths": [],
        "collection_errors": [],
    }
    if snapshot["system"] != "Linux":
        snapshot["collection_errors"].append("Linux collection requires a Linux host; no Linux readiness inferred here")
        return snapshot
    release = _read_text("/etc/os-release")
    for line in (release or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            snapshot["os_release"][key] = value.strip('"')
    memory = _read_text("/proc/meminfo")
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", memory or "", re.MULTILINE)
    if match:
        snapshot["host_memory_bytes"] = int(match.group(1)) * 1024
    try:
        snapshot["affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        snapshot["collection_errors"].append("Cannot read CPU affinity")
    cgroups = _read_text("/proc/self/cgroup")
    if not cgroups or not all(re.fullmatch(r"\d+:[^:]*:/.*", line) for line in cgroups.splitlines()):
        snapshot["collection_errors"].append("Cannot read a valid process cgroup hierarchy")
    snapshot["container_detected"] = (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or bool(re.search(r"docker|kubepods|containerd|lxc", cgroups or "", re.IGNORECASE))
    )
    status = _read_text("/proc/self/status")
    match = re.search(r"^CapEff:\s+([0-9a-fA-F]+)$", status or "", re.MULTILINE)
    if match:
        snapshot["cap_sys_module"] = bool(int(match.group(1), 16) & (1 << 16))
    for version, path in _cgroup_paths(cgroups, "memory", ("memory.limit_in_bytes", "memory.max")):
        value = _read_text(path)
        if value is not None:
            snapshot["cgroup_memory_limits"].append({"source": str(path), "version": version, "value": value})
    for version, path in _cgroup_paths(cgroups, "cpu", ("cpu.cfs_quota_us", "cpu.max")):
        value = _read_text(path)
        if value is None:
            continue
        if version == 2:
            values = value.split()
            quota, period = values if len(values) == 2 else (value, None)
        else:
            quota, period = value, _read_text(path.with_name("cpu.cfs_period_us"))
        snapshot["cgroup_cpu_limits"].append(
            {"source": str(path), "version": version, "quota": quota, "period": period}
        )
    try:
        if not workspace.is_dir():
            raise OSError("Workspace must be an existing directory")
        snapshot["workspace_free_bytes"] = shutil.disk_usage(workspace).free
        snapshot["workspace_ssd"] = _workspace_ssd(workspace)
    except OSError as error:
        snapshot["collection_errors"].append("Cannot measure workspace disk: " + str(error))
    code, output, error = _run(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader,nounits"]
    )
    if code != 0:
        snapshot["collection_errors"].append("nvidia-smi failed: " + error.strip())
    else:
        for row in csv.reader(output.splitlines()):
            if len(row) != 3:
                snapshot["collection_errors"].append("Malformed nvidia-smi GPU row")
                continue
            driver, name, memory = (item.strip() for item in row)
            if snapshot["driver_version"] not in (None, driver):
                snapshot["collection_errors"].append("GPU driver versions disagree")
            snapshot["driver_version"] = driver
            snapshot["gpus"].append({"name": name, "memory_total_mib": _integer(memory)})
    icds = _nvidia_icds()
    snapshot["nvidia_icd_paths"] = icds
    vulkan_env = os.environ.copy()
    if icds:
        vulkan_env["VK_DRIVER_FILES"] = os.pathsep.join(icds)
    code, output, error = _run(["vulkaninfo"], env=vulkan_env)
    snapshot["vulkan"] = {
        "available": code == 0,
        "nvidia": bool(re.search(r"NVIDIA|vendorID\s*=\s*0x10de", output, re.IGNORECASE)) if code == 0 else None,
        "extensions": sorted(set(re.findall(r"\bVK_[A-Z0-9]+_[A-Za-z0-9_]+\b", output))) if code == 0 else [],
        "error": error.strip() if code != 0 else None,
    }
    if runtime:
        snapshot["runtime"] = _runtime_snapshot()
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("bootstrap", "runtime"), default="bootstrap")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument(
        "--allow-older-gpu", action="store_true", help="Explicitly allow RTX 3090 class below published minimum"
    )
    parser.add_argument("--amp", action="store_true", help="Require BF16 learner support during runtime checks")
    args = parser.parse_args()
    report = evaluate_snapshot(
        collect_snapshot(args.workspace, runtime=args.phase == "runtime"),
        phase=args.phase,
        allow_older_gpu=args.allow_older_gpu,
        amp=args.amp,
    )
    if args.json_path:
        try:
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            report["errors"].append("Cannot write requested JSON report: " + str(error))
            report["ok"] = False
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
