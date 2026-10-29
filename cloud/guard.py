# SPDX-License-Identifier: BSD-3-Clause
"""Bound a Linux training process group without changing simulation fidelity."""

import json
import math
import os
import runpy
import signal
import subprocess
import time
from contextlib import suppress
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

GIB = 1024**3


def group_rss(group, proc=Path("/proc")):
    pages = 0
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                pages += int((directory / "statm").read_text().split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return pages * os.sysconf("SC_PAGE_SIZE")


def live_group_members(group):
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                result.append(int(directory.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


@lru_cache(maxsize=1)
def cgroup_path_function():
    return runpy.run_path(str(Path(__file__).with_name("preflight.py")))["_cgroup_paths"]


def available_memory():
    entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    limits = [int(entries["MemAvailable"].split()[0]) * 1024]
    current_cgroup = Path("/proc/self/cgroup").read_text()
    for version, limit_path in cgroup_path_function()(
        current_cgroup, "memory", ("memory.limit_in_bytes", "memory.max")
    ):
        try:
            limit = limit_path.read_text().strip()
            used = int(limit_path.with_name("memory.current" if version == 2 else "memory.usage_in_bytes").read_text())
            if limit not in ("max", "-1") and int(limit) < 2**60:
                limits.append(max(0, int(limit) - used))
        except FileNotFoundError:
            continue
    return min(limits)


def run_guarded(command, *, cwd, log, report, wall_seconds, max_rss_gib, reserve_gib=8, env=None):
    if not sys_linux() or any(not math.isfinite(x) or x <= 0 for x in (wall_seconds, max_rss_gib, reserve_gib)):
        raise ValueError("Guard requires Linux and finite positive resource budgets")
    start, peak, reason, child = time.monotonic(), 0, None, None
    result = dict(
        command=command,
        wall_seconds=wall_seconds,
        max_rss_gib=max_rss_gib,
        started_utc=datetime.now(timezone.utc).isoformat(),
        started_unix_time=time.time(),
        billing_note="Process termination does not stop provider billing; stop the instance separately",
    )
    cleaned = True
    try:
        with Path(log).open("wb") as stream:
            child = subprocess.Popen(
                command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env
            )
            while child.poll() is None:
                peak = max(peak, group_rss(child.pid))
                if peak > max_rss_gib * GIB:
                    reason = "resident_memory_budget"
                elif available_memory() < reserve_gib * GIB:
                    reason = "memory_reserve"
                elif time.monotonic() - start > wall_seconds:
                    reason = "wall_time_budget"
                if reason:
                    break
                time.sleep(0.25)
    except KeyboardInterrupt:
        reason = "operator_interrupt"
    except Exception as error:
        reason = "monitor_or_launch_error"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if child is not None:
            if child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            # Only this owned process group is addressed, including remaining helpers.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + 3
            while live_group_members(child.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if live_group_members(child.pid):
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                time.sleep(0.1)
            cleaned = not live_group_members(child.pid)
            result["child_exit_code"] = child.returncode
        result.update(
            elapsed_seconds=time.monotonic() - start,
            peak_group_rss_bytes=peak,
            stop_reason=reason,
            owned_process_group_cleaned=cleaned,
            complete=child is not None and child.returncode == 0 and reason is None and cleaned,
            scope="Aggregate RSS of owned process group; detached helpers outside the group are not accounted",
        )
        Path(report).write_text(json.dumps(result, indent=2) + "\n")
    if result["complete"]:
        return 0
    if reason == "operator_interrupt":
        return 130
    if reason == "monitor_or_launch_error":
        return 1
    return 124 if reason else (result.get("child_exit_code") or 1)


def sys_linux():
    import sys

    return sys.platform.startswith("linux")
