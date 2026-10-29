# SPDX-License-Identifier: BSD-3-Clause
"""Plan or execute bounded HiWonder/Eka cloud jobs on the pinned Linux install."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def plan_run(
    workspace,
    stage,
    *,
    task=None,
    precision=None,
    envs=None,
    iterations=None,
    checkpoint=None,
    wall_seconds=None,
    max_rss_gib=24,
    seed=20261029,
):
    if stage not in ("smoke", "pilot", "train", "resume"):
        raise ValueError("Unknown stage")
    for value in (envs, iterations, wall_seconds, max_rss_gib):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Budgets, environment counts and iterations must be positive")
    if precision not in (None, "fp32", "bf16"):
        raise ValueError("Precision must be fp32 or bf16")
    if stage == "resume" and checkpoint is None:
        raise ValueError("Resume requires a checkpoint")
    if task is not None and not re.fullmatch(r"[a-z_]+(?::[a-z_]+)?", task):
        raise ValueError("Unsafe task selector")
    workspace = Path(workspace).resolve()
    installed = workspace / "IsaacLab-HiWonder"
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    selector = task or ("resume" if stage == "resume" else "peg_insert")
    run = workspace / "runs" / selector.replace(":", "_") / name
    job = workspace / "jobs" / name
    command = [
        str(installed / ".venv/bin/python"),
        str(installed / "source/isaaclab_hiwonder/tools/train_hiwonder.py"),
        "train",
    ]
    chosen_precision = precision or (None if stage == "resume" else "fp32")
    profile = None
    chosen_envs = envs if envs is not None else (None if stage == "resume" else 1 if stage == "smoke" else 4)
    if stage == "resume":
        command += ["--resume", str(Path(checkpoint).resolve())]
    else:
        task = task or "peg_insert"
        if task not in json.loads((ROOT / "cloud/task_presets.json").read_text()):
            raise ValueError("Unknown task preset; see cloud/task_presets.json")
        profile = json.loads((ROOT / "source/isaaclab_hiwonder/configs/eka_v6_rgb_spots.json").read_text())
        if stage == "smoke":
            profile["episode_seconds"] = 0.2
        command += [
            "--task",
            task,
            "--profile",
            str(job / "cloud-profile.json"),
            "--sampling",
            "sgs",
            "--seed",
            str(seed),
        ]
        command += ["--bank-size", "64" if stage == "smoke" else "1024" if stage == "pilot" else "32768"]
    if stage == "resume" and task is not None:
        command += ["--task", task]
    if chosen_envs is not None:
        command += ["--num-envs", str(chosen_envs)]
    if chosen_precision is not None:
        command.append("--mixed-precision" if chosen_precision == "bf16" else "--no-mixed-precision")
    command += [
        "--iterations",
        str(iterations or (2 if stage == "smoke" else 20 if stage == "pilot" else 1000)),
        "--output",
        str(run),
    ]
    if stage == "smoke":
        command += ["--smoke", "--rollout-steps", "4", "--minibatches", "2"]
    return dict(
        stage=stage,
        command=command,
        profile=profile,
        output=str(run),
        job_directory=str(job),
        installed=str(installed),
        overlay_payload_sha256=hashlib.sha256((ROOT / "integration/checksums.json").read_bytes()).hexdigest(),
        cloud_entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),
        unlock_status="not_verified_or_applied_by_this_launcher",
        precision=chosen_precision or "checkpoint",
        num_envs=chosen_envs,
        wall_seconds=wall_seconds or (600 if stage == "smoke" else 1800 if stage == "pilot" else 3600),
        max_rss_gib=max_rss_gib,
        required_reserve_gib=8,
        billing_note="Stopping a job does not stop or delete the rented instance",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "pilot", "train", "resume"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--max-rss-gib", type=float, default=24)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--allow-older-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = plan_run(
        args.workspace,
        args.stage,
        task=args.task,
        precision=args.precision,
        envs=args.envs,
        iterations=args.iterations,
        checkpoint=args.checkpoint,
        wall_seconds=args.wall_seconds,
        max_rss_gib=args.max_rss_gib,
        seed=args.seed,
    )
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    if sys.platform != "linux" or os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() != "YES":
        raise SystemExit("Run on the rented Linux host after accepting the Isaac Sim license")
    checker = [
        str(Path(plan["installed"]) / ".venv/bin/python"),
        str(ROOT / "cloud/preflight.py"),
        "--phase",
        "runtime",
        "--workspace",
        str(args.workspace),
    ]
    if args.allow_older_gpu:
        checker.append("--allow-older-gpu")
    if args.precision == "bf16":
        checker.append("--amp")
    subprocess.run(checker, check=True)
    cache_root = args.workspace.resolve() / "cache"
    environment = dict(os.environ)
    for variable, name in (
        ("UV_CACHE_DIR", "uv"),
        ("WARP_CACHE_PATH", "warp"),
        ("XDG_CACHE_HOME", "xdg"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("__GL_SHADER_DISK_CACHE_PATH", "gl"),
    ):
        path = cache_root / name
        path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    destination = Path(plan["job_directory"])
    destination.mkdir(parents=True, exist_ok=False)
    if plan["profile"] is not None:
        (destination / "cloud-profile.json").write_text(json.dumps(plan["profile"], indent=2) + "\n")
    (destination / "cloud-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    from guard import run_guarded

    code = run_guarded(
        plan["command"],
        cwd=plan["installed"],
        log=destination / "console.log",
        report=destination / "cloud-status.json",
        wall_seconds=plan["wall_seconds"],
        max_rss_gib=plan["max_rss_gib"],
        reserve_gib=plan["required_reserve_gib"],
        env=environment,
    )
    if Path(plan["output"]).is_dir():
        for name in ("cloud-plan.json", "cloud-status.json"):
            shutil.copy2(destination / name, Path(plan["output"]) / name)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
