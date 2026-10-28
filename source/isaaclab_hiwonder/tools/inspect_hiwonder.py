# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect real wrist RGB and policy observations with live physical lighting controls.

Run from the Isaac Lab root::

    uv run --no-sync python source/isaaclab_hiwonder/tools/inspect_hiwonder.py --viz kit

Open the printed loopback URL for controls and lossless full-resolution PNGs.
Omit --viz kit for browser-only inspection; the configured camera still renders.
PNG export cadence is independent of simulation, sensor capture and native display.
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path


def _arguments():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="peg_insert")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--profile", type=Path, default=Path(__file__).resolve().parents[1] / "configs/rgb_spots.json")
    parser.add_argument("--output", type=Path, help="New inspection directory; must not already exist")
    parser.add_argument("--port", type=int, default=8769, help="127.0.0.1 HTTP port; 0 chooses a free port")
    parser.add_argument(
        "--export-hz", type=float, default=2.0, help="CPU PNG export only; leaves camera capture unchanged"
    )
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or not 0 <= args.port <= 65535:
        parser.error("num-envs must be positive and port must be between 0 and 65535")
    if not math.isfinite(args.export_hz) or args.export_hz <= 0:
        parser.error("export-hz must be finite and positive")
    return args


class _ImagePanel:
    """Retain double-buffered RGBA GPU storage for the Kit provider lifetime."""

    def __init__(self, url, width, height, device):
        import torch

        import omni.ui as ui

        self.providers = [ui.ByteImageProvider(), ui.ByteImageProvider()]
        self.rgba = [torch.full((2, height, width, 4), 255, dtype=torch.uint8, device=device) for _ in range(2)]
        self.policy_rgb = torch.empty((height, width, 3), dtype=torch.float32, device=device)
        self.slot = 0
        self.gpu = str(device).startswith("cuda")
        self.cpu_buffers = [None, None]
        self.window = ui.Window("HiWonder | full-resolution raw RGB + actual policy RGB", width=1320, height=600)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label(f"Lighting, rooms, reset, pause and stop controls: {url}", height=26)
                self.status = ui.Label("Starting full-resolution camera display...", height=24)
                with ui.HStack(height=22):
                    ui.Label("RAW WRIST RGB")
                    ui.Label("ACTUAL POLICY RGB | ISP + smudge + image randomization")
                with ui.HStack():
                    ui.ImageWithProvider(self.providers[0], width=width, height=height)
                    ui.ImageWithProvider(self.providers[1], width=width, height=height)
        viewport = ui.Workspace.get_window("Viewport")
        if viewport:
            self.window.dock_in(viewport, ui.DockPosition.BOTTOM, 0.58)

    def update(self, raw, policy, text):
        self.slot = 1 - self.slot
        self.rgba[0][self.slot, ..., :3].copy_(raw)
        self.policy_rgb.copy_(policy)
        self.policy_rgb.clamp_(0, 1).mul_(255).round_()
        self.rgba[1][self.slot, ..., :3].copy_(self.policy_rgb)
        for index, provider in enumerate(self.providers):
            image = self.rgba[index][self.slot]
            if self.gpu:
                try:
                    import omni.gpu_foundation_factory as gf

                    provider.set_bytes_data_from_gpu(
                        int(image.data_ptr()), [int(image.shape[1]), int(image.shape[0])], gf.TextureFormat.RGBA8_UNORM
                    )
                    continue
                except Exception as error:
                    print(f"INSPECTION_GPU_UPLOAD_FALLBACK: {error}", flush=True)
                    self.gpu = False
            self.cpu_buffers[index] = image.detach().cpu().numpy()
            provider.set_bytes_data(self.cpu_buffers[index].reshape(-1).data, [image.shape[1], image.shape[0]])
        self.status.text = text + (" | GPU upload" if self.gpu else " | CPU upload fallback")


def main():
    args = _arguments()
    import numpy as np
    import torch
    from isaaclab_hiwonder.inspection import InspectionServer
    from isaaclab_hiwonder.sim.profiles import load_profile
    from isaaclab_hiwonder.tasks import build_task
    from isaaclab_hiwonder.tasks.native_backend import physics_config

    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg

    profile = replace(
        load_profile(args.profile), task=args.task, num_envs=args.num_envs, seed=args.seed, actor_mode="rgb"
    )
    destination = args.output or Path("logs/hiwonder/inspection") / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "profile.json").write_text(json.dumps(asdict(profile), indent=2) + "\n", encoding="utf-8")
    name, _, variant = args.task.partition(":")
    definition = build_task(name, variant or None, seed=profile.seed, task_parameters=profile.task_parameters)
    bootstrap = SimulationCfg(
        dt=1 / (profile.control_hz * profile.physics_substeps),
        render_interval=profile.physics_substeps,
        physics=physics_config(definition),
        use_fabric=not bool(definition.deformables),
        device=args.device,
    )
    args.enable_cameras = True
    args.require_kit = True
    random.seed(profile.seed)
    np.random.seed(profile.seed)
    torch.manual_seed(profile.seed)
    # Kit must start before importing USD/pxr-backed environment code on Windows.
    with launch_simulation(bootstrap, args):
        from isaaclab_hiwonder.sim.env import make_cfg, make_env

        cfg = make_cfg(
            profile=profile, output_directory=destination / "scene", sampling="uniform", split="heldout", bank_size=128
        )
        cfg.sim.device = args.device
        env = make_env(cfg=cfg)
        try:
            with InspectionServer(destination, num_envs=profile.num_envs, port=args.port) as server:
                print(f"INSPECTION_URL {server.url}\nINSPECTION_OUTPUT {destination}", flush=True)
                _inspect(env, profile, args, server)
        finally:
            env.close()


def _inspect(env, profile, args, server):
    import torch

    observations, _ = env.reset()
    zero = torch.zeros((profile.num_envs, 6), device=env.device)
    camera = env.scene["wrist_camera"]
    origins = env.scene.env_origins.detach().cpu().numpy()
    # Structural heights are fixed for this launch. No per-frame randomization snapshots.
    heights = env.randomization.snapshot()["table"]["height"].detach().cpu().tolist()

    def select_room(index):
        origin = origins[index]
        height = heights[index]
        env.sim.set_camera_view(tuple(origin + (1.05, 1.05, height + 0.80)), tuple(origin + (0.10, 0.0, height + 0.16)))

    panel = None
    if args.visualizer and "kit" in args.visualizer and not args.headless:
        panel = _ImagePanel(server.url, profile.camera.width, profile.camera.height, env.device)
    room, paused, stopped, step = 0, False, False, 0
    select_room(room)
    lighting = env.set_lighting_preview(None)
    error = None
    exported_at = -math.inf
    measured_at = time.monotonic()
    frames = 0
    wall_hz = 0.0
    status = {}
    print("INSPECTION_INITIALIZED", flush=True)
    while env.sim.is_running() and not stopped:
        started = time.monotonic()
        for command in server.drain_commands():
            action = command["action"]
            try:
                if action in ("lighting", "clear"):
                    lighting = env.set_lighting_preview(command["controls"] if action == "lighting" else None)
                elif action == "room":
                    room = command["index"]
                    select_room(room)
                elif action == "reset":
                    observations, _ = env.reset()
                elif action == "pause":
                    paused = command["paused"]
                elif action == "stop":
                    stopped = True
                error = None
            except ValueError as exception:
                error = str(exception)
        if stopped:
            break
        if paused:
            env.sim.render()
            # Refresh only the raw render. Keep the policy observation and its
            # randomization/history state at the last actual environment step.
            camera.update(1 / profile.camera.fps, force_recompute=True)
        else:
            with torch.inference_mode():
                observations, _, _, _, _ = env.step(zero)
            step += 1
        frames += 1
        now = time.monotonic()
        if now - measured_at >= 1.0:
            wall_hz = frames / (now - measured_at)
            measured_at, frames = now, 0
        mode = "paused" if paused else "zero_action_hold"
        text = (
            f"Room {room} | step {step} | {mode} | wall display {wall_hz:.1f} Hz | "
            f"capture configured {profile.camera.fps:g} Hz"
        )
        if paused:
            text += " | raw: render-only preview | policy: last simulation observation"
        raw = camera.data.output["rgb"].torch[room, ..., :3].detach()
        policy = observations["policy_image"][room, -3:].detach().permute(1, 2, 0)
        if panel is not None:
            with torch.inference_mode():
                panel.update(raw, policy, text)
        status = {
            "mode": mode,
            "room": room,
            "num_envs": profile.num_envs,
            "seed": profile.seed,
            "step": step,
            "paused": paused,
            "raw_image_mode": "render_only_preview" if paused else "simulation_capture",
            "policy_image_mode": "last_simulation_observation" if paused else "current_simulation_observation",
            "display_hz": wall_hz if panel is not None else 0.0,
            "control_wall_hz": wall_hz,
            "native_upload": "gpu" if panel is not None and panel.gpu else "cpu" if panel is not None else "disabled",
            "control_hz": profile.control_hz,
            "physics_substeps": profile.physics_substeps,
            "physics_dt": env.physics_dt,
            "camera": asdict(profile.camera),
            "lighting": lighting,
            "source_profile": str(args.profile.resolve()),
            "export_requested_hz": args.export_hz,
            "split": "heldout",
            "sampling": "uniform",
            "error": error,
            "wall_time_unix": time.time(),
        }
        server.publish_status(status)
        if now - exported_at >= 1 / args.export_hz:
            # Copy only for browser/export cadence. Use the already-transformed policy observation.
            with torch.inference_mode():
                policy_bytes = policy.clamp(0, 1).mul(255).round().to(torch.uint8)
                server.submit_frames(raw.cpu().numpy(), policy_bytes.cpu().numpy())
            exported_at = now
        time.sleep(max(0.0, 1 / profile.control_hz - (time.monotonic() - started)))
    server.publish_status({**status, "mode": "stopped", "paused": paused})


if __name__ == "__main__":
    main()
