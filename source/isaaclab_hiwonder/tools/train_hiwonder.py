# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train, resume, play and uniformly evaluate HiWonder visual PPO on the task catalog.

Examples (from the Isaac Lab root)::

    uv run --no-sync python source/isaaclab_hiwonder/tools/train_hiwonder.py train --task peg_insert --smoke
    uv run --no-sync python source/isaaclab_hiwonder/tools/train_hiwonder.py train --resume logs/hiwonder/run/final.pt
    uv run --no-sync python source/isaaclab_hiwonder/tools/train_hiwonder.py eval --checkpoint run/final.pt

The camera remains at its configured native fidelity; state_debug is an explicitly
different actor preset. No physical robot connection is made by this tool.
"""

import argparse
import copy
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


def resolve_sampling(mode: str, requested: str | None, saved: dict | None) -> str:
    """Preserve the checkpoint's reset policy when continuing an ablation."""
    if mode != "train":
        return "uniform"
    checkpoint = saved["hiwonder"] if saved else {}
    recorded = checkpoint.get("environment", {}).get("sampling")
    metadata_mode = checkpoint.get("metadata", {}).get("arguments", {}).get("sampling")
    if recorded and metadata_mode and recorded != metadata_mode:
        raise ValueError("Checkpoint sampling mode disagrees with its run metadata")
    previous = recorded or metadata_mode
    if previous and requested and previous != requested:
        raise ValueError(
            "Resume sampling mode differs from checkpoint; use a separate training run for a sampler ablation"
        )
    return requested or previous or "sgs"


def resolve_runner_config(args, profile, saved: dict | None) -> dict:
    """Validate exact continuation and apply explicit learner settings to a fresh optimizer."""
    from isaaclab_hiwonder.rl.config import PPOTrainingConfig

    if saved:
        config = copy.deepcopy(saved["hiwonder"]["runner_config"])
        if config["hiwonder"]["actor_mode"] != profile.actor_mode:
            raise ValueError("Selected actor mode differs from checkpoint")
        if config["hiwonder"].get("frame_stack", 1) != profile.frame_stack:
            raise ValueError("Selected frame stack differs from checkpoint actor architecture")
        if args.resume and config["algorithm"]["gamma"] != profile.reward.shaping_discount:
            raise ValueError("Resume PPO discount differs from saved reward-shaping discount")
        if not args.resume:
            config["seed"] = profile.seed
    else:
        config = PPOTrainingConfig(
            actor_mode=profile.actor_mode,
            frame_stack=profile.frame_stack,
            gamma=profile.reward.shaping_discount,
            seed=profile.seed,
        ).to_runner_config()
    for argument, owner, key, smoke_default in (
        ("rollout_steps", config, "num_steps_per_env", 8),
        ("epochs", config["algorithm"], "num_learning_epochs", 1),
        ("minibatches", config["algorithm"], "num_mini_batches", 2),
    ):
        requested = getattr(args, argument)
        if requested is None and args.smoke:
            requested = smoke_default
        if requested is not None:
            if requested < 1:
                raise ValueError(f"{argument} must be positive")
            if args.resume and owner[key] != requested:
                raise ValueError(f"Resume {argument} differs from checkpoint; use --initialize-from for a fresh run")
            owner[key] = requested
    config["algorithm"]["gamma"] = profile.reward.shaping_discount
    return config


def _arguments():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("train", "eval", "play"), default="train")
    parser.add_argument("--task", help="Catalog name optionally followed by :variant, e.g. cloth_fold:shirt")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--actor-mode", choices=("rgb", "state_debug"))
    parser.add_argument(
        "--frame-stack", type=int, help="Decision-time RGB frames, oldest to newest; capture holds stay explicit"
    )
    parser.add_argument("--reset-families", help="Comma-separated reaching,near_goal,stable_grasp")
    parser.add_argument(
        "--grasp-library", type=Path, help="Physics-validated stable-grasp receipt for the exact robot asset"
    )
    parser.add_argument("--profile", type=Path, help="Strict partial JSON HiWonder profile")
    parser.add_argument("--lighting", choices=("stationary", "white_disco", "rgb_disco"))
    parser.add_argument(
        "--sampling", choices=("sgs", "uniform"), default=None, help="Training sampler; resume keeps its saved mode"
    )
    parser.add_argument("--bank-size", type=int)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--minibatches", type=int)
    parser.add_argument("--smoke", action="store_true", help="Two optimizer iterations, eight-step rollouts, one epoch")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume", type=Path, help="Restore model, optimizer, SGS/environment and random streams"
    )
    checkpoint_group.add_argument(
        "--initialize-from", type=Path, help="Actor/critic weights on a fresh cohort with new optimizer, RNG and SGS"
    )
    checkpoint_group.add_argument(
        "--checkpoint", type=Path, help="Acting weights for play/eval; training reset state is not loaded"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-eval-steps", type=int)
    parser.add_argument("--output", type=Path)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.mode == "train" and args.checkpoint:
        parser.error("Use --resume for training continuation")
    if args.mode != "train" and (not args.checkpoint or args.resume):
        parser.error("play/eval require --checkpoint and do not accept --resume")
    if args.mode != "train" and args.initialize_from:
        parser.error("--initialize-from is for fresh training, not play/eval")
    return args


def main():
    args = _arguments()
    import numpy as np
    import torch
    from isaaclab_hiwonder.config import HiWonderEnvConfig, resolve_backend
    from isaaclab_hiwonder.sim.profiles import load_profile, profile_from_dict
    from isaaclab_hiwonder.tasks import build_task
    from isaaclab_hiwonder.tasks.native_backend import physics_config

    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg

    checkpoint_path = args.resume or args.checkpoint
    weights_path = checkpoint_path or args.initialize_from
    saved = torch.load(weights_path, weights_only=False, map_location="cpu") if weights_path else None
    if saved and saved.get("hiwonder", {}).get("version") != 1:
        raise ValueError("Checkpoint lacks compatible HiWonder environment state")
    args.sampling = resolve_sampling(args.mode, args.sampling, saved if args.resume else None)
    args.run_kind = (
        "exact_resume"
        if args.resume
        else "fresh_cohort_initialization"
        if args.initialize_from
        else "fresh_training"
        if args.mode == "train"
        else "policy_evaluation"
    )
    profile_path = args.profile
    embedded_profile = saved["hiwonder"].get("metadata", {}).get("profile") if saved else None
    if profile_path:
        profile = load_profile(profile_path)
    elif embedded_profile is not None:
        profile = profile_from_dict(embedded_profile)
    elif checkpoint_path and (checkpoint_path.parent / "profile.json").exists():
        profile = load_profile(checkpoint_path.parent / "profile.json")
    else:
        profile = HiWonderEnvConfig(num_envs=4)
    previous_task = saved["hiwonder"]["metadata"].get("arguments", {}).get("task") if saved else None
    task = args.task or (
        profile.task if args.profile or embedded_profile is not None else previous_task or profile.task
    )
    name, _, variant = task.partition(":")
    name = {"peg_in_hole": "peg_insert", "color_cube_stacking": "cube_stack"}.get(name, name)
    task = name + (":" + variant if variant else "")
    task_changed = task != profile.task
    if task_changed:
        # Goal parameters describe the selected task; an explicit task override
        # cannot carry another task's commanded goal into this environment.
        profile = replace(profile, task_parameters={})
    if args.initialize_from and task_changed and args.profile is None:
        # A task transfer keeps the saved sensor/control profile, while the new
        # task chooses its supported solver. Explicit profiles retain their backend constraint.
        profile = replace(profile, backend="auto")
    mode = args.actor_mode or (
        saved["hiwonder"]["runner_config"]["hiwonder"]["actor_mode"] if saved else profile.actor_mode
    )
    seed = args.seed if args.seed is not None else profile.seed + (1_000_003 if args.mode != "train" else 0)
    if args.mode != "train":
        previous = saved["hiwonder"]
        training_seed = previous.get("metadata", {}).get("profile", {}).get("seed")
        if training_seed is None:
            training_seed = previous.get("metadata", {}).get("arguments", {}).get("seed")
        if training_seed is None:
            training_seed = previous["runner_config"]["seed"]
        if seed == training_seed:
            raise ValueError("Held-out evaluation seed must differ from the checkpoint training seed")
    default_stack = (
        saved["hiwonder"]["runner_config"]["hiwonder"].get("frame_stack", 1)
        if args.initialize_from and args.profile is None
        else profile.frame_stack
    )
    profile = replace(
        profile,
        task=task,
        num_envs=args.num_envs if args.num_envs is not None else profile.num_envs,
        seed=seed,
        actor_mode=mode,
        frame_stack=args.frame_stack if args.frame_stack is not None else default_stack,
        reset_families=tuple(args.reset_families.split(","))
        if args.reset_families is not None
        else profile.reset_families,
        grasp_library=str(args.grasp_library) if args.grasp_library is not None else profile.grasp_library,
    )
    args.task, args.num_envs, args.seed, args.actor_mode = task, profile.num_envs, seed, mode
    if args.lighting:
        profile = replace(
            profile,
            randomization=replace(
                profile.randomization, lighting=replace(profile.randomization.lighting, profile=args.lighting)
            ),
        )
    runner_config = resolve_runner_config(args, profile, saved)
    bank_size = args.bank_size or (
        saved["hiwonder"]["environment"]["sampler"]["num_configs"] if checkpoint_path else 32768
    )
    destination = args.output or Path("logs") / "hiwonder" / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    destination.mkdir(parents=True, exist_ok=False)
    definition = build_task(name, variant or None, seed=seed, task_parameters=profile.task_parameters)
    profile = replace(profile, backend=resolve_backend(profile.backend, definition.physics_family))
    # On Windows, importing scene USD/pxr before Kit can select the incompatible standalone DLL.
    # Launch using pure config first, then import/build the USD environment inside its context.
    sim_cfg = SimulationCfg(
        dt=1 / (profile.control_hz * profile.physics_substeps),
        render_interval=profile.physics_substeps,
        physics=physics_config(definition),
        use_fabric=not bool(definition.deformables),
        device=args.device,
    )
    args.enable_cameras = mode == "rgb"
    args.require_kit = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with launch_simulation(sim_cfg, args):
        from isaaclab_hiwonder.rl.evaluate import evaluate_held_out
        from isaaclab_hiwonder.rl.provenance import write_run_provenance
        from isaaclab_hiwonder.rl.runner import HiWonderRunner
        from isaaclab_hiwonder.sim.env import make_cfg, make_env

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        sampling = args.sampling if args.mode == "train" else "uniform"
        split = "train" if args.mode == "train" else "heldout"
        cfg = make_cfg(
            task,
            profile.num_envs,
            seed,
            mode,
            profile=profile,
            output_directory=destination / "scene",
            sampling=sampling,
            split=split,
            bank_size=bank_size,
        )
        cfg.sim.device = args.device
        raw = make_env(task, profile.num_envs, seed, mode, cfg=cfg)
        env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
        try:
            metadata = write_run_provenance(raw, destination, runner_config, vars(args))
            runner = HiWonderRunner(env, runner_config, log_dir=str(destination), device=args.device, metadata=metadata)
            if args.resume:
                runner.load(str(args.resume))
            elif args.initialize_from:
                runner.initialize_weights(str(args.initialize_from))
            if args.mode == "train":
                runner.learn(2 if args.smoke else args.iterations)
                if runner.logger.writer is not None:
                    runner.logger.writer.flush()
                finite = all(torch.isfinite(value).all().item() for value in runner.alg.actor.state_dict().values())
                if not finite:
                    raise RuntimeError("Nonfinite actor parameters after PPO update")
                runner.save(str(destination / "final.pt"))
                p = raw.sampling_probabilities()
                result = dict(
                    mode="train",
                    run_kind=args.run_kind,
                    sampling=raw.cfg.hiwonder_sampling,
                    frame_stack=profile.frame_stack,
                    actor_mode=mode,
                    parameters_finite=finite,
                    completed_iterations=runner.current_learning_iteration,
                    optimizer_steps=max(
                        (int(state["step"]) for state in runner.alg.optimizer.state.values()), default=0
                    ),
                    sgs_entropy=float(-(p * p.log()).sum()),
                    sgs_covered_fraction=float((raw.sampler.counts > 0).float().mean()),
                    checkpoint=str((destination / "final.pt").resolve()),
                )
            else:
                runner.load_policy(str(args.checkpoint))
                result = evaluate_held_out(
                    env, runner.get_inference_policy(), episodes=args.episodes, max_steps=args.max_eval_steps
                )
                result.update(mode=args.mode, actor_mode=mode, seed=seed)
            (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
