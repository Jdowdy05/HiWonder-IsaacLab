# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Qualify configured motor latency and terminal/deadline precedence in the real environment."""

import argparse
import json
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-model", choices=("stock", "eka_v6"), default="stock")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg

    bootstrap = SimulationCfg(dt=1 / 240, render_interval=8, device=args.device)
    with launch_simulation(bootstrap, {"device": args.device, "enable_cameras": True, "require_kit": True}):
        import numpy as np
        import torch

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from isaaclab_hiwonder.config import HiWonderEnvConfig
        from isaaclab_hiwonder.sim.env import make_cfg, make_env

        profile = HiWonderEnvConfig(
            robot_model=args.robot_model,
            num_envs=3,
            seed=31,
            actor_mode="state_debug",
            physics_substeps=8,
            episode_seconds=0.1,
        )
        profile = replace(
            profile,
            randomization=replace(
                profile.randomization,
                actuator=replace(profile.randomization.actuator, delay_seconds=(0.5, 0.5)),
            ),
        )
        cfg = make_cfg(
            "peg_insert",
            profile=profile,
            output_directory=args.output_directory / "scene",
            bank_size=96,
            sampling="uniform",
        )
        cfg.sim.device = args.device
        raw = make_env(cfg=cfg)
        env = RslRlVecEnvWrapper(raw)
        try:
            env_ids = torch.arange(3, device=raw.device)
            initial = raw.targets[:, raw.joint_order].clone()
            raw.channel.reset(env_ids, initial)
            requested = initial + 0.1
            delay = torch.full_like(initial, 0.5)
            slack = torch.zeros_like(initial)
            deliveries = [raw.channel.step(requested, delay, slack).clone() for _ in range(18)]
            delivery_tick = next(
                (index for index, value in enumerate(deliveries) if not torch.equal(value, initial)), None
            )
            # Tick0 writes the first command; fifteen30Hz intervals correspond to0.5s.
            expected_delivery_tick = 15
            raw.channel.reset(env_ids, initial)

            original_states = raw.runtime.states
            state_reads = [0]

            def counted_states():
                state_reads[0] += 1
                return original_states()

            raw.runtime.states = counted_states
            env.step(torch.zeros((3, raw.cfg.action_space), device=raw.device))
            ordinary_state_reads = state_reads[0]
            assert np.allclose(raw.runtime.time, raw.step_dt)

            peg = raw.scene[raw.runtime.metadata["task_body_assets"]["peg"]]
            failed_ids = torch.tensor([0, 2], dtype=torch.long, device=raw.device)
            poses = peg.data.root_link_pose_w.torch[failed_ids].clone()
            poses[:, 2] = torch.as_tensor(raw.runtime.offsets[failed_ids.cpu(), 2], device=raw.device) - 0.08
            peg.write_root_pose_to_sim_index(root_pose=poses, env_ids=failed_ids)
            peg.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((2, 6), device=raw.device), env_ids=failed_ids
            )
            raw.episode_length_buf[:2] = raw.max_episode_length - 1
            raw.episode_length_buf[2] = 0
            observations, reward, dones, extras = env.step(torch.zeros((3, raw.cfg.action_space), device=raw.device))
            terminated = raw.reset_terminated.cpu().tolist()
            truncated = raw.reset_time_outs.cpu().tolist()
            bootstrapped = extras["time_outs"].cpu().tolist()
            reset_critic_heights = [
                float(observations["critic"][i, raw.cfg.hiwonder_metadata["critic_layout"][i]["actor/peg"][0] + 2])
                for i in range(3)
            ]
            history_count = int(raw.sampler.counts.sum())
            assert torch.isfinite(reward).all() and dones.cpu().tolist() == [1, 1, 1]
            result = {
                "status": "PASS_RUNTIME_CONTRACTS",
                "motor_delay_seconds": 0.5,
                "first_delivery_tick": delivery_tick,
                "expected_delivery_tick": expected_delivery_tick,
                "terminated": terminated,
                "truncated": truncated,
                "rsl_bootstrapped_timeouts": bootstrapped,
                "case_order": ["failure_at_deadline", "timeout_only", "failure_before_deadline"],
                "uniform_training_completed_history_count": history_count,
                "uniform_training_history_pass": history_count == 3,
                "ordinary_post_step_state_reads": ordinary_state_reads,
                "post_autoreset_critic_peg_heights_m": reset_critic_heights,
                "post_autoreset_critic_fresh": all(value > 0 for value in reset_critic_heights),
                "motor_delay_pass": delivery_tick == expected_delivery_tick,
                "termination_precedence_pass": terminated == [True, False, True]
                and truncated == [False, True, False]
                and bootstrapped == [False, True, False],
            }
            if not all(
                result[name]
                for name in (
                    "motor_delay_pass",
                    "termination_precedence_pass",
                    "uniform_training_history_pass",
                    "post_autoreset_critic_fresh",
                )
            ):
                result["status"] = "FAIL_RUNTIME_CONTRACTS"
            (args.output_directory / "runtime-contracts.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
            if result["status"] != "PASS_RUNTIME_CONTRACTS":
                raise AssertionError("Configured latency or true termination was silently changed")
        finally:
            env.close()


if __name__ == "__main__":
    main()
