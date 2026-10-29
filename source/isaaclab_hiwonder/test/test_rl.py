# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real RSL-RL model/storage/optimizer contracts on a small deterministic vector system."""

import pytest
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from isaaclab_hiwonder.randomization import RandomizationManager
from isaaclab_hiwonder.rl.config import PPOTrainingConfig
from isaaclab_hiwonder.rl.runner import HiWonderRunner
from isaaclab_hiwonder.sgs import SuccessGuidedSampler


class _ControlSystem(VecEnv):
    """Small real dynamics fixture; no simulator or camera correctness claims."""

    def __init__(self, frame_stack=1, num_actions=6):
        self.num_envs = 2
        self.num_actions = num_actions
        self.device = "cpu"
        self.max_episode_length = 4
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.cfg = {}
        self.position = torch.zeros(2, num_actions)
        self.sampler = SuccessGuidedSampler(4, seed=11)
        self.randomization = RandomizationManager(2, seed=11)
        self.config_ids = self.sampler.sample(2)
        self.frame_stack = frame_stack
        from isaaclab_hiwonder.sim.history import FrameHistory

        self.history = FrameHistory(2, frame_stack, 3, 32, 40)
        self.reset_family_ids = torch.zeros(2, dtype=torch.long)
        self.reset_family_names = ("reaching", "near_goal", "stable_grasp")

    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 40), indexing="ij")
        image = torch.stack(
            (
                xx.expand(2, -1, -1),
                yy.expand(2, -1, -1),
                torch.sigmoid(self.position[:, 0, None, None]).expand(-1, 32, 40),
            ),
            1,
        )
        image = self.history.push(image)
        return TensorDict(
            dict(
                policy_image=image,
                proprio=self.position.clone(),
                command=torch.ones(2, 3),
                critic=torch.cat((self.position, self.position.square()), -1),
            ),
            batch_size=[2],
        )

    def step(self, actions):
        self.position = 0.9 * self.position + 0.1 * actions.tanh()
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        rewards = 1 - self.position.square().sum(-1)
        if dones.any():
            self.sampler.update(self.config_ids[dones], rewards[dones] > 0.9)
            self.config_ids[dones] = self.sampler.sample(int(dones.sum()))
            self.episode_length_buf[dones] = 0
        return self.get_observations(), rewards, dones.long(), dict(episode_success=rewards > 0.9)

    def state_dict(self):
        return dict(
            sampler=self.sampler.state_dict(),
            randomization=self.randomization.state_dict(),
            position=self.position.clone(),
            config_ids=self.config_ids.clone(),
            episode_length_buf=self.episode_length_buf.clone(),
        )

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state["sampler"])
        self.randomization.load_state_dict(state["randomization"])
        self.position = state["position"].clone()
        self.config_ids = state["config_ids"].clone()
        self.episode_length_buf = state["episode_length_buf"].clone()


def _runner(env, use_mixed_precision=False):
    config = PPOTrainingConfig(
        num_steps_per_env=4,
        num_learning_epochs=1,
        num_mini_batches=2,
        hidden_dims=(32, 16),
        frame_stack=env.frame_stack,
        use_mixed_precision=use_mixed_precision,
    )
    return HiWonderRunner(env, config.to_runner_config(), device="cpu")


@pytest.mark.parametrize("frame_stack", [1, 2])
def test_visual_actor_excludes_privileged_state_and_rollout_replays_exact_collected_images(frame_stack):
    torch.manual_seed(4)
    env = _ControlSystem(frame_stack)
    runner = _runner(env)
    obs = env.get_observations()
    original = obs.clone()
    altered = obs.clone()
    altered["critic"] += 500
    actor = runner.get_inference_policy()
    torch.testing.assert_close(actor(obs), actor(altered), rtol=0, atol=0)
    altered["policy_image"] = 1 - altered["policy_image"]
    assert not torch.equal(actor(obs), actor(altered))
    with torch.inference_mode():
        actions = runner.alg.act(obs)
        next_obs, rewards, dones, extras = env.step(actions)
        runner.alg.process_env_step(next_obs, rewards, dones, extras)
    obs["policy_image"].zero_()
    stored = runner.alg.storage.observations[0]
    assert torch.equal(stored["policy_image"], original["policy_image"])
    actor(stored, stochastic_output=True)
    torch.testing.assert_close(
        actor.get_output_log_prob(actions), runner.alg.storage.actions_log_prob[0, :, 0], rtol=1e-5, atol=1e-6
    )


def test_real_ppo_update_and_checkpoint_restore_sampler_optimizer_and_rng(tmp_path):
    torch.manual_seed(18)
    runner = _runner(_ControlSystem())
    before = {key: value.clone() for key, value in runner.alg.actor.state_dict().items()}
    runner.learn(1)
    assert any(not torch.equal(value, before[key]) for key, value in runner.alg.actor.state_dict().items())
    assert all(torch.isfinite(value).all() for value in runner.alg.actor.state_dict().values())
    assert runner.current_learning_iteration == 1
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    expected_ids = runner.env.sampler.sample(30)
    expected_noise = torch.rand(5)
    expected_position = runner.env.position.clone()
    restored = _runner(_ControlSystem())
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == 1
    assert torch.equal(restored.env.sampler.sample(30), expected_ids)
    assert torch.equal(torch.rand(5), expected_noise)
    assert torch.equal(restored.env.position, expected_position)
    assert max(int(value["step"]) for value in restored.alg.optimizer.state.values()) > 0
    held_out = _runner(_ControlSystem())
    held_out.env.position += 0.8
    preserved_position = held_out.env.position.clone()
    held_out.load_policy(str(checkpoint))
    assert torch.equal(held_out.env.position, preserved_position)
    assert held_out.current_learning_iteration == 0 and not held_out.alg.optimizer.state
    initialized = _runner(_ControlSystem())
    rng_before = torch.get_rng_state().clone()
    initialized.initialize_weights(str(checkpoint))
    assert initialized.current_learning_iteration == 0 and not initialized.alg.optimizer.state
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert initialized.env.sampler.state_dict()["counts"].sum() == 0
    for source, target in ((runner.alg.actor, initialized.alg.actor), (runner.alg.critic, initialized.alg.critic)):
        for name, value in source.state_dict().items():
            assert torch.equal(value, target.state_dict()[name])


@pytest.mark.parametrize("use_mixed_precision", [False, True])
def test_nine_action_learner_updates_and_policy_restore_rejects_stock_model(tmp_path, use_mixed_precision):
    from types import SimpleNamespace

    env = _ControlSystem(num_actions=9)
    env.profile = SimpleNamespace(robot_model="eka_v6")
    runner = _runner(env, use_mixed_precision=use_mixed_precision)
    runner.metadata = {"profile": {"robot_model": "eka_v6"}}
    before = {key: value.clone() for key, value in runner.alg.actor.state_dict().items()}
    runner.learn(1)
    assert any(not torch.equal(value, before[key]) for key, value in runner.alg.actor.state_dict().items())
    assert all(torch.isfinite(value).all() for value in runner.alg.actor.state_dict().values())
    assert all(value.dtype == torch.float32 for value in runner.alg.actor.parameters())
    assert all(value.dtype == torch.float32 for value in runner.alg.critic.parameters())
    assert env.position.dtype == torch.float32
    assert env.get_observations()["policy_image"].dtype == torch.float32
    assert runner.alg.use_mixed_precision is use_mixed_precision
    checkpoint = tmp_path / "eka.pt"
    runner.save(str(checkpoint))
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["hiwonder"]["runner_config"]["algorithm"]["use_mixed_precision"] is use_mixed_precision
    expected = "bf16_autocast" if use_mixed_precision else "fp32"
    assert saved["hiwonder"]["runner_config"]["hiwonder"]["learner_precision"] == expected
    restored_env = _ControlSystem(num_actions=9)
    restored_env.profile = SimpleNamespace(robot_model="eka_v6")
    restored = _runner(restored_env, use_mixed_precision=use_mixed_precision)
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == 1
    assert max(int(value["step"]) for value in restored.alg.optimizer.state.values()) > 0
    assert restored.get_inference_policy()(restored_env.get_observations()).shape == (2, 9)
    with pytest.raises(ValueError, match="robot model"):
        _runner(_ControlSystem()).load_policy(str(checkpoint))


def test_ppo_precision_defaults_to_fp32_and_opt_in_keeps_native_camera_configuration():
    from isaaclab_hiwonder.config import HiWonderEnvConfig

    default = PPOTrainingConfig().to_runner_config()
    assert default["algorithm"]["use_mixed_precision"] is False
    assert default["hiwonder"].get("learner_precision") == "fp32"
    mixed = PPOTrainingConfig(use_mixed_precision=True).to_runner_config()
    assert mixed["algorithm"]["use_mixed_precision"] is True
    assert mixed["hiwonder"]["learner_precision"] == "bf16_autocast"
    assert mixed["hiwonder"]["image_resize"] is None
    assert (HiWonderEnvConfig().camera.width, HiWonderEnvConfig().camera.height) == (640, 480)


@pytest.mark.parametrize("invalid", [None, "bf16", 1])
def test_ppo_precision_requires_a_boolean(invalid):
    with pytest.raises(ValueError, match="use_mixed_precision.*bool"):
        PPOTrainingConfig(use_mixed_precision=invalid)


def test_uniform_held_out_evaluation_uses_terminal_outcomes_and_captured_reset_ids():
    from types import SimpleNamespace

    from isaaclab_hiwonder.rl.evaluate import evaluate_held_out

    env = _ControlSystem()
    runner = _runner(env)
    policy = runner.get_inference_policy()
    env.cfg = SimpleNamespace(hiwonder_sampling="sgs", hiwonder_split="train")
    import pytest

    with pytest.raises(ValueError, match="uniform"):
        evaluate_held_out(env, policy, episodes=3)
    env.cfg = SimpleNamespace(hiwonder_sampling="uniform", hiwonder_split="heldout")
    result = evaluate_held_out(env, policy, episodes=3)
    assert result["completed_episodes"] == 3
    assert result["success_rate"] == 1
    assert not result["budget_exhausted"]
    assert sum(value["episodes"] for value in result["by_configuration"].values()) == 3


@pytest.fixture
def cohort_evaluation():
    from types import SimpleNamespace

    from isaaclab_hiwonder.rl.evaluate import evaluate_held_out

    class Cohorts:
        num_envs, device, max_episode_length = 2, "cpu", 10
        cfg = SimpleNamespace(hiwonder_sampling="uniform", hiwonder_split="heldout")
        config_ids = torch.tensor([0, 1])
        reset_family_ids = torch.tensor([0, 1])
        reset_family_names = ("reaching", "near_goal")

        def __init__(self, bank_size):
            self.elapsed = torch.zeros(2, dtype=torch.long)
            self.sampler = SimpleNamespace(num_configs=bank_size)

        @property
        def unwrapped(self):
            return self

        def get_observations(self):
            return torch.zeros(2, 1)

        def step(self, actions):
            self.elapsed += 1
            done = self.elapsed >= torch.tensor([1, 10])
            self.elapsed[done] = 0
            return (
                self.get_observations(),
                torch.tensor([0.0, 1.0]),
                done.long(),
                {"episode_success": torch.tensor([False, True])},
            )

    class Policy:
        def eval(self):
            return self

        def __call__(self, observations):
            return torch.zeros(2, 6)

    return lambda bank_size=2, **kwargs: evaluate_held_out(Cohorts(bank_size), Policy(), **kwargs)


@pytest.mark.parametrize("bank_size, expected", [(2, 0.5), (3, 1 / 3)])
def test_uniform_evaluation_balances_different_cohort_durations_and_bank_slice_sizes(
    cohort_evaluation, bank_size, expected
):
    result = cohort_evaluation(bank_size, episodes=20)
    assert result["completed_episodes"] == 20
    assert result["success_rate"] == pytest.approx(expected)
    assert result["by_configuration"]["1"]["episodes"] >= 6


def test_incomplete_cohort_quotas_do_not_publish_a_uniform_score(cohort_evaluation):
    partial = cohort_evaluation(episodes=20, max_steps=5)
    assert partial["completed_episodes"] > 0 and partial["budget_exhausted"]
    assert partial["success_rate"] is None and partial["mean_return"] is None
    assert partial["cohort_coverage"] == 0.5
    insufficient = cohort_evaluation(episodes=1)
    assert insufficient["completed_episodes"] == 1 and not insufficient["budget_exhausted"]
    assert insufficient["success_rate"] is None and insufficient["cohort_coverage"] == 0.5
