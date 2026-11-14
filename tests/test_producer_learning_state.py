"""CPU checks of real replay and learning-state owners; no Kit or physical claims."""

from __future__ import annotations

import random
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from agents.flashsac import FlashSACAgent, FlashSACConfig
from agents.storage.replay_buffer import ReplayBuffer
from common.utils import capture_rng_state, restore_rng_state
from envs.curriculum.assist import XfrcAssistCurriculum
from runner.off_policy_runner import LEARNING_STATE_FILE, OffPolicyRunner


def _buffer(capacity=64):
    return ReplayBuffer(
        TensorDict({"obs": torch.zeros(1, 2), "aux": torch.zeros(1, 2, dtype=torch.int32)}, batch_size=[1]),
        num_actions=1, n_step=3, gamma=.9, max_length=capacity,
        min_length=8, sample_batch_size=4, device="cpu", obs_storage_dtype=torch.float16,
        aux_groups=("aux",),
    )


def _append(buffer, start, count):
    for frame in range(start, start + count):
        obs = torch.tensor([[frame, 0], [frame, 1]], dtype=torch.float32)
        buffer.add({
            "observation": TensorDict({"obs": obs, "aux": obs.to(torch.int32)}, batch_size=[2]),
            "next_observation": TensorDict({"obs": obs + torch.tensor([1., 0.]),
                "aux": (obs + torch.tensor([1., 0.])).to(torch.int32)}, batch_size=[2]),
            "action": torch.zeros(2, 1), "reward": torch.ones(2),
            "terminated": torch.zeros(2), "truncated": torch.zeros(2),
        })


def test_sparse_save_serializes_occupied_rows_and_keeps_live_capacity(tmp_path):
    buffer = _buffer(capacity=4096)
    _append(buffer, 0, 9)
    path = tmp_path / "replay.pt"
    buffer.save(str(path))
    saved = torch.load(path, map_location="cpu", mmap=True)
    assert saved["capacity"] == buffer._max_length == 4096
    assert saved["num_in_buffer"] == 14
    assert saved["pending_collector_rows"] == 4
    for tensor in [*saved["observation"].values(), *(saved[key] for key in buffer._ROW_ATTRS)]:
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert buffer._observations["obs"].shape == (4096, 2)
    assert saved["observation"]["aux"].dtype == torch.int32
    torch.testing.assert_close(saved["observation"]["obs"], buffer._observations["obs"][:14])


@pytest.mark.parametrize("restart", ["fresh_process", "shared_eval"])
def test_old_frontier_keeps_factual_successors_after_new_physical_fragment(tmp_path, restart):
    buffer = _buffer()
    _append(buffer, 0, 10)
    if restart == "fresh_process":
        path = tmp_path / "replay.pt"
        buffer.save(str(path))
        buffer = _buffer()
        buffer.load(str(path))
    else:
        buffer.end_collection()
    # The old implicit tail is observations5,6,7, with factual successors8,9,10.
    _append(buffer, 100, 9)
    samples = buffer.sample(torch.arange(10, 16))
    expected = torch.tensor([[8, 0], [8, 1], [9, 0], [9, 1], [10, 0], [10, 1]], dtype=torch.float32)
    torch.testing.assert_close(samples["next_observation"]["obs"], expected)
    torch.testing.assert_close(samples["reward"], torch.full((6,), 1 + .9 + .9**2))
    # Ordinary sampling, including the old now-eligible edge, sees the same facts.
    eligible = buffer._num_in_buffer - buffer._n_step * buffer._add_batch_size
    all_rows = buffer.sample(torch.arange(eligible))
    torch.testing.assert_close(all_rows["next_observation"]["obs"][10:16], expected)
    assert all_rows["next_observation"]["obs"][16, 0] == 103
    # Once overwritten by natural wraparound, old edge overrides must disappear.
    _append(buffer, 109, 40)
    overwritten = buffer.sample(torch.arange(10, 16))
    assert bool((overwritten["observation"]["obs"][:, 0] >= 100).all())
    torch.testing.assert_close(overwritten["next_observation"]["obs"][:, 0],
                               overwritten["observation"]["obs"][:, 0] + 3)


def test_timeout_endpoint_survives_full_resume_and_fragment_boundary(tmp_path):
    buffer = _buffer()
    for frame in range(9):
        obs = torch.tensor([[frame if frame <= 6 else frame + 100, 0], [frame, 1]], dtype=torch.float32)
        next_obs = obs + torch.tensor([1., 0.])
        if frame == 6:
            next_obs[0, 0] = 6.25
        buffer.add({
            "observation": TensorDict({"obs": obs, "aux": obs.to(torch.int32)}, batch_size=[2]),
            "next_observation": TensorDict({"obs": next_obs, "aux": next_obs.to(torch.int32)}, batch_size=[2]),
            "action": torch.zeros(2, 1), "reward": torch.ones(2),
            "terminated": torch.zeros(2), "truncated": torch.tensor([frame == 6, False]),
        })
    path = tmp_path / "replay.pt"
    buffer.save(str(path))
    restored = _buffer()
    restored.load(str(path))
    _append(restored, 1000, 9)
    sample = restored.sample(torch.tensor([8, 9]))
    torch.testing.assert_close(sample["next_observation"]["obs"], torch.tensor([[6.25, 0], [7, 1]]))
    torch.testing.assert_close(sample["truncated"], torch.tensor([1., 0.]))


def test_rng_roundtrip_covers_independent_gym_and_learning_generators(tmp_path):
    action_space = gym.spaces.Box(-1, 1, shape=(2, 3), dtype=np.float32)
    action_space.seed(17)
    random.seed(18)
    np.random.seed(19)
    torch.manual_seed(20)
    path = tmp_path / "rng.pt"
    torch.save(capture_rng_state(action_space), path)
    expected = (random.random(), np.random.randn(3), torch.randn(3), action_space.sample())
    random.seed(200)
    np.random.seed(201)
    torch.manual_seed(202)
    action_space.seed(203)
    restore_rng_state(torch.load(path, weights_only=False), action_space)
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.randn(3), expected[1])
    torch.testing.assert_close(torch.randn(3), expected[2], rtol=0, atol=0)
    np.testing.assert_array_equal(action_space.sample(), expected[3])


def _curriculum():
    command = SimpleNamespace()
    curriculum = XfrcAssistCurriculum(
        command=command, num_envs=2, max_episode_length=100, num_steps_per_env=4,
        kp_init=80, kv_init=5, force_range=50, wait_epochs=0, deque_len=2,
        reward_terms={"task": ("task",)}, rew_thresholds={"task": .6},
        upper_ratios={"kp": .9, "kv": .9, "fr": 1},
        lower_ratios={"kp": 1, "kv": 1, "fr": 1},
        dialback_completion_thres=.5, dialback_min_epochs=500,
        dialback_ratios={"kp": .98, "kv": .98, "fr": .98},
        decay_cooldown_epochs=1, kp_zero_thres=.05, kp_lower_zero_thres=.1,
        completion_thres=.65, device="cpu",
    )
    env = SimpleNamespace(
        common_step_counter=0, episode_length_buf=torch.zeros(2, dtype=torch.long),
        reward_manager=SimpleNamespace(active_terms=["task"],
            _term_cfgs=[SimpleNamespace(weight=1)], _step_reward=torch.ones(2, 1)),
        termination_manager=SimpleNamespace(time_outs=torch.zeros(2, dtype=torch.bool),
                                            terminated=torch.zeros(2, dtype=torch.bool)),
    )
    return curriculum, env


def _curriculum_step(curriculum, env, step):
    env.common_step_counter = step
    env.episode_length_buf += 1
    curriculum.on_step(env)


def test_curriculum_restores_histories_and_partial_epoch_on_fresh_control_clock():
    original, env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(original, env, step)
    saved = original.state_dict()
    assert saved["control_step"] == 6 and saved["epoch_steps"] == 2
    restored, fresh_env = _curriculum()
    restored.load_state_dict(saved, fresh_env.common_step_counter)
    assert restored.state_dict() == saved
    assert restored._prev_ep_len is None
    for original_step, fresh_step in [(7, 1), (8, 2), (9, 3)]:
        _curriculum_step(original, env, original_step)
        _curriculum_step(restored, fresh_env, fresh_step)
    # Physical episode lengths differ; global history, phase, gains and decisions agree.
    assert restored.state_dict() == original.state_dict()


def test_shared_env_runner_reset_does_not_count_old_physical_episode_as_completed():
    curriculum, env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(curriculum, env, step)
    global_state = curriculum.state_dict()
    assert curriculum._prev_ep_len.tolist() == [6, 6]
    buffer = _buffer()
    _append(buffer, 0, 10)

    def reset_worlds():
        env.episode_length_buf.zero_()
        return np.zeros((2, 2), dtype=np.float32), {}

    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner._agent = SimpleNamespace(reset_collection=buffer.end_collection)
    runner._train_cmd = SimpleNamespace(_xfrc_curr_ctrl=curriculum)
    runner._train_env = SimpleNamespace(reset=reset_worlds)
    runner._obs_aux = None
    runner._reset_train_env()
    assert not buffer._n_step_transitions
    assert buffer._boundary_next_observations
    assert curriculum.state_dict() == global_state
    assert curriculum._prev_ep_len is None
    _curriculum_step(curriculum, env, 7)
    assert not curriculum._epoch_achieved
    assert curriculum._epoch_steps == global_state["epoch_steps"] + 1


def _agent():
    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"),
                               version_base=None):
        cfg = compose(config_name="flashdexretarget", overrides=[
            "commands.motion_file=synthetic-not-read.pt", "num_envs=2", "device=cpu",
            "runner.agent.device_type=cpu", "runner.agent.buffer_device_type=cpu",
            "runner.agent.buffer_max_length=64", "runner.agent.buffer_min_length=8",
            "runner.agent.sample_batch_size=4", "runner.agent.n_step=3",
            "runner.agent.actor_hidden_dim=8", "runner.agent.critic_hidden_dim=16",
            "runner.agent.critic_num_bins=11", "runner.agent.use_amp=false",
            "runner.agent.use_compile=false", "runner.agent.buffer_compact_obs=false",
            "runner.agent.asymmetric_observation=false", "runner.agent.traj_encoder.enable=false",
        ])
        fields = OmegaConf.to_container(cfg.runner.agent, resolve=True)
    fields.pop("class_name")
    return FlashSACAgent(
        gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        {"actor_observation_size": (6,), "reward_heads": ["right", "left"],
         "action_sides": {"right": [0, 1], "left": [2, 3]}},
        FlashSACConfig(**fields),
    )


def _feed_agent(agent):
    for frame in range(9):
        agent.observe({
            "observation": np.full((2, 6), frame / 100, dtype=np.float32),
            "next_observation": np.full((2, 6), (frame + 1) / 100, dtype=np.float32),
            "action": np.zeros((2, 4), dtype=np.float32),
            "reward": np.full((2, 2), .01, dtype=np.float32),
            "terminated": np.zeros(2, dtype=bool), "truncated": np.zeros(2, dtype=bool),
        })


def _tree_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _tree_equal(a, b)
    else:
        assert left == right


def _runner(agent, action_space, curriculum, path):
    # Real save/restore methods with small CPU owners; no alternate training loop.
    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner._agent = agent
    runner._train_env = SimpleNamespace(action_space=action_space,
                                      unwrapped=SimpleNamespace(common_step_counter=curriculum._last_control_step))
    runner._train_cmd = SimpleNamespace(_xfrc_curr_ctrl=curriculum)
    runner._n_envs, runner._step_offset = 2, 100
    runner._nonfinite_steps = 0
    runner._wall_offset, runner._loop_start = 3., time.perf_counter()
    runner._save_path_base = path
    return runner


def test_real_sac_learning_and_full_owner_restore_keep_schedules_rng_and_fresh_episode(tmp_path):
    original = _agent()
    _feed_agent(original)
    for _ in range(2):
        info = original.update()
        assert info and all(np.isfinite(value) for value in info.values())
    assert original._alg.update_step == 2
    curriculum, env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(curriculum, env, step)
    space = gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32)
    space.seed(21)
    checkpoint = tmp_path / "full"
    checkpoint.mkdir()
    original.save(str(checkpoint))
    original.save_buffer(str(checkpoint))
    runner = _runner(original, space, curriculum, tmp_path / "first")
    runner._save_learning_state(checkpoint, 100)
    state = torch.load(checkpoint / LEARNING_STATE_FILE, weights_only=False)
    restore_rng_state(state["rng"], space)
    expected_draws = (torch.randn(3), space.sample())

    restored = _agent()
    restored.load(str(checkpoint))
    restored.load_buffer(str(checkpoint))
    fresh_curriculum, _ = _curriculum()
    fresh_space = gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32)
    fresh = _runner(restored, fresh_space, fresh_curriculum, tmp_path / "fresh")
    saved_algorithm = torch.load(checkpoint / "flashsac.pt")
    before_reset = restored._alg.save()
    _tree_equal(saved_algorithm, before_reset)
    fresh._restore_learning_state(checkpoint)
    actual_restored = torch.load(fresh._save_path_base / "learning-restored.pt", weights_only=False)
    _tree_equal(actual_restored["rng"], state["rng"])
    assert actual_restored["curriculum"] == state["curriculum"]
    torch.testing.assert_close(torch.randn(3), expected_draws[0], rtol=0, atol=0)
    np.testing.assert_array_equal(fresh_space.sample(), expected_draws[1])
    assert fresh_curriculum.state_dict() == state["curriculum"]
    assert restored._alg._cached_noise is None
    assert not restored._alg.replay_buffer._n_step_transitions
    reward_state = restored._alg.reward_normalizer.state_dict()
    assert not bool(reward_state["G_r"].any())
    for key in reward_state.keys() - {"G_r"}:
        _tree_equal(reward_state[key], saved_algorithm["reward_normalizer_state_dict"][key])
    for key in ["actor_optimizer_state_dict", "critic_optimizer_state_dict",
                "temperature_optimizer_state_dict", "actor_scheduler_state_dict",
                "critic_scheduler_state_dict", "temperature_scheduler_state_dict",
                "grad_scaler_state_dict", "update_step"]:
        _tree_equal(restored._alg.save()[key], saved_algorithm[key])
    restored.update()
    assert restored._alg.update_step == 3
    assert restored._alg.critic_scheduler.last_epoch == original._alg.critic_scheduler.last_epoch + 1
