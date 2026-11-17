"""Selected manager reset semantics; synthetic CPU state, no native solve."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)
from mjlab.utils.noise.noise_cfg import (
    ConstantNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
)
from test_native_backend_ports import _initial_control_case

from envs.commands.motion_tracking.monitor.metrics import MetricsMixin


def _reset_case(tmp_path, *, concatenate=True):
    env, command = _initial_control_case(tmp_path)
    # Extend the existing three-state synthetic source, without another loader.
    for name, value in vars(command.motion_lib).items():
        if torch.is_tensor(value) and value.shape[0] == 3:
            setattr(command.motion_lib, name, value.repeat((4,) + (1,) * (value.ndim - 1)))
        elif isinstance(value, dict):
            for key, tensor in value.items():
                if torch.is_tensor(tensor) and tensor.shape[0] == 3:
                    value[key] = tensor.repeat((4,) + (1,) * (tensor.ndim - 1))
    command.motion_lib._motion_num_frames[:] = 12
    command.motion_lib.native_time_ns = 2**60 + torch.arange(12, dtype=torch.int64) * 20_000_000
    command.motion_lib.robot_joint_pos[:, command.native_command_joint_ids] = (
        torch.arange(12, dtype=torch.float32)[:, None] * .01
    )
    env.observation_manager = ObservationManager({
        "actor": ObservationGroupCfg(
            terms={"source": ObservationTermCfg(
                func=lambda current: current.command_manager.get_term("tracking").ref_joint_pos,
                history_length=3,
                delay_min_lag=1,
                delay_max_lag=2,
                delay_update_period=3,
                noise=NoiseModelWithAdditiveBiasCfg(
                    noise_cfg=ConstantNoiseCfg(bias=.01),
                    bias_noise_cfg=ConstantNoiseCfg(bias=.02),
                ),
            )},
            enable_corruption=True,
            concatenate_terms=concatenate,
        )
    }, env)
    env.reset()
    return env, command


def _source_packet(command, row):
    return {
        "motion_id": command.motion_ids[row].clone(),
        "motion_step": command.motion_steps[row].clone(),
        "q": command.ref_joint_pos[row].clone(),
        "v": command.ref_joint_vel[row].clone(),
        "timer": command.time_left[row].clone(),
        "resamples": command.command_counter[row].clone(),
        "phase": {key: value[row].clone() for key, value in command.reference_phase.items()},
        "metrics": {key: value[row].clone() for key, value in command.metrics.items()},
        "step_metrics": {key: value[row].clone() for key, value in command.native_step_metrics.items()},
        "assist": {name: value[row].clone() for name, value in vars(command).items()
                   if name.startswith("_assist_") and torch.is_tensor(value)},
    }


def _equal(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("concatenate", [True, False])
def test_partial_reset_preserves_unselected_source_clock_and_history(tmp_path, concatenate):
    env, command = _reset_case(tmp_path, concatenate=concatenate)
    action = torch.zeros(2, env.action_manager.total_action_dim)
    for _ in range(4):
        env.step(action)
    assert command.motion_steps.tolist() == [5, 5]
    before = _source_packet(command, 0)
    plant = env.sim.plant
    physical = tuple(value[0].clone() for value in (plant.q, plant.v, plant.control, plant.physics_time()))
    original_observations = env.obs_buf
    observations = deepcopy(original_observations)
    action_history = tuple(value[0].clone() for value in (
        env.action_manager.action, env.action_manager.prev_action, env.action_manager.prev_prev_action))
    history = env.observation_manager._group_obs_term_history_buffer["actor"]["source"]
    delay = env.observation_manager._group_obs_term_delay_buffer["actor"]["source"]
    history_before = (history._pointer, history._num_pushes[0].clone(), history._buffer[:, 0].clone())
    delay_before = (delay._buffer._pointer, delay._buffer._num_pushes[0].clone(),
                    delay._buffer._buffer[:, 0].clone(), delay._step_count[0].clone(),
                    delay.current_lags[0].clone(), delay._phase_offsets[0].clone())
    bias_before = env.observation_manager._group_obs_class_instances["source"]._bias[0].clone()

    returned, _ = env.reset(env_ids=torch.tensor([1]))

    _equal(_source_packet(command, 0), before)
    for actual, expected in zip((plant.q, plant.v, plant.control, plant.physics_time()), physical, strict=True):
        _equal(actual[0], expected)
    for actual, expected in zip((env.action_manager.action, env.action_manager.prev_action,
                                 env.action_manager.prev_prev_action), action_history, strict=True):
        _equal(actual[0], expected)
    assert history._pointer == history_before[0]
    _equal(history._num_pushes[0], history_before[1])
    _equal(history._buffer[:, 0], history_before[2])
    assert delay._buffer._pointer == delay_before[0]
    for actual, expected in zip((delay._buffer._num_pushes[0], delay._buffer._buffer[:, 0],
                                 delay._step_count[0], delay.current_lags[0], delay._phase_offsets[0]),
                                delay_before[1:], strict=True):
        _equal(actual, expected)
    _equal(env.observation_manager._group_obs_class_instances["source"]._bias[0], bias_before)
    selected = returned["actor"] if concatenate else returned["actor"]["source"]
    previous = observations["actor"] if concatenate else observations["actor"]["source"]
    _equal(selected[0], previous[0])
    # The returned new packet must not mutate an earlier caller's observations.
    _equal(original_observations, observations)
    _equal(env.observation_manager._obs_buffer, returned)
    assert command.reference_phase["motion_steps"].tolist() == [4, 0]
    assert command.motion_steps.tolist() == [5, 1]
    assert command.native_step_metrics["reference_time_ns"][1] == 2**60
    assert history._num_pushes[1] == delay._buffer._num_pushes[1] == 1
    assert delay._step_count[1] == 1
    expected = command.ref_joint_pos[1] + .01 + env.observation_manager._group_obs_class_instances["source"]._bias[1]
    _equal(history.buffer[1], expected.expand(3, -1))
    env.step(action)
    assert command.reference_phase["motion_steps"].tolist() == [5, 1]
    assert command.motion_steps.tolist() == [6, 2]


def test_full_reset_and_autoreset_keep_original_global_command_step(tmp_path):
    env, command = _reset_case(tmp_path)
    with patch.object(env.command_manager, "compute", wraps=env.command_manager.compute) as compute, \
            patch.object(env.command_manager, "refresh", wraps=env.command_manager.refresh) as refresh:
        env.reset()
        compute.assert_called_once_with(dt=0.0)
        refresh.assert_not_called()
    assert command.reference_phase["motion_steps"].tolist() == [0, 0]
    assert command.motion_steps.tolist() == [1, 1]
    env.cfg.auto_reset = True
    env.termination_manager._zeros[1] = True
    with patch.object(env.command_manager, "compute", wraps=env.command_manager.compute) as compute, \
            patch.object(env.command_manager, "refresh", wraps=env.command_manager.refresh) as refresh:
        env.step(torch.zeros(2, env.action_manager.total_action_dim))
        compute.assert_called_once_with(dt=env.step_dt)
        refresh.assert_not_called()
    assert command.reference_phase["motion_steps"].tolist() == [1, 0]
    assert command.motion_steps.tolist() == [2, 1]


def test_selected_stock_metrics_preserve_unselected_contact_history():
    class Command(MetricsMixin, SimpleNamespace):
        pass

    command = Command(num_envs=2, device="cpu", metrics={}, finger_names=["index"],
        _side_list=["left"], has_objects=True, motion_ids=torch.zeros(2, dtype=torch.long),
        motion_steps=torch.tensor([4, 0]), motion_lib=SimpleNamespace(hand_sides=["left"]),
        _levels=lambda: (), mano_all_joints_lin_vel_w=lambda side: torch.zeros(2, 1, 3),
        robot_all_joints_lin_vel_w=lambda side: torch.zeros(2, 1, 3))
    for name in ("mano_wrist_trans_w", "robot_wrist_trans_w", "mano_wrist_lin_vel_w",
                 "robot_wrist_lin_vel_w", "mano_wrist_ang_vel_w", "robot_wrist_ang_vel_w",
                 "mano_tip_lin_vel_w", "robot_tip_lin_vel_w", "ref_obj_trans_w", "sim_obj_trans_w",
                 "ref_obj_lin_vel_w", "sim_obj_lin_vel_w", "ref_obj_ang_vel_w", "sim_obj_ang_vel_w"):
        setattr(command, name, torch.zeros(2, 1, 3))
    for name in ("mano_wrist_quat_w", "robot_wrist_quat_w", "ref_obj_quat_w", "sim_obj_quat_w"):
        setattr(command, name, torch.tensor([[[1., 0., 0., 0.]]]).expand(2, 1, 4).clone())
    command.mano_tip_trans_w = command.robot_tip_trans_w = command.ref_contact_trans_w = torch.zeros(2, 1, 1, 3)
    command.mano_tip_lin_vel_w = command.robot_tip_lin_vel_w = torch.zeros(2, 1, 1, 3)
    command.ref_contact_flags = torch.ones(2, 1, 1)
    command._env = SimpleNamespace(scene={
        "l_fingertip_penetration": SimpleNamespace(data=SimpleNamespace(found=torch.ones(2, 1), dist=torch.zeros(2, 1))),
        "l_fingertip_contact": SimpleNamespace(data=SimpleNamespace(force=torch.ones(2, 1, 3))),
    })
    MetricsMixin._init_metrics(command)
    MetricsMixin._update_metrics(command)
    old = {name: value[0].clone() for name, value in command._contact_sum.items()}
    old_count = command._contact_count_ref["l_index"][0].clone()
    MetricsMixin._update_metrics(command, env_ids=torch.tensor([1]))
    for name, expected in old.items():
        _equal(command._contact_sum[name][0], expected)
    _equal(command._contact_count_ref["l_index"][0], old_count)
    assert command._contact_count_ref["l_index"].tolist() == [1., 2.]
