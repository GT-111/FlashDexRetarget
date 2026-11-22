"""Shared manager-based RL env: mirrors mjlab 1.3.0 step/reset semantics
plus the mjlab_ext deltas (auto_reset single pass, capture_final_obs)."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.managers.action_manager import ActionManager, ActionTermCfg
from mjlab.managers.command_manager import CommandManager, CommandTermCfg
from mjlab.managers.curriculum_manager import (
    CurriculumManager,
    CurriculumTermCfg,
    NullCurriculumManager,
)
from mjlab.managers.event_manager import EventManager, EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.utils.spaces import Box, Dict as DictSpace

if TYPE_CHECKING:
    from envs.actions.native import NativeActionControlContext
    from envs.commands.motion_tracking.native import NativeCommandControlContext
    from simulator.native.scene import NativeSceneControlContext


@dataclass
class NativeInitialControlContext:
    """Same-instance, all-world reset boundary; no model or source payload."""

    scene: NativeSceneControlContext
    actions: dict[str, NativeActionControlContext]
    commands: dict[str, NativeCommandControlContext]
    action_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    observation_buffer: dict | None
    observations: dict
    reward_sums: dict[str, torch.Tensor]
    reward_buffer: torch.Tensor
    step_rewards: torch.Tensor
    term_dones: dict[str, torch.Tensor]
    terminated: torch.Tensor
    time_outs: torch.Tensor
    event_reset_steps: list[torch.Tensor]
    event_reset_once: list[torch.Tensor]
    sim_step_counter: int
    common_step_counter: int
    episode_length: torch.Tensor
    manual_reset_pending: torch.Tensor
    extras: dict
    step_buffers: dict[str, torch.Tensor]
    rng: dict


def _install_compute_snapshot() -> None:
    """Port of mjlab_ext._observation_manager (cache-bypassing obs pass)."""
    if hasattr(ObservationManager, "compute_snapshot"):
        return

    def _compute_snapshot(self):
        for group_name in self._group_obs_term_names:
            if self._group_obs_term_delay_buffer.get(group_name) or (
                self._group_obs_term_history_buffer.get(group_name)
            ):
                raise RuntimeError(
                    "compute_snapshot() does not support delay/history obs terms"
                )
        self._obs_buffer = None
        return self.compute(update_history=False)

    ObservationManager.compute_snapshot = _compute_snapshot


@dataclass(kw_only=True)
class ManagerEnvCfg:
    """Shared manager configuration (manager cfg dicts + timing)."""

    decimation: int
    physics_dt: float
    episode_length_s: float = 20.0
    seed: int | None = None
    num_envs: int = 1
    observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    actions: dict[str, ActionTermCfg] = field(default_factory=dict)
    events: dict[str, EventTermCfg] = field(default_factory=dict)
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    commands: dict[str, CommandTermCfg] = field(default_factory=dict)
    curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    scale_rewards_by_dt: bool = True
    auto_reset: bool = True

    # settable per-instance (FlatObsVecEnv writes it), mjlab_ext parity
    capture_final_obs: bool = False


class _NullManager:
    active_terms: list = []

    def compute(self, *a, **k):
        return None

    def compute_substep(self) -> None:
        pass

    def reset(self, env_ids=None) -> dict:
        return {}

    def record_pre_reset(self, env_ids) -> None:
        pass

    def record_post_reset(self, env_ids) -> None:
        pass

    def record_post_step(self) -> None:
        pass

    def close(self) -> None:
        pass


class ManagerBasedRlEnv:
    """One manager loop driven by the selected native scene/simulator."""

    is_vector_env = True
    cfg: ManagerEnvCfg

    def __init__(self, cfg: ManagerEnvCfg, scene, sim, device: str) -> None:
        _install_compute_snapshot()
        self.cfg = cfg
        self.scene = scene
        self.sim = sim
        self._device = device
        if cfg.seed is not None:
            self.seed(cfg.seed)

        self._sim_step_counter = 0
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}
        self.obs_buf: dict[str, torch.Tensor] = {}
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=device, dtype=torch.long
        )
        self._manual_reset_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )

        self.load_managers()

    # ── properties (mjlab parity) ──────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def physics_dt(self) -> float:
        return self.cfg.physics_dt

    @property
    def step_dt(self) -> float:
        return self.cfg.physics_dt * self.cfg.decimation

    @property
    def max_episode_length_s(self) -> float:
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def unwrapped(self) -> "ManagerBasedRlEnv":
        return self

    # ── managers ───────────────────────────────────────────────────────────

    def load_managers(self) -> None:
        self.event_manager = EventManager(self.cfg.events, self)
        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

        self.command_manager = CommandManager(self.cfg.commands, self)
        # Commands finish their reference admission before scene-owned views
        # can read them; observations/rewards are constructed only afterwards.
        for name in self.command_manager.active_terms:
            self.scene.bind_tracking_command(self.command_manager.get_term(name))
        self.action_manager = ActionManager(self.cfg.actions, self)
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.reward_manager = RewardManager(
            self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
        )
        if len(self.cfg.curriculum) > 0:
            self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        else:
            self.curriculum_manager = NullCurriculumManager()
        self.metrics_manager = _NullManager()
        self.recorder_manager = _NullManager()

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def _configure_gym_env_spaces(self) -> None:
        from mjlab.utils.spaces import batch_space

        self.single_observation_space = DictSpace()
        for group_name, _terms in self.observation_manager.active_terms.items():
            group_dim = self.observation_manager.group_obs_dim[group_name]
            assert isinstance(group_dim, tuple)
            self.single_observation_space.spaces[group_name] = Box(
                shape=group_dim, low=-math.inf, high=math.inf
            )
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = Box(
            shape=(action_dim,), low=-math.inf, high=math.inf
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    # ── reset / step (mjlab 1.3.0 + mjlab_ext bodies) ─────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ):
        del options
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        if seed is not None:
            self.seed(seed)
        full_reset = env_ids.numel() == self.num_envs
        previous_observations = None
        if not full_reset:
            previous_observations = self.obs_buf or self.observation_manager.compute()
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        if full_reset:
            self.command_manager.compute(dt=0.0)
        else:
            self.command_manager.refresh(env_ids)
        self.sim.sense()
        if full_reset:
            self.obs_buf = self.observation_manager.compute(update_history=True)
        else:
            self.obs_buf = self.observation_manager.refresh(
                env_ids, previous_observations=previous_observations
            )
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor):
        if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
            raise RuntimeError("pending manual resets before step()")

        # Action feedback reads the completed solve interval before the next one opens.
        self.action_manager.process_action(action.to(self.device))
        self.sim.begin_control_step()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.command_manager.compute_substep()
            self.scene.write_data_to_sim()
            self.sim.step()
            self._bump_clock()
            self.scene.update(dt=self.physics_dt)
        self.sim.end_control_step()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.extras.pop("final_obs", None)
        if self.cfg.auto_reset and len(reset_env_ids) > 0:
            if getattr(self.cfg, "capture_final_obs", False) and bool(
                torch.any(self.reset_time_outs)
            ):
                self.sim.forward()
                self.extras["final_obs"] = self.observation_manager.compute_snapshot()
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()

        self.sim.forward()

        self.command_manager.compute(dt=self.step_dt)

        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if not self.cfg.auto_reset and len(reset_env_ids) > 0:
            self._manual_reset_pending[reset_env_ids] = True

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        self.curriculum_manager.compute(env_ids=env_ids)
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        self.extras["log"] = dict()
        for mgr in (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
        ):
            info = mgr.reset(env_ids)
            if info:
                self.extras["log"].update(info)
        self.episode_length_buf[env_ids] = 0
        self._manual_reset_pending[env_ids] = False
        self._bump_clock()

    def _native_context_terms(self):
        from envs.actions.native import NativeResidualAction
        from envs.commands.motion_tracking.native import NativeMotionTrackingCommand
        from simulator.native.scene import NativeScene, NativeSim

        if not isinstance(self.scene, NativeScene) or not isinstance(self.sim, NativeSim):
            raise NotImplementedError("complete control context is implemented only for NativePlant scenes")
        actions = {name: self.action_manager.get_term(name) for name in self.action_manager.active_terms}
        commands = {name: self.command_manager.get_term(name) for name in self.command_manager.active_terms}
        if any(not isinstance(term, NativeResidualAction) for term in actions.values()) or any(
            not isinstance(term, NativeMotionTrackingCommand) for term in commands.values()
        ):
            raise NotImplementedError("control context requires native action and command owners")
        observations = self.observation_manager
        if any(observations._group_obs_term_delay_buffer.values()) or any(
            observations._group_obs_term_history_buffer.values()
        ):
            raise NotImplementedError("control context does not restore observation delay/history")
        if any(observations._group_obs_class_term_cfgs.values()) or observations._group_obs_class_instances:
            raise NotImplementedError("control context does not restore stateful observation terms")
        if self.curriculum_manager.active_terms or self.metrics_manager.active_terms or self.recorder_manager.active_terms:
            raise NotImplementedError("control context requires paused curriculum and inactive manager recorders")
        if {"step", "interval"}.intersection(self.event_manager.available_modes):
            raise NotImplementedError("control context does not restore running event terms")
        if any(self.event_manager._mode_class_term_cfgs.values()) or any(
            getattr(manager, "_class_term_cfgs", ())
            for manager in (self.reward_manager, self.termination_manager)
        ):
            raise NotImplementedError("control context does not restore stateful reward/reset/termination terms")
        return actions, commands

    def capture_initial_control_context(self) -> NativeInitialControlContext:
        """Capture after complete all-world reset, before its first control.

        History/delay and unknown backend/controller carry are rejected rather
        than represented by Q/Qvel. The same env/config owns every restore.
        """
        from common.utils import capture_rng_state

        actions, commands = self._native_context_terms()
        if bool(self.episode_length_buf.any()) or bool(self._manual_reset_pending.any()) or self.sim._pending is not None:
            raise RuntimeError("initial control context requires a committed all-world reset boundary")
        interval = self.sim.plant.contact_interval
        if interval is not None and (interval._active or bool(interval.sample_recorded.any())):
            raise RuntimeError("initial control context requires no executed control interval")
        return NativeInitialControlContext(
            self.scene.capture_control_context(),
            {name: term.capture_control_context() for name, term in actions.items()},
            {name: term.capture_control_context() for name, term in commands.items()},
            tuple(value.clone() for value in (self.action_manager.action,
                                             self.action_manager.prev_action,
                                             self.action_manager.prev_prev_action)),
            deepcopy(self.observation_manager._obs_buffer), deepcopy(self.obs_buf),
            deepcopy(self.reward_manager._episode_sums), self.reward_manager._reward_buf.clone(),
            self.reward_manager._step_reward.clone(), deepcopy(getattr(self.termination_manager, "_term_dones", {})),
            self.termination_manager.terminated.clone(), self.termination_manager.time_outs.clone(),
            deepcopy(self.event_manager._reset_term_last_triggered_step_id),
            deepcopy(self.event_manager._reset_term_last_triggered_once),
            self._sim_step_counter, self.common_step_counter, self.episode_length_buf.clone(),
            self._manual_reset_pending.clone(), deepcopy(self.extras),
            {name: getattr(self, name).clone() for name in
             ("reset_buf", "reset_terminated", "reset_time_outs", "reward_buf") if hasattr(self, name)},
            capture_rng_state(None),
        )

    def restore_initial_control_context(self, context: NativeInitialControlContext) -> None:
        """Restore exact initial owner state, without reset/resample or integration."""
        from common.utils import restore_rng_state

        actions, commands = self._native_context_terms()
        if actions.keys() != context.actions.keys() or commands.keys() != context.commands.keys():
            raise ValueError("initial control context requires the same named terms")
        self.scene.restore_control_context(context.scene)
        for name, term in actions.items():
            term.restore_control_context(context.actions[name])
        for name, term in commands.items():
            term.restore_control_context(context.commands[name])
        for target, value in zip((self.action_manager.action, self.action_manager.prev_action,
                                  self.action_manager.prev_prev_action), context.action_history, strict=True):
            target.copy_(value)
        self.observation_manager._obs_buffer = deepcopy(context.observation_buffer)
        self.obs_buf = deepcopy(context.observations)
        self.reward_manager._episode_sums = deepcopy(context.reward_sums)
        self.reward_manager._reward_buf.copy_(context.reward_buffer)
        self.reward_manager._step_reward.copy_(context.step_rewards)
        for name, value in context.term_dones.items():
            self.termination_manager._term_dones[name].copy_(value)
        self.termination_manager.terminated.copy_(context.terminated)
        self.termination_manager.time_outs.copy_(context.time_outs)
        for target, value in zip(self.event_manager._reset_term_last_triggered_step_id, context.event_reset_steps, strict=True):
            target.copy_(value)
        for target, value in zip(self.event_manager._reset_term_last_triggered_once, context.event_reset_once, strict=True):
            target.copy_(value)
        self._sim_step_counter, self.common_step_counter = context.sim_step_counter, context.common_step_counter
        self.episode_length_buf.copy_(context.episode_length)
        self._manual_reset_pending.copy_(context.manual_reset_pending)
        self.extras = deepcopy(context.extras)
        for name in ("reset_buf", "reset_terminated", "reset_time_outs", "reward_buf"):
            if name in context.step_buffers:
                setattr(self, name, context.step_buffers[name].clone())
            else:
                self.__dict__.pop(name, None)
        # mjlab spaces carry shape metadata, not a sampler. Restore global
        # draws last; an external Gym wrapper retains its own sampler state.
        restore_rng_state(context.rng, None)

    def _bump_clock(self) -> None:
        clock = getattr(self.scene, "_clock", None)
        if clock is not None:
            clock.bump()

    def close(self) -> None:
        pass

    @staticmethod
    def seed(seed: int = -1) -> int:
        if seed == -1:
            seed = int(np.random.randint(0, 10_000))
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed
