from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def gravity_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    schedule_steps: int = 1920,
    full_g: float = 9.81,
) -> dict[str, torch.Tensor]:
    """One scene-global -z gravity schedule, including subset-reset calls."""
    if schedule_steps > 0 and not env.sim.supports_dynamic_gravity:
        raise NotImplementedError("this backend supports constant constructor gravity only")
    s = int(env.common_step_counter)
    if schedule_steps <= 0:
        frac = 1.0
    else:
        frac = min(s / max(1, schedule_steps), 1.0)
    gz = -float(full_g) * frac
    actual = env.sim.set_gravity_world_m_per_s2((0.0, 0.0, gz))
    return {"gravity_z": torch.tensor(actual[2])}


def xfrc_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    command_name: str = "motion",
    omega_n_start: float = 0.0,
    omega_n_end: float = 0.0,
    schedule_steps: int = 0,
    delay_steps: int = 0,
    zeta: float = 1.0,
    drive_omega_rot: bool = True,
    rot_omega_mult: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Soft-attractor curriculum for pin_mode="xfrc": kp_pos = m·ω_n², kv_pos =
    2ζ·m·ω_n (mass auto-detected); ω_n ramps start -> end over schedule_steps."""
    s = int(env.common_step_counter)
    s_post_delay = max(0, s - int(delay_steps))
    if schedule_steps <= 0:
        w = float(omega_n_start)
    else:
        frac = min(s_post_delay / max(1, schedule_steps), 1.0)
        w = float(omega_n_start) + frac * (float(omega_n_end) - float(omega_n_start))

    cmd = env.command_manager.get_term(command_name)
    sides = list(cmd.cfg.object.entity_names or {})
    # Preserve the side-mean gain recipe independently in every world. Actual
    # selected slots and mixed shared-object aliases use the state-read owner.
    mass_env = torch.stack([cmd._obj_mass_env(side) for side in sides]).mean(0)
    cmd._xfrc_kp_pos_env = mass_env * w * w
    cmd._xfrc_kv_pos_env = 2.0 * float(zeta) * mass_env * w
    m = float(mass_env.mean())
    kp = m * w * w
    kv = 2.0 * float(zeta) * m * w
    cmd.cfg.object.xfrc.kp.pos = kp
    cmd.cfg.object.xfrc.kd.pos = kv
    # ω_rot = rot_omega_mult·ω_n: mult>1 holds orientation stiffer (kp_rot ∝ mult²)
    # yet decays to 0 on the same schedule; mult=1 keeps the old ω_rot = ω_n.
    w_rot = float(w) * float(rot_omega_mult)
    if drive_omega_rot:
        cmd.cfg.object.xfrc.omega_rot = w_rot
    return {
        "xfrc_omega_n": torch.tensor(w),
        "xfrc_kp_pos": torch.tensor(kp),
        "xfrc_kv_pos": torch.tensor(kv),
        "xfrc_obj_mass": torch.tensor(m),
        "xfrc_omega_rot": torch.tensor(w_rot)
        if drive_omega_rot
        else torch.tensor(0.0),
    }


def xfrc_curriculum_adaptive(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    command_name: str = "motion",
    kp_init: float = 80.0,
    kv_init: float = 5.0,
    kp_lower_init: float | None = None,
    kv_lower_init: float | None = None,
    epochs_since_decay_init: int = 0,
    force_range: float = 50.0,
    wait_epochs: int = 100,
    deque_len: int = 30,
    reward_terms: dict | None = None,
    rew_thresholds: dict | None = None,
    upper_ratios: dict | None = None,
    lower_ratios: dict | None = None,
    dialback_completion_thres: float = 0.5,
    dialback_min_epochs: int = 500,
    dialback_ratios: dict | None = None,
    num_steps_per_env: int = 16,
    decay_cooldown_epochs: int = 40,
    kp_zero_thres: float = 0.05,
    kp_lower_zero_thres: float = 0.1,
    completion_thres: float = 0.9,
    rot_stiffness_mult: float = 1.0,
    kp_floor: float = 0.0,
    kv_floor: float = 0.0,
    per_dof_gains: bool = False,
    zero_epoch: int = -1,
    completion_term: str | None = None,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Adaptive DexMachina-style object-assist: PD assist gains decay once rewards +
    completion clear thresholds. Stateful logic lives in ``XfrcAssistCurriculum``."""
    from .assist import XfrcAssistCurriculum

    cmd = env.command_manager.get_term(command_name)
    ctrl = getattr(cmd, "_xfrc_curr_ctrl", None)
    if ctrl is None:
        ctrl = XfrcAssistCurriculum(
            command=cmd,
            num_envs=env.num_envs,
            max_episode_length=int(env.max_episode_length),
            num_steps_per_env=int(num_steps_per_env),
            kp_init=float(kp_init),
            kv_init=float(kv_init),
            kp_lower_init=kp_lower_init,
            kv_lower_init=kv_lower_init,
            epochs_since_decay_init=int(epochs_since_decay_init),
            force_range=float(force_range),
            wait_epochs=int(wait_epochs),
            deque_len=int(deque_len),
            reward_terms=dict(reward_terms or {}),
            rew_thresholds=dict(rew_thresholds or {}),
            upper_ratios=dict(upper_ratios or {}),
            lower_ratios=dict(lower_ratios or {}),
            dialback_completion_thres=float(dialback_completion_thres),
            dialback_min_epochs=int(dialback_min_epochs),
            dialback_ratios=dict(dialback_ratios or {}),
            decay_cooldown_epochs=int(decay_cooldown_epochs),
            kp_zero_thres=float(kp_zero_thres),
            kp_lower_zero_thres=float(kp_lower_zero_thres),
            completion_thres=float(completion_thres),
            rot_stiffness_mult=float(rot_stiffness_mult),
            kp_floor=float(kp_floor),
            kv_floor=float(kv_floor),
            per_dof_gains=bool(per_dof_gains),
            zero_epoch=int(zero_epoch),
            completion_term=completion_term,
            seed=int(seed),
            device=str(env.device),
        )
        cmd._xfrc_curr_ctrl = ctrl
    return ctrl.log_dict()


def xfrc_curriculum_adaptive_omega(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    command_name: str = "motion",
    omega_n_start: float = 20.0,
    omega_floor: float = 1.0,
    rot_omega_mult: float = 1.5,
    zeta: float = 1.0,
    decay_ratio: float = 0.9,
    dialup_ratio: float = 1.1111,
    wait_epochs: int = 100,
    deque_len: int = 30,
    cooldown_epochs: int = 40,
    num_steps_per_env: int = 32,
    completion_thres: float = 0.7,
    dialback_thres: float = 0.5,
    dialback_min_epochs: int = 200,
) -> dict[str, torch.Tensor]:
    """Adaptive ω_n object-assist for the base pin: decays ω_n (kp = m·ω², omega_rot =
    rot_omega_mult·ω) gated on clip-completion rate. Stateful: ``OmegaAssistCurriculum``."""
    from .omega_assist import OmegaAssistCurriculum

    cmd = env.command_manager.get_term(command_name)
    ctrl = getattr(cmd, "_xfrc_curr_ctrl", None)
    if ctrl is None:
        ctrl = OmegaAssistCurriculum(
            command=cmd,
            env=env,
            num_steps_per_env=int(num_steps_per_env),
            omega_n_start=float(omega_n_start),
            omega_floor=float(omega_floor),
            rot_omega_mult=float(rot_omega_mult),
            zeta=float(zeta),
            decay_ratio=float(decay_ratio),
            dialup_ratio=float(dialup_ratio),
            wait_epochs=int(wait_epochs),
            deque_len=int(deque_len),
            cooldown_epochs=int(cooldown_epochs),
            completion_thres=float(completion_thres),
            dialback_thres=float(dialback_thres),
            dialback_min_epochs=int(dialback_min_epochs),
            device=str(env.device),
        )
        cmd._xfrc_curr_ctrl = ctrl
    return ctrl.log_dict()


def build_obj_term_stages(
    delay_steps: int,
    duration_steps: int,
) -> tuple[list[dict], list[dict]]:
    """(pos_stages, rot_stages) for ManipTrans-style obj-termination tightening:
    thresholds shrink cubically from ~0.058 m / ~87° to 0.02 m / 30° over fracs 0..1."""
    pos_stages: list[dict] = []
    rot_stages: list[dict] = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        scale = (math.e * 2.0) ** (-frac) * 0.3 + 0.7
        pos_thr = 0.02 / 0.343 * scale**3
        rot_thr = 30.0 / 0.343 * scale**3
        s = delay_steps + int(duration_steps * frac)
        pos_stages.append({"step": s, "params": {"threshold": float(pos_thr)}})
        rot_stages.append({"step": s, "params": {"threshold_deg": float(rot_thr)}})
    return pos_stages, rot_stages
