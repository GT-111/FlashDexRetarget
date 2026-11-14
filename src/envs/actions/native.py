"""Flash residual/absolute actions for named FREE native hand articulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_box_plus,
    quat_inv,
    quat_mul,
    unscale_transform,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

from envs.scene.native import NativeSceneCfg

from .lpf import ema_alpha


@dataclass(kw_only=True)
class NativeResidualActionCfg(ActionTermCfg):
    command_name: str
    native_scene: dict
    translation_residual_m: float
    rotation_residual_rad: float
    finger_residual_rad: float | None = None
    action_scale: float = 1.0
    action_offset: float = 0.0
    cutoff_hz: dict[str, float | None] = field(default_factory=dict)

    def build(self, env) -> NativeResidualAction:
        return NativeResidualAction(self, env)


class NativeResidualAction(ActionTerm):
    """WORLD-left SE(3) root residuals plus independent native finger commands.

    Layout is all named root translations, all named root rotation vectors,
    then the declared independent finger actuators. Passive mimic coordinates
    never become policy outputs. The source root is already registered to the
    physical target body; raw MANO wrist poses are not accepted here.
    """

    def __init__(self, cfg: NativeResidualActionCfg, env):
        super().__init__(cfg, env)
        self.layout = NativeSceneCfg.from_mapping(cfg.native_scene)
        self.hand_sides = tuple(h.side for h in self.layout.hands)
        self.hand_body_names = tuple(h.root_body_name for h in self.layout.hands)
        self.finger_actuator_names = tuple(
            a for h in self.layout.hands for a in h.command_actuator_names
        )
        finger_ids, self.finger_joint_names = (
            self._entity.find_joints_by_actuator_names(self.finger_actuator_names)
        )
        self._finger_ids = torch.tensor(
            finger_ids, dtype=torch.long, device=self.device
        )
        limits = torch.as_tensor(
            self._entity.native_command_limits(self.finger_actuator_names),
            dtype=torch.float32,
            device=self.device,
        )
        self._lower, self._upper = limits.unbind(-1)
        roots = len(self.hand_sides)
        trans_names = [
            f"{s}_root_translation_{a}" for s in self.hand_sides for a in "xyz"
        ]
        rot_names = [f"{s}_root_rotation_{a}" for s in self.hand_sides for a in "xyz"]
        self.action_names = (*trans_names, *rot_names, *self.finger_actuator_names)
        root_sides = tuple(s for s in self.hand_sides for _ in "xyz")
        self.action_sides = (
            *root_sides,
            *root_sides,
            *(h.side for h in self.layout.hands for _ in h.command_actuator_names),
        )
        self._translation_slice = slice(0, 3 * roots)
        self._rotation_slice = slice(3 * roots, 6 * roots)
        self._finger_slice = slice(6 * roots, len(self.action_names))
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._filter_state = torch.zeros_like(self._raw_actions)
        self._filter_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._alpha = torch.cat(
            [
                torch.full((size,), ema_alpha(cfg.cutoff_hz.get(group), env.step_dt))
                for group, size in (
                    ("wrist_trans", 3 * roots),
                    ("wrist_rot", 3 * roots),
                    ("finger", len(finger_ids)),
                )
            ]
        ).to(self.device)
        self._clip = None
        if cfg.clip is not None:
            self._clip = torch.tensor(
                [[-float("inf"), float("inf")]], device=self.device
            ).repeat(self.action_dim, 1)
            ids, _, values = resolve_matching_names_values(
                dict(cfg.clip), self.action_names
            )
            self._clip[ids] = torch.tensor(values, device=self.device)
        self._finger_target = torch.zeros(
            self.num_envs, len(finger_ids), device=self.device
        )
        self._finger_source_support = torch.zeros_like(
            self._finger_target, dtype=torch.bool
        )
        self._root_target = torch.zeros(self.num_envs, roots, 13, device=self.device)
        self._root_support = torch.zeros(
            self.num_envs, roots, dtype=torch.bool, device=self.device
        )
        self._linear_support = torch.zeros_like(self._root_support)
        self._angular_support = torch.zeros_like(self._root_support)
        self._installed_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reference_motion_ids = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._reference_motion_steps = torch.zeros_like(self._reference_motion_ids)
        self._root_force = torch.zeros(self.num_envs, roots, 3, device=self.device)
        self._root_torque = torch.zeros_like(self._root_force)
        self._root_control_metadata = {
            "law": "WORLD_subtree_mass_inertia_PD_at_link_COM",
            "gains_and_bounds": {
                h.side: asdict(h.root_control) for h in self.layout.hands
            },
            "effort_authority": "hand_roots_only_no_object_assistance",
            "main_tracker_numeric_parity": False,
            "application": getattr(env.scene, "root_wrench_application", None),
        }
        self._action_recipe = {
            "action_names": self.action_names,
            "action_sides": self.action_sides,
            "scale": cfg.action_scale,
            "offset": cfg.action_offset,
            "processed_action_clip": None
            if self._clip is None
            else self._clip.cpu().tolist(),
            "translation_residual_per_axis_m": cfg.translation_residual_m,
            "rotation_residual_per_axis_rad": cfg.rotation_residual_rad,
            "rotation_convention": "world_left_rotvec",
            "finger_residual_rad": cfg.finger_residual_rad,
            "finger_effective_limits_rad": limits.cpu().tolist(),
            "cutoff_hz": dict(cfg.cutoff_hz),
            "filter_alpha": self._alpha.cpu().tolist(),
            "control_period_s": float(env.step_dt),
            "condition": "external_native_producer_not_main_tracker_numeric_parity",
            "backend": getattr(env.scene, "backend_identity", None),
        }
        self.ref_joint_pos_used = None

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def finger_command_limits(self) -> torch.Tensor:
        return torch.stack((self._lower, self._upper), -1)

    @property
    def installed_control(self) -> dict:
        """Accepted targets, separate from raw SAC outputs and source authority."""
        return {
            "hand_sides": self.hand_sides,
            "hand_body_names": self.hand_body_names,
            "finger_actuator_names": self.finger_actuator_names,
            "finger_joint_names": tuple(self.finger_joint_names),
            "finger_target_rad": self._finger_target,
            "finger_source_support": self._finger_source_support,
            "root_state_w": self._root_target,
            "root_pose_support": self._root_support,
            "root_linear_velocity_support": self._linear_support,
            "root_angular_velocity_support": self._angular_support,
            "installed_valid": self._installed_valid,
            "reference_motion_ids": self._reference_motion_ids,
            "reference_motion_steps": self._reference_motion_steps,
            "root_force_n": self._root_force,
            "root_torque_nm": self._root_torque,
            "root_control": self._root_control_metadata,
            "action_recipe": self._action_recipe,
        }

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        processed = actions * self.cfg.action_scale + self.cfg.action_offset
        if self._clip is not None:
            processed = processed.clamp(self._clip[:, 0], self._clip[:, 1])
        self._filter_state[self._filter_pending] = processed[self._filter_pending]
        self._filter_state.lerp_(processed, self._alpha)
        self._filter_pending[:] = False
        self._processed_actions[:] = self._filter_state

    def apply_actions(self) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        ids = command._motion_flat_ids
        ml = command.motion_lib
        self._reference_motion_ids[:] = command.motion_ids
        self._reference_motion_steps[:] = command.motion_steps
        self.ref_joint_pos_used = command.ref_joint_pos
        support = ml.native_joint_position_support[ids][:, self._finger_ids]
        action = self._processed_actions[:, self._finger_slice]
        if self.cfg.finger_residual_rad is None:
            requested = unscale_transform(action, self._lower, self._upper)
        else:
            requested = (
                self.ref_joint_pos_used[:, self._finger_ids]
                + action * self.cfg.finger_residual_rad
            )
        carried = self._entity.data.joint_pos_target[:, self._finger_ids]
        self._finger_target[:] = torch.where(support, requested, carried).clamp(
            self._lower, self._upper
        )
        self._finger_source_support[:] = support
        self._entity.set_joint_position_target(self._finger_target, self._finger_ids)
        translation = self._processed_actions[:, self._translation_slice].reshape(
            self.num_envs, -1, 3
        )
        rotation = self._processed_actions[:, self._rotation_slice].reshape(
            self.num_envs, -1, 3
        )
        for i, hand in enumerate(self.layout.hands):
            if ml.native_root_body_names[hand.side] != hand.root_body_name:
                raise ValueError(
                    "source native root reference names a different physical body"
                )
            target = ml.native_root_state_w[hand.side][ids].clone()
            target[:, :3] += self._env.scene.env_origins
            pose_support = ml.native_root_pose_support[hand.side][ids]
            linear_support = (
                pose_support & ml.native_root_linear_velocity_support[hand.side][ids]
            )
            angular_support = (
                pose_support & ml.native_root_angular_velocity_support[hand.side][ids]
            )
            target[:, :3] += translation[:, i] * self.cfg.translation_residual_m
            reference_quat = target[:, 3:7].clone()
            target[:, 3:7] = quat_box_plus(
                reference_quat, rotation[:, i] * self.cfg.rotation_residual_rad
            )
            target[:, 7:10] = torch.where(linear_support[:, None], target[:, 7:10], 0.0)
            angular = quat_apply(
                quat_mul(target[:, 3:7], quat_inv(reference_quat)), target[:, 10:13]
            )
            target[:, 10:13] = torch.where(angular_support[:, None], angular, 0.0)
            self._root_target[:, i] = target
            self._root_support[:, i] = pose_support
            self._linear_support[:, i] = linear_support
            self._angular_support[:, i] = angular_support
            force, torque = self._entity.hands[hand.side].apply_root_pose_target(
                target, pose_support, hand.root_control
            )
            self._root_force[:, i], self._root_torque[:, i] = force, torque
        self._installed_valid[:] = True

    def reset(self, env_ids=None) -> None:
        rows = slice(None) if env_ids is None else env_ids
        self._raw_actions[rows] = 0.0
        self._filter_pending[rows] = True
        self._installed_valid[rows] = False
