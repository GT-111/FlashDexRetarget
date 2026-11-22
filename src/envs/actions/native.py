"""Flash residual/absolute actions for named FREE native hand articulations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_box_minus,
    quat_box_plus,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    unscale_transform,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

from envs.scene.native import NativeSceneCfg

from .lpf import ema_alpha


def _contact_compression(
    solve, roles, solved, body_hands, body_parts, command_indices, command_count, travel_m,
):
    """Project each loaded pad's inward normal travel to the named command coordinates."""
    bodies = solve["native_bodies"]
    world = solve["world"].clamp(0, roles.shape[0] - 1)
    hand, part = body_hands[bodies], body_parts[bodies]
    orientation = (hand[:, 0] >= 0) & (part[:, 1] >= 0)
    assigned_hand = torch.where(orientation, hand[:, 0], hand[:, 1])
    assigned_part = torch.where(orientation, part[:, 1], part[:, 0])
    hand_body = torch.where(orientation, bodies[:, 0], bodies[:, 1])
    force = solve["normal_force_n"]
    eligible = (solve["valid"] & solved[world] & (assigned_hand >= 0) & (assigned_part >= 0)
                & roles[world, assigned_hand.clamp_min(0), assigned_part.clamp_min(0)] & (force >= .08))
    parts = roles.shape[-1]
    pair_slot = (world * roles.shape[1] + assigned_hand.clamp_min(0)) * parts + assigned_part.clamp_min(0)
    bearing = force.new_zeros(roles.numel())
    bearing.scatter_reduce_(0, pair_slot, torch.where(eligible, force, 0.), reduce="amax")
    slot = world * len(body_hands) + hand_body
    strongest = force.new_zeros(roles.shape[0] * len(body_hands))
    strongest.scatter_reduce_(0, slot, torch.where(eligible, force, 0.), reduce="amax")
    contact = torch.arange(len(force), device=force.device)
    pick = torch.full_like(strongest, len(force), dtype=torch.long)
    selected = eligible & (force == strongest[slot])
    pick.scatter_reduce_(0, slot, torch.where(selected, contact, len(force)), reduce="amin")
    present = (pick < len(force)).reshape(roles.shape[0], -1)
    pick = pick.clamp_max(len(force) - 1).reshape_as(present)
    forward = orientation[pick]
    solver_body = torch.where(forward, solve["solver_bodies"][pick, 0], solve["solver_bodies"][pick, 1])
    centers = solve["subtree_com_w"].gather(1, solve["body_root"][solver_body, None].expand(-1, -1, 3))
    normal = solve["normal_w"][pick] * torch.where(forward[..., None], 1., -1.)
    arm = solve["point_w"][pick] - centers
    subspace = solve["motion_subspace"]
    derivative = (torch.einsum("wbi,wai->wba", normal, subspace[..., 3:])
                  + torch.einsum("wbi,wai->wba", torch.linalg.cross(arm, normal), subspace[..., :3]))
    derivative = torch.where(present[..., None] & solve["body_is_dof_ancestor"][solver_body], derivative, 0.)
    projected = derivative.new_zeros((*derivative.shape[:2], command_count))
    projected.scatter_add_(-1, command_indices.expand_as(derivative), derivative)
    pressure = torch.where(present, force[pick], 0.)
    travel = travel_m * (pressure / .30).clamp(max=1.)
    transpose = projected.transpose(1, 2)
    normal = transpose @ projected
    normal.diagonal(dim1=-2, dim2=-1).add_(.014 ** 2)
    correction = torch.linalg.solve_ex(normal, (transpose @ travel[..., None]).squeeze(-1)).result
    return correction, bearing.reshape_as(roles) > 0.


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
    source_contact_approach: dict[str, float] | None = None
    wrist_target_twist: Literal[
        "source_only", "filtered_correction_feedforward"
    ] = "source_only"

    def build(self, env) -> NativeResidualAction:
        return NativeResidualAction(self, env)


@dataclass
class NativeActionControlContext:
    raw_actions: torch.Tensor
    processed_actions: torch.Tensor
    filter_state: torch.Tensor
    filter_pending: torch.Tensor
    source_contact_translation: torch.Tensor
    source_contact_part: torch.Tensor
    source_contact_compression: torch.Tensor
    source_hold_active: torch.Tensor
    source_hold_offset_m: torch.Tensor
    source_hold_force_n: torch.Tensor
    source_transport_part: torch.Tensor
    source_transport_pose: torch.Tensor
    source_transport_commands: torch.Tensor
    source_transport_rotation: torch.Tensor
    idle_support_offset_m: torch.Tensor
    correction_twist: torch.Tensor
    correction_interval_valid: torch.Tensor
    finger_requested: torch.Tensor
    finger_requested_support: torch.Tensor
    finger_target: torch.Tensor
    finger_source_support: torch.Tensor
    root_target: torch.Tensor
    root_support: torch.Tensor
    linear_support: torch.Tensor
    angular_support: torch.Tensor
    installed_valid: torch.Tensor
    reference_motion_ids: torch.Tensor
    reference_motion_steps: torch.Tensor
    root_force: torch.Tensor
    root_torque: torch.Tensor
    reference_command_position: torch.Tensor | None


class NativeResidualAction(ActionTerm):
    """WORLD-left SE(3) root residuals plus independent native finger commands.

    Layout is all named root translations, all named root rotation vectors,
    then the declared independent logical commands. Passive mimic coordinates
    never become policy outputs. The source root is already registered to the
    physical target body; raw MANO wrist poses are not accepted here.
    """

    def __init__(self, cfg: NativeResidualActionCfg, env):
        super().__init__(cfg, env)
        if cfg.wrist_target_twist not in (
            "source_only", "filtered_correction_feedforward"
        ):
            raise ValueError("unsupported native wrist target-twist convention")
        self._source_contact_scale = 0.
        self._source_contact_guard = None
        self._source_contact_speed_m_s = 0.12
        self._pad_travel_m = 0.
        self._stationary_hold_force_n = 0.
        self._transport_position_gain = 0.
        self._idle_support_clearance_m = 0.
        if cfg.source_contact_approach is not None:
            self._source_contact_guard = dict(cfg.source_contact_approach)
            self._source_contact_scale = self._source_contact_guard.pop("translation_m")
            self._source_contact_speed_m_s = self._source_contact_guard.pop(
                "max_speed_m_s", self._source_contact_speed_m_s)
            self._pad_travel_m = self._source_contact_guard.pop("pad_compression_m", 0.)
            self._stationary_hold_force_n = self._source_contact_guard.pop("stationary_hold_force_n", 0.)
            self._transport_position_gain = self._source_contact_guard.pop("transport_position_gain", 0.)
            self._idle_support_clearance_m = self._source_contact_guard.pop("idle_support_clearance_m", 0.)
            if (not 0. < self._source_contact_scale <= cfg.translation_residual_m
                    or not all(math.isfinite(value) and value > 0. for value in self._source_contact_guard.values())):
                raise ValueError("source contact approach requires positive pose and tip bounds within the wrist budget")
            if not 0. <= self._pad_travel_m <= self._source_contact_scale:
                raise ValueError("pad compression must be between zero and the source contact approach distance")
            if not math.isfinite(self._source_contact_speed_m_s) or self._source_contact_speed_m_s <= 0.:
                raise ValueError("source contact approach requires a finite positive speed")
            if not math.isfinite(self._stationary_hold_force_n) or self._stationary_hold_force_n < 0.:
                raise ValueError("stationary hold force requires a finite nonnegative load")
            if not 0. <= self._transport_position_gain <= 1. or (
                    self._transport_position_gain and not self._stationary_hold_force_n):
                raise ValueError("rigid transport requires a position gain in [0, 1] and a supported hold")
            if not 0. <= self._idle_support_clearance_m <= self._source_contact_scale:
                raise ValueError("idle support clearance must fit the source contact approach bound")
        self.layout = NativeSceneCfg.from_mapping(cfg.native_scene)
        self.hand_sides = tuple(h.side for h in self.layout.hands)
        self.hand_body_names = tuple(h.root_body_name for h in self.layout.hands)
        self.finger_actuator_names = tuple(
            a for h in self.layout.hands for a in h.command_actuator_names
        )
        finger_ids, self.finger_joint_names = (
            self._entity.find_joints_by_actuator_names(self.finger_actuator_names)
        )
        self.finger_command_names, _, mapping = self.layout.command_coordinates(self.finger_joint_names)
        self._actuator_command_mapping = mapping
        self.actuator_command_indices = torch.tensor(mapping, dtype=torch.long, device=self.device)
        counts = torch.bincount(self.actuator_command_indices, minlength=len(self.finger_command_names))
        self.actuator_command_weights = counts[self.actuator_command_indices].float().reciprocal()
        self._finger_command_hand = torch.as_tensor(
            [i for i, hand in enumerate(self.layout.hands) for _ in range(hand.command_count)],
            dtype=torch.long, device=self.device)
        if self._pad_travel_m:
            self._plant = getattr(getattr(env, "sim", None), "plant", None)
            if self._plant is None:
                raise ValueError("native pad compression requires a solved contact differential")
            model = self._plant.model
            actuators = [model.actuator(name).id for name in self.finger_actuator_names]
            dofs = torch.as_tensor(model.jnt_dofadr[model.actuator_trnid[actuators, 0]],
                                   dtype=torch.long, device=self.device)
            self._plant.configure_contact_differential(dofs)
            self._pad_body_hand = torch.full((model.nbody,), -1, dtype=torch.long, device=self.device)
            roots = torch.as_tensor(model.body_rootid, device=self.device)
            for i, hand in enumerate(self.layout.hands):
                self._pad_body_hand[roots == model.body(hand.root_body_name).id] = i
            self._pad_body_part = torch.full_like(self._pad_body_hand, -1)
            for i, (_, part) in enumerate(self.layout.part_bindings):
                self._pad_body_part[model.body(part.body_name).id] = i
        self._finger_ids = torch.tensor(
            finger_ids, dtype=torch.long, device=self.device
        )
        self.finger_actuator_limits = torch.as_tensor(
            self._entity.native_command_limits(self.finger_actuator_names),
            dtype=torch.float32,
            device=self.device,
        )
        limits, action_names = [], []
        start = 0
        for hand in self.layout.hands:
            end = start + len(hand.command_actuator_names)
            actuator_limits = self.finger_actuator_limits[start:end]
            limits.append(torch.as_tensor(hand.command_limits, device=self.device,
                                          dtype=actuator_limits.dtype) if hand.command_limits else actuator_limits)
            # Keep exact one-actuator policy columns named as in existing checkpoints.
            exact_order = hand.actuator_command_indices == tuple(range(hand.command_count))
            action_names.extend(hand.command_actuator_names if exact_order or not hand.command_names
                                else hand.command_names)
            start = end
        limits = torch.cat(limits)
        self._lower, self._upper = limits.unbind(-1)
        roots = len(self.hand_sides)
        trans_names = [
            f"{s}_root_translation_{a}" for s in self.hand_sides for a in "xyz"
        ]
        rot_names = [f"{s}_root_rotation_{a}" for s in self.hand_sides for a in "xyz"]
        self.action_names = (*trans_names, *rot_names, *action_names)
        root_sides = tuple(s for s in self.hand_sides for _ in "xyz")
        self.action_sides = (
            *root_sides,
            *root_sides,
            *(h.side for h in self.layout.hands for _ in range(h.command_count)),
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
        self._source_contact_translation = torch.zeros(self.num_envs, roots, 3, device=self.device)
        self._source_contact_part = torch.full((self.num_envs, roots), -1, dtype=torch.long, device=self.device)
        self._source_contact_compression = torch.zeros(
            self.num_envs, len(self.finger_command_names), device=self.device)
        self._source_hold_offset_m = torch.zeros(self.num_envs, roots, device=self.device)
        self._source_hold_force_n = torch.zeros_like(self._source_hold_offset_m)
        self._source_hold_active = torch.zeros_like(self._source_hold_offset_m, dtype=torch.bool)
        self._source_transport_part = torch.full_like(self._source_contact_part, -1)
        self._source_transport_pose = torch.zeros(self.num_envs, roots, 7, device=self.device)
        self._source_transport_pose[..., 3] = 1.
        self._source_transport_commands = torch.full_like(self._source_contact_compression, torch.nan)
        self._source_transport_rotation = torch.zeros_like(self._source_contact_translation)
        self._idle_support_offset_m = torch.zeros_like(self._source_hold_offset_m)
        self._idle_support_planes = ()
        if self._idle_support_clearance_m:
            import mujoco

            from .native_support import bind_idle_support_planes

            plant = getattr(getattr(env, "sim", None), "plant", None)
            model = plant.model if plant is not None else mujoco.MjModel.from_xml_path(self.layout.scene_xml_path)
            self._idle_support_planes = bind_idle_support_planes(
                model, self.hand_body_names, self._entity.body_names, self.device)
        self._source_hold_pairs = ()
        self._source_hold_reference_rows = None
        if self._stationary_hold_force_n:
            if env.scene.hand_part_contact_interval_facts() is None:
                raise ValueError("native stationary hold requires solved hand-part contact intervals")
            support_pairs = env.scene.support_contact_interval_pairs()
            if all(key in support_pairs for key in self.layout.support_pair_bindings):
                self._source_hold_pairs = tuple(self.layout.support_pair_bindings)
                object_index = {obj.entity_name: i for i, obj in enumerate(self.layout.objects)}
                self._source_hold_weights_n = env.scene.native_object_weights_n[[
                    object_index[obj.entity_name] for obj, _ in self.layout.part_bindings]]
        self._control_dt = float(env.step_dt)
        self._correction_twist = torch.zeros(
            self.num_envs, roots, 6, device=self.device
        )
        self._correction_interval_valid = torch.zeros_like(self._filter_pending)
        self._alpha = torch.cat(
            [
                torch.full((size,), ema_alpha(cfg.cutoff_hz.get(group), self._control_dt))
                for group, size in (
                    ("wrist_trans", 3 * roots),
                    ("wrist_rot", 3 * roots),
                    ("finger", len(self.finger_command_names)),
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
        self._finger_requested = torch.zeros(self.num_envs, len(self.finger_command_names), device=self.device)
        self._finger_requested_support = torch.zeros_like(self._finger_requested, dtype=torch.bool)
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
            "wrist_target_twist": cfg.wrist_target_twist,
            "root_rate_support": "independent_source_component_support",
            "correction_rate": None if cfg.wrist_target_twist == "source_only"
            else "adjacent_effective_world_increment_after_guide_and_limits_first_packet_zero",
            "finger_residual_rad": cfg.finger_residual_rad,
            "finger_effective_limits_rad": limits.cpu().tolist(),
            "cutoff_hz": dict(cfg.cutoff_hz),
            "filter_alpha": self._alpha.cpu().tolist(),
            "control_period_s": self._control_dt,
            "condition": "external_native_producer_not_main_tracker_numeric_parity",
            "backend": getattr(env.scene, "backend_identity", None),
        }
        if cfg.source_contact_approach is not None:
            self._action_recipe["source_contact_approach"] = {
                **cfg.source_contact_approach, "max_speed_m_s": self._source_contact_speed_m_s,
            }
            if self._stationary_hold_force_n:
                self._action_recipe["stationary_hold_feedback"] = "acquired_load_from_completed_solver_interval"
        self.ref_command_pos_used = None

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
    def wrist_control_state(self) -> dict:
        """Effective prior filter coordinates in the configured hand order."""
        coordinates = torch.cat((
            self._filter_state[:, self._translation_slice].reshape(self.num_envs, -1, 3),
            self._filter_state[:, self._rotation_slice].reshape(self.num_envs, -1, 3),
        ), -1)
        # A pending first action overwrites obsolete carry, even after partial reset.
        return {
            "hand_sides": self.hand_sides,
            "filtered_coordinates": torch.where(
                self._filter_pending[:, None, None], 0., coordinates).detach(),
            "filter_pending": self._filter_pending.detach(),
        }

    @property
    def installed_control(self) -> dict:
        """Accepted targets, separate from raw SAC outputs and source authority."""
        return {
            "hand_sides": self.hand_sides,
            "hand_body_names": self.hand_body_names,
            "finger_actuator_names": self.finger_actuator_names,
            "finger_joint_names": tuple(self.finger_joint_names),
            "finger_command_names": self.finger_command_names,
            "actuator_command_indices": self._actuator_command_mapping,
            "finger_requested": self._finger_requested,
            "finger_requested_support": self._finger_requested_support,
            "finger_target_rad": self._finger_target,
            "finger_source_support": self._finger_source_support,
            "root_state_w": self._root_target,
            "root_pose_support": self._root_support,
            "root_linear_velocity_support": self._linear_support,
            "root_angular_velocity_support": self._angular_support,
            "installed_valid": self._installed_valid,
            "reference_motion_ids": self._reference_motion_ids,
            "reference_motion_steps": self._reference_motion_steps,
            "source_contact_part_index": self._source_contact_part,
            "source_contact_part_body_names": tuple(part.body_name for _, part in self.layout.part_bindings),
            "source_transport_part_index": self._source_transport_part,
            "idle_support_offset_m": self._idle_support_offset_m,
            "root_force_n": self._root_force,
            "root_torque_nm": self._root_torque,
            "root_control": self._root_control_metadata,
            "action_recipe": self._action_recipe,
        }

    def capture_control_context(self) -> NativeActionControlContext:
        reference = getattr(self, "ref_command_pos_used", None)
        return NativeActionControlContext(
            **{name: getattr(self, "_" + name).clone()
               for name in NativeActionControlContext.__dataclass_fields__
               if name != "reference_command_position"},
            reference_command_position=None if reference is None else reference.clone(),
        )

    def restore_control_context(self, context: NativeActionControlContext) -> None:
        for name, value in vars(context).items():
            if name != "reference_command_position":
                getattr(self, "_" + name).copy_(value)
        if context.reference_command_position is None:
            self.__dict__.pop("ref_command_pos_used", None)
        else:
            self.ref_command_pos_used = context.reference_command_position.clone()

    def _wrist_translation(self, *, idle_clearance=True) -> torch.Tensor:
        correction = self._processed_actions[:, self._translation_slice].reshape(self.num_envs, -1, 3)
        cap = self.cfg.translation_residual_m
        translation = correction * cap
        if self._source_contact_guard is not None:
            translation = translation + self._source_contact_translation
            if idle_clearance and self._idle_support_planes:
                translation = translation + self._idle_support_offset_m[..., None] * self._env.scene.gravity_up_w
            translation = translation.clamp(-cap, cap)
        return translation

    def _wrist_rotation(self) -> torch.Tensor:
        rotation = self._processed_actions[:, self._rotation_slice].reshape(self.num_envs, -1, 3)
        rotation = rotation * self.cfg.rotation_residual_rad
        if self._transport_position_gain:
            rotation = torch.where(self._source_transport_part[..., None] >= 0,
                                   self._source_transport_rotation, rotation)
        return rotation

    def _source_loaded_approach(self, command):
        """Approach from the declared carriers, then preserve achieved pad engagement."""
        ids = command._motion_flat_ids
        intent = command.source_hand_part_intent(ids)
        fallback, nearest, drives_joint, moving = command.source_contact_approach_direction(
            ids, **self._source_contact_guard)
        if intent is None:
            self._source_contact_compression.zero_()
            return fallback, nearest
        hands = [intent["metadata"]["hand_sides"].index(side) for side in self.hand_sides]
        parts = [intent["metadata"]["part_body_names"].index(part.body_name) for _, part in self.layout.part_bindings]
        roles = (intent["active"] & intent["known"])[:, hands][:, :, parts]
        if self._pad_travel_m:
            direction, bearing = self._compress_loaded_pads(command, ids, roles, parts, fallback, nearest)
            direction = torch.where(drives_joint[..., None], fallback, direction)
        else:
            direction = fallback
            solved = self._env.scene.hand_part_contact_facts()
            if solved is None:
                return direction, nearest
            normal = torch.stack([
                torch.stack([
                    solved["pairs"][f"{side}/{obj.entity_name}/{part.name}"]["normal_force_n"]
                    for obj, part in self.layout.part_bindings
                ], dim=-1) for side in self.hand_sides
            ], dim=1)
            bearing = roles & solved["measurement_support"][:, None, None] & (normal >= .08)
        # Keep loaded support while another hand acquires the active drive side.
        part_columns = nearest.new_tensor(parts)
        retained = (bearing & (self._source_contact_part[..., None] == part_columns)).any(-1)
        was_moving = moving.gather(1, self._source_contact_part.clamp_min(0))
        retain_guide = retained & (nearest != self._source_contact_part) & ~(drives_joint | was_moving)
        direction = torch.where(retain_guide[..., None],
                                self._source_contact_translation / self._source_contact_scale, direction)
        return direction, torch.where(retain_guide, self._source_contact_part, nearest)

    def _compress_loaded_pads(self, command, ids, roles, parts, fallback, nearest):
        roots = torch.stack([command.motion_lib.native_root_state_w[side][ids, :3] for side in self.hand_sides], 1)
        roots = roots + self._env.scene.env_origins[:, None]
        live = command.sim_object_part_state_w[:, parts]
        requested = command.object_part_reference(ids)[:, parts]
        close_position = ((live[..., :3] - requested[..., :3]).norm(dim=-1)
                          <= self._source_contact_guard["max_part_position_error_m"])
        close_rotation = (quat_error_magnitude(live[..., 3:7], requested[..., 3:7])
                          <= self._source_contact_guard["max_part_rotation_error_rad"])
        approaching = roles & (close_position & close_rotation)[:, None]
        part = approaching.long().argmax(-1)
        unique = approaching.sum(-1) == 1
        aim = live[..., :3].gather(1, part[..., None].expand(-1, -1, 3)) - roots
        if len(self.hand_sides) == 2:
            paired = unique.all(1) & (part[:, 0] == part[:, 1])
            opposed = roots.flip(1) - roots
            gravity = opposed.new_tensor(self.layout.hands[0].root_control.gravity_world_m_per_s2)
            up = torch.nn.functional.normalize(gravity, dim=0)
            opposed = opposed - (opposed @ up)[..., None] * up
            aim = torch.where(paired[:, None, None], opposed, aim)
        part_columns = nearest.new_tensor(parts)
        use_grasp_axis = unique & (part_columns[part] == nearest)
        direction = torch.where(use_grasp_axis[..., None], torch.nn.functional.normalize(aim, dim=-1), fallback)
        correction, bearing = _contact_compression(
            self._plant.solved_contact_differential(), roles, self._plant.contact_support,
            self._pad_body_hand, self._pad_body_part, self.actuator_command_indices,
            len(self.finger_command_names), self._pad_travel_m,
        )
        self._source_contact_compression.lerp_(correction, .3)
        self._source_contact_compression.masked_fill_(~bearing.any(-1)[:, self._finger_command_hand], 0.)
        return direction, bearing

    def _supported_hold_translation(self, command, part, fallback):
        """Keep the contact approach until it has established a supported load."""
        scene = self._env.scene
        solved = scene.hand_part_contact_interval_facts()
        support_pairs = scene.support_contact_interval_pairs()
        index = part.clamp_min(0)
        if self._source_hold_reference_rows is None:
            reach_steps = max(1, round(self._source_contact_scale / (self._source_contact_speed_m_s * self._control_dt)))
            self._source_hold_reference_rows = command.source_stationary_part_rows(reach_steps)
        stationary = (part >= 0) & self._source_hold_reference_rows[command._motion_flat_ids].gather(1, index)
        up = scene.gravity_up_w
        support_up = torch.stack([support_pairs[key]["force_on_second_w_n"].mean(1) @ up
                                  for key in self._source_hold_pairs], 1).gather(1, index)
        hand_contacts = [[solved["pairs"][f"{hand.side}/{obj.entity_name}/{p.name}"]
                          for obj, p in self.layout.part_bindings] for hand in self.layout.hands]

        def selected(field):
            by_part = torch.stack([torch.stack([pair[field] for pair in hand], dim=1)
                                   for hand in hand_contacts], dim=1)
            address = index[..., None, None].expand(-1, -1, 1, by_part.shape[-1])
            return by_part.gather(2, address).squeeze(2)

        normal = selected("normal_force_n")
        force = normal.mean(-1)
        raw_distance = selected("force_bearing_distance_max_m")
        distance = torch.where(normal > 0., torch.nan_to_num(raw_distance, nan=float("inf")), 0.).amax(-1)
        measured = solved["interval_complete"] & solved["measurement_support"].all(-1)
        stationary &= measured[:, None] & torch.isfinite(force) & (support_up >= .25 * self._source_hold_weights_n[index])
        limit = min(.02, self._source_contact_scale)
        approach_down = (-fallback @ up - limit).clamp_min(0.)
        fallback = fallback + (torch.where(stationary, approach_down, 0.)[..., None] * up)
        capture = stationary & ~self._source_hold_active & (force >= .08) & (distance <= .0025)
        self._source_hold_active.copy_(stationary & (self._source_hold_active | capture))
        prior_vertical = self._source_contact_translation @ up
        prior_offset = (-prior_vertical).clamp(0., limit)
        self._source_hold_offset_m.copy_(torch.where(capture, prior_offset, self._source_hold_offset_m))
        self._source_hold_force_n.lerp_(torch.where(self._source_hold_active, force, 0.), .4)
        self._source_hold_force_n.copy_(torch.where(capture, force, self._source_hold_force_n))
        error = self._stationary_hold_force_n - self._source_hold_force_n
        # Relieve normal load gradually so one strong interval cannot undo the acquired approach.
        delta = torch.where(error >= 0., .001 * error, .00004 * error)
        delta = torch.where(distance <= .0025, delta, -self._source_hold_offset_m)
        max_step_m = .035 * self._control_dt
        self._source_hold_offset_m.add_(delta.clamp(-max_step_m, max_step_m)).clamp_(0., limit)
        self._source_hold_offset_m.masked_fill_(~self._source_hold_active, 0.)
        self._source_hold_force_n.masked_fill_(~self._source_hold_active, 0.)
        tangent = self._source_contact_translation - prior_vertical[..., None] * up
        return torch.where(self._source_hold_active[..., None],
                           tangent - self._source_hold_offset_m[..., None] * up, fallback)

    def _rigid_transport_translation(self, command, fallback):
        """Carry the prior solved hand target in a departing rigid part's source frame."""
        ids, ml = command._motion_flat_ids, command.motion_lib
        intent = command.source_hand_part_intent(ids)
        if intent is None:
            self._source_transport_part.fill_(-1)
            return fallback
        hands = [intent["metadata"]["hand_sides"].index(side) for side in self.hand_sides]
        roles = (intent["active"] & intent["known"])[:, hands]
        index = roles.long().argmax(-1)
        pose_support = ml.native_object_part_pose_support[ids].gather(1, index)
        wrist_support = torch.stack([ml.native_root_pose_support[side][ids] for side in self.hand_sides], 1)
        carrier = (roles.sum(-1) == 1) & command.rigid_part_mask[index] & pose_support & wrist_support

        scene = self._env.scene
        solved = scene.hand_part_contact_facts()
        support_pairs = scene.support_contact_interval_pairs()
        keys = [[f"{side}/{obj.entity_name}/{part.name}" for obj, part in self.layout.part_bindings]
                for side in self.hand_sides]

        def bearing_values(pairs, field, indices):
            values = torch.stack([torch.stack([pairs[key][field] for key in hand], -1) for hand in keys], -2)
            return values.gather(-1, indices).squeeze(-1)

        force = bearing_values(solved["pairs"], "normal_force_n", index[..., None])
        depth = bearing_values(solved["pairs"], "force_bearing_distance_max_m", index[..., None])
        eligible = carrier & solved["measurement_support"][:, None] & (force >= .08) & (depth <= .0025)
        same_clip = (self._installed_valid & (command.motion_ids == self._reference_motion_ids))[:, None]
        prior_ids = ml.length_starts[self._reference_motion_ids] + self._reference_motion_steps
        previous = command.source_hand_part_intent(prior_ids)
        prior_roles = (previous["active"] & previous["known"])[:, hands]
        previous_stationary = self._source_hold_reference_rows[prior_ids].gather(1, index)
        stationary = self._source_hold_reference_rows[ids].gather(1, index)
        up = scene.gravity_up_w
        support_up = torch.stack([support_pairs[key]["force_on_second_w_n"][:, -1] @ up
                                  for key in self._source_hold_pairs], 1).gather(1, index)
        adjacent = (ids == prior_ids + 1)[:, None]
        capture = (eligible & same_clip & self._root_support & adjacent
                   & prior_roles.gather(2, index[..., None]).squeeze(-1)
                   & previous_stationary & ~stationary
                   & (support_up >= .25 * self._source_hold_weights_n[index]))
        xyz_wxyz = index[..., None].expand(-1, -1, 7)
        pose = command.object_part_reference(ids)[..., :7].gather(1, xyz_wxyz)
        live_position = command.sim_object_part_state_w[..., :3].gather(1, index[..., None].expand(-1, -1, 3))
        error = pose[..., :3] - live_position
        interval = scene.hand_part_contact_interval_facts()
        samples = interval["measurement_support"]
        columns = index[:, None, :, None].expand(-1, samples.shape[1], -1, 1)
        loaded = (samples[:, :, None]
                  & (bearing_values(interval["pairs"], "normal_force_n", columns) >= .08)
                  & (bearing_values(interval["pairs"], "force_bearing_distance_max_m", columns) <= .0025))
        recent = interval["interval_complete"][:, None] & (loaded.sum(1) >= (samples.shape[1] + 1) // 2)
        nearby = error.norm(dim=-1) <= self._source_contact_guard["max_part_position_error_m"]
        retain = (same_clip & (self._source_transport_part == index)
                  & (eligible | (carrier & adjacent & nearby & recent)))
        capture &= ~retain
        active = retain | capture
        self._source_transport_part.copy_(torch.where(active, index, -1))

        prior_pose = command.object_part_reference(prior_ids)[..., :7].gather(1, xyz_wxyz)
        inv = quat_inv(prior_pose[..., 3:7])
        local = torch.cat((quat_apply(inv, self._root_target[..., :3] - prior_pose[..., :3]),
                           quat_mul(inv, self._root_target[..., 3:7])), -1)
        self._source_transport_pose.copy_(torch.where(capture[..., None], local, self._source_transport_pose))
        last_commands = torch.where(self._finger_requested_support, self._finger_requested, torch.nan)
        self._source_transport_commands.copy_(torch.where(capture[:, self._finger_command_hand], last_commands,
                                                          self._source_transport_commands))

        planar_error = error - (error @ up)[..., None] * up
        goal = (pose[..., :3] + quat_apply(pose[..., 3:7], self._source_transport_pose[..., :3])
                + self._transport_position_gain * planar_error)
        wrist = torch.stack([ml.native_root_state_w[side][ids, :7] for side in self.hand_sides], 1)
        offset = self._processed_actions[:, self._translation_slice].reshape_as(goal) * self.cfg.translation_residual_m
        goal = goal - (wrist[..., :3] + self._env.scene.env_origins[:, None]) - offset
        wanted_quat = quat_mul(pose[..., 3:7], self._source_transport_pose[..., 3:7])
        cap = self.cfg.rotation_residual_rad
        self._source_transport_rotation.copy_(quat_box_minus(wanted_quat, wrist[..., 3:7]).clamp(-cap, cap))
        return torch.where(active[..., None], goal, fallback)

    def _clear_idle_support(self, command, source_step, rate_limit):
        from .native_support import idle_plane_depth

        ids, ml = command._motion_flat_ids, command.motion_lib
        scene = self._env.scene
        intent = command.source_hand_part_intent(ids)
        solved = scene.hand_part_contact_facts()
        wanted = torch.zeros_like(self._idle_support_offset_m)
        if intent is not None and solved is not None:
            hands = [intent["metadata"]["hand_sides"].index(side) for side in self.hand_sides]
            idle = (~intent["active"][:, hands].any(-1) & intent["known"][:, hands].all(-1)
                    & solved["measurement_support"][:, None])
            position = self._entity.data.body_link_pos_w - scene.env_origins[:, None]
            rotation = self._entity.data.body_link_quat_w
            reference = torch.stack([ml.native_root_state_w[side][ids, :7] for side in self.hand_sides], 1)
            target_position = reference[..., :3] + self._wrist_translation(idle_clearance=False)
            target_rotation = quat_box_plus(reference[..., 3:7], self._wrist_rotation())
            for binding in self._idle_support_planes:
                hand = binding[0]
                side = self.hand_sides[hand]
                force = torch.stack([
                    solved["pairs"][f"{side}/{obj.entity_name}/{part.name}"]["normal_force_n"]
                    for obj, part in self.layout.part_bindings
                ], -1)
                depth = idle_plane_depth(position, rotation, target_position, target_rotation, scene.gravity_up_w, binding)
                eligible = (idle[:, hand] & ml.native_root_pose_support[side][ids]
                            & (force < .08).all(-1) & (depth > 0.))
                offset = (depth + self._idle_support_clearance_m).clamp(max=self._source_contact_scale)
                wanted[:, hand] = torch.where(eligible, offset, 0.)
        idle_step = (wanted - self._idle_support_offset_m).clamp(-rate_limit, rate_limit)
        combined = source_step + idle_step[..., None] * scene.gravity_up_w
        fraction = rate_limit / combined.norm(dim=-1).clamp_min(rate_limit)
        self._source_contact_translation.sub_(source_step * (1. - fraction[..., None]))
        self._idle_support_offset_m.add_(idle_step * fraction)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        processed = actions * self.cfg.action_scale + self.cfg.action_offset
        if self._clip is not None:
            processed = processed.clamp(self._clip[:, 0], self._clip[:, 1])
        pending = self._filter_pending.clone()
        if self.cfg.wrist_target_twist == "filtered_correction_feedforward":
            previous_translation = self._wrist_translation()
            previous_rotation = self._wrist_rotation()
            self._correction_interval_valid[:] = ~pending
        self._filter_state[pending] = processed[pending]
        self._filter_state.lerp_(processed, self._alpha)
        self._filter_pending[:] = False
        self._processed_actions[:] = self._filter_state
        if self._source_contact_guard is not None:
            command = self._env.command_manager.get_term(self.cfg.command_name)
            direction, part = self._source_loaded_approach(command)
            desired = self._source_contact_translation.lerp(direction * self._source_contact_scale, 0.35)
            if self._source_hold_pairs:
                changed_part = part != self._source_contact_part
                self._source_hold_active.masked_fill_(changed_part, False)
                self._source_hold_offset_m.masked_fill_(changed_part, 0.)
                self._source_hold_force_n.masked_fill_(changed_part, 0.)
                desired = self._supported_hold_translation(command, part, desired)
                if self._transport_position_gain:
                    desired = self._rigid_transport_translation(command, desired)
            step = desired - self._source_contact_translation
            limit = self._source_contact_speed_m_s * self._control_dt
            step = step * (limit / step.norm(dim=-1, keepdim=True).clamp_min(limit))
            self._source_contact_translation.add_(step)
            self._source_contact_part.copy_(part)
            if self._idle_support_planes:
                self._clear_idle_support(command, step, limit)
        if self.cfg.wrist_target_twist == "filtered_correction_feedforward":
            self._correction_twist[..., :3] = (
                (self._wrist_translation() - previous_translation) / self._control_dt
            )
            rotation = self._wrist_rotation()
            identity = rotation.new_zeros((*rotation.shape[:-1], 4))
            identity[..., 0] = 1.0
            previous = quat_box_plus(identity, previous_rotation)
            current = quat_box_plus(identity, rotation)
            rate = quat_box_minus(current, previous) / self._control_dt
            # Cancel static roundoff without discarding the rate's local Jacobian.
            self._correction_twist[..., 3:] = torch.where(
                (rotation != previous_rotation).any(-1, keepdim=True),
                rate,
                rate - rate.detach(),
            )
            # A reset seeds the filter; obsolete bytes never define a velocity.
            self._correction_twist.masked_fill_(
                ~self._correction_interval_valid[:, None, None], 0.0
            )

    def apply_actions(self) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        ids = command._motion_flat_ids
        ml = command.motion_lib
        self._reference_motion_ids[:] = command.motion_ids
        self._reference_motion_steps[:] = command.motion_steps
        self.ref_command_pos_used, support = command.requested_command_reference(ids)
        action = self._processed_actions[:, self._finger_slice]
        if self.cfg.finger_residual_rad is None:
            requested = unscale_transform(action, self._lower, self._upper)
        else:
            requested = self.ref_command_pos_used + action * self.cfg.finger_residual_rad
        if self._pad_travel_m:
            requested = requested + self._source_contact_compression
            if self.cfg.finger_residual_rad is not None:
                requested = self.ref_command_pos_used + (requested - self.ref_command_pos_used).clamp(
                    -self.cfg.finger_residual_rad, self.cfg.finger_residual_rad)
        if self._transport_position_gain:
            locked = self._source_transport_commands
            use_carried = (self._source_transport_part[:, self._finger_command_hand] >= 0) & torch.isfinite(locked)
            if self.cfg.finger_residual_rad is not None:
                residual = self.cfg.finger_residual_rad
                locked = locked.clamp(self.ref_command_pos_used - residual, self.ref_command_pos_used + residual)
            requested = torch.where(use_carried, locked, requested)
        self._finger_requested[:] = torch.where(support, requested, 0.0)
        self._finger_requested_support[:] = support
        mapping = self.actuator_command_indices
        actuator_support = support[:, mapping]
        carried = self._entity.data.joint_pos_target[:, self._finger_ids]
        self._finger_target[:] = torch.where(actuator_support, requested[:, mapping], carried).clamp(
            self.finger_actuator_limits[:, 0], self.finger_actuator_limits[:, 1]
        )
        self._finger_source_support[:] = actuator_support
        self._entity.set_joint_position_target(self._finger_target, self._finger_ids)
        translation = self._wrist_translation()
        rotation = self._wrist_rotation()
        servo_twist = getattr(ml, "native_root_servo_twist_w", None)
        for i, hand in enumerate(self.layout.hands):
            if ml.native_root_body_names[hand.side] != hand.root_body_name:
                raise ValueError(
                    "source native root reference names a different physical body"
                )
            target = ml.native_root_state_w[hand.side][ids].clone()
            if servo_twist is not None:
                target[:, 7:13] = servo_twist[hand.side][ids]
            target[:, :3] += self._env.scene.env_origins
            pose_support = ml.native_root_pose_support[hand.side][ids]
            linear_support = (
                pose_support & ml.native_root_linear_velocity_support[hand.side][ids]
            )
            angular_support = (
                pose_support & ml.native_root_angular_velocity_support[hand.side][ids]
            )
            target[:, :3] += translation[:, i]
            reference_quat = target[:, 3:7].clone()
            target[:, 3:7] = quat_box_plus(
                reference_quat, rotation[:, i]
            )
            target[:, 7:10] = torch.where(linear_support[:, None], target[:, 7:10], 0.0)
            angular = quat_apply(
                quat_mul(target[:, 3:7], quat_inv(reference_quat)), target[:, 10:13]
            )
            target[:, 10:13] = torch.where(angular_support[:, None], angular, 0.0)
            if self.cfg.wrist_target_twist == "filtered_correction_feedforward":
                correction_support = pose_support & self._correction_interval_valid
                target[:, 7:13] += torch.where(
                    correction_support[:, None], self._correction_twist[:, i], 0.0
                )
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
        self._source_contact_translation[rows] = 0.0
        self._source_contact_part[rows] = -1
        self._source_contact_compression[rows] = 0.0
        self._source_hold_active[rows] = False
        self._source_hold_offset_m[rows] = 0.0
        self._source_hold_force_n[rows] = 0.0
        self._source_transport_part[rows] = -1
        self._idle_support_offset_m[rows] = 0.
        self._correction_twist[rows] = 0.0
        self._correction_interval_valid[rows] = False
        self._installed_valid[rows] = False
        self._finger_requested[rows] = 0.0
        self._finger_requested_support[rows] = False
        self._finger_source_support[rows] = False
