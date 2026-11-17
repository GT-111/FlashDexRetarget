"""Supported native root/part objectives; no object assistance or force proxy."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_error_magnitude

from .._common import action_term
from . import _SIDES
from .motion_tracking import MotionTrackingRewards, _worst_k_mean


def _supported_mean(value, support):
    return torch.where(support, value, 0.0).sum(-1) / support.sum(-1).clamp_min(1)


def _source_geometry_mean(value, support, metadata, side):
    """Use the existing builder's half-weight sides and one common denominator."""
    count = support.sum(-1).clamp_min(1)
    if side is None:
        return torch.where(support, value, 0.0).sum(-1) / count
    if side not in metadata["hand_sides"]:
        return torch.zeros_like(count, dtype=value.dtype)
    index = metadata["hand_sides"].index(side)
    begin, end = metadata["hand_offsets"][index:index + 2]
    own = torch.where(support[:, begin:end], value[:, begin:end], 0.0).sum(-1)
    return len(_SIDES) * own / count


def _source_intent_mean(value, required, metadata, side):
    """Keep one source-required-hand mass under the existing side split."""
    count = required.sum(-1).clamp_min(1)
    value = torch.where(required, value, 0.0)
    if side is None:
        return value.sum(-1) / count
    if side not in metadata["hand_sides"]:
        return torch.zeros_like(count, dtype=value.dtype)
    index = metadata["hand_sides"].index(side)
    return len(_SIDES) * value[:, index] / count


class NativeMotionTrackingRewards(MotionTrackingRewards):
    @staticmethod
    def native_part_surface_displacement(
        env, command_name, scale, n_points=128, n_obj_verts=2048,
        topk=0, topk_frac=1.0, deadband=0.0, norm_radius=0.0, part_reduce="mean",
    ):
        """Shared supported surface displacement; radius scales only rotation leverage."""
        if (not all(math.isfinite(value) for value in (scale, topk_frac, deadband, norm_radius))
                or scale <= 0 or not 0 < topk_frac <= 1 or topk < 0 or deadband < 0 or norm_radius < 0
                or part_reduce not in ("mean", "worst")):
            raise ValueError("native surface reward requires finite scales, valid pooling and mean/worst part reduction")
        command = env.command_manager.get_term(command_name)
        installed = action_term(env).installed_control
        reference = command.part_surface_reference(
            motion_ids=installed["reference_motion_ids"], motion_steps=installed["reference_motion_steps"])
        current, target = command.sim_object_part_state_w, reference["state"].clone()
        target[..., :3] += env.scene.env_origins[:, None]
        points = command.part_surface_points(n_points, n_obj_verts)
        if norm_radius > 0:
            points = points * (norm_radius / points.norm(dim=-1).amax(-1).clamp_min(1e-6))[:, None, None]
        delta_rotation = matrix_from_quat(target[..., 3:7]) - matrix_from_quat(current[..., 3:7])
        displacement = (target[..., :3] - current[..., :3])[..., None, :] + torch.einsum(
            "bpij,pnj->bpni", delta_rotation, points)
        error = (_worst_k_mean(displacement.norm(dim=-1), topk, topk_frac) - deadband).clamp_min(0.)
        support = reference["pose_support"] & installed["installed_valid"][:, None]
        value = torch.exp(-scale * error)
        if part_reduce == "mean":
            return _supported_mean(value, support)
        return torch.where(support.any(-1), torch.where(support, value, 1.).amin(-1), 0.)

    @staticmethod
    def native_source_hand_part_proximity(env, command_name, distance_scale_m, side=None):
        """Source-intended surface approach; SDF proximity is never solved contact."""
        if not math.isfinite(distance_scale_m) or distance_scale_m <= 0:
            raise ValueError("source hand-part proximity requires a positive finite distance scale")
        command = env.command_manager.get_term(command_name)
        intent = command.source_hand_part_intent_facts()
        geometry = command.source_tip_geometry_facts()
        if intent is None or geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        metadata, points = intent["metadata"], geometry["metadata"]
        execution = intent["execution_valid"] & geometry["execution_valid"]
        for key in ("reference_motion_ids", "reference_motion_steps", "reference_time_ns"):
            execution &= intent[key] == geometry[key]
        active = intent["active"] & intent["known"]
        required = active.any(-1) & execution[:, None]
        parts = [points["part_body_names"].index(name) for name in metadata["part_body_names"]]
        values = []
        for hand, name in enumerate(metadata["hand_sides"]):
            index = points["hand_sides"].index(name)
            begin, end = points["hand_offsets"][index:index + 2]
            if begin == end:
                values.append(torch.zeros(env.num_envs, device=env.device))
                continue
            distance = geometry["live_part_distance_m"][:, begin:end, parts]
            support = (geometry["live_part_query_support"][:, begin:end, parts]
                       & geometry["achieved_position_support"][:, begin:end, None]
                       & geometry["live_part_pose_support"][:, None, parts]
                       & active[:, hand, None] & execution[:, None, None]
                       & torch.isfinite(distance))
            closest = torch.where(support, distance.abs(), float("inf")).flatten(1).amin(-1)
            values.append(torch.exp(-closest / distance_scale_m))
        return _source_intent_mean(torch.stack(values, -1), required, metadata, side)

    @staticmethod
    def native_source_hand_part_contact(
        env, command_name, force_scale_n, separation_bound_m, side=None
    ):
        """Bounded solved-force credit on installed source candidates, not a task score."""
        if (not math.isfinite(force_scale_n) or force_scale_n <= 0
                or not math.isfinite(separation_bound_m) or separation_bound_m < 0):
            raise ValueError("source hand-part contact requires a positive force scale and nonnegative separation bound")
        command = env.command_manager.get_term(command_name)
        intent = command.source_hand_part_intent_facts()
        interval = env.scene.hand_part_contact_interval_facts()
        if intent is None or interval is None:
            return torch.zeros(env.num_envs, device=env.device)
        active = intent["active"] & intent["known"]
        required = active.any(-1) & intent["execution_valid"][:, None]
        # The manager finishes this interval before reward and resets afterwards.
        endpoint = interval["sample_end_time_s"][:, -1] == env.scene.physics_time_s()
        valid = (interval["sample_recorded"] & interval["clock_support"]
                 & interval["interval_complete"][:, None] & endpoint[:, None]
                 & intent["execution_valid"][:, None])
        duration = torch.where(valid, interval["sample_end_time_s"] - interval["sample_begin_time_s"], 0.0)
        total = duration.sum(-1, keepdim=True)
        weights = duration / torch.where(total > 0, total, torch.ones_like(total))
        addresses = {(row["hand_side"], row["hand_body_name"], row["part_entity_name"], row["part_body_name"]): name
                     for name, row in interval["addresses"].items()}
        hands = {hand.side: hand.root_body_name for hand in command.native_layout.hands}
        owners = {part.body_name: obj.entity_name for obj, part in command.native_layout.part_bindings}
        values = []
        for hand, name in enumerate(intent["metadata"]["hand_sides"]):
            credit = torch.zeros_like(duration, dtype=torch.float32)
            for part, body in enumerate(intent["metadata"]["part_body_names"]):
                address = (name, hands[name], owners[body], body)
                if address not in addresses:
                    continue  # No measurement for this actual pair; no contact credit.
                pair = interval["pairs"][addresses[address]]
                force, present, distance = (pair[key] for key in (
                    "normal_force_n", "force_bearing_contact_present", "force_bearing_distance_max_m"))
                known = (valid & interval["measurement_support"] & torch.isfinite(force)
                         & (force >= 0) & (~present | (torch.isfinite(distance) & (distance >= 0))))
                eligible = (known & present & (distance <= separation_bound_m)
                            & active[:, hand, part, None])
                bounded = (torch.where(known, force, 0.0) / force_scale_n).clamp(0.0, 1.0)
                credit = torch.maximum(credit, torch.where(eligible, bounded, 0.0))
            values.append((credit * weights.to(credit.dtype)).sum(-1))
        return _source_intent_mean(torch.stack(values, -1), required, intent["metadata"], side)

    @staticmethod
    def native_source_tip_position(env, command_name, scale, side=None):
        """Supported source-tip position mechanism, not a target-Q/BC label."""
        facts = env.command_manager.get_term(command_name).source_tip_geometry_facts()
        if facts is None:
            raise ValueError("native source geometry requires named source-point position support")
        support = (facts["source_position_support"] & facts["achieved_position_support"]
                   & facts["execution_valid"][:, None])
        delta = torch.where(support[..., None], facts["achieved_position_w"] - facts["source_position_w"], 0.0)
        value = torch.exp(-scale * delta.norm(dim=-1))
        return _source_geometry_mean(value, support, facts["metadata"], side)

    @staticmethod
    def native_source_tip_object_bound(env, command_name, sigma_m, side=None):
        """Declared part-SDF proximity mechanism; never paired contact/penetration."""
        command = env.command_manager.get_term(command_name)
        if len({part.distance_sign for _, part in command.native_layout.part_bindings}) != 1:
            raise ValueError("native source object bound requires compatible part distance signs")
        facts = command.source_tip_geometry_facts()
        if facts is None:
            raise ValueError("native source geometry requires named source-point position support")
        # Minima must describe the SAME admitted named-part subset for each point.
        common = (facts["desired_part_query_support"] & facts["live_part_query_support"]
                  & facts["desired_part_pose_support"][:, None] & facts["live_part_pose_support"][:, None])
        support = (common.any(-1) & facts["source_position_support"]
                   & facts["achieved_position_support"] & facts["execution_valid"][:, None])
        desired = torch.where(common, facts["desired_part_distance_m"], float("inf")).amin(-1)
        actual = torch.where(common, facts["live_part_distance_m"], float("inf")).amin(-1)
        excess = torch.where(support, actual - desired, 0.0).clamp_min(0.0)
        value = torch.exp(-excess / sigma_m)
        return _source_geometry_mean(value, support, facts["metadata"], side)

    @staticmethod
    def native_command_bc(env, command_name, scale, side=None):
        command = env.command_manager.get_term(command_name)
        term = action_term(env)
        installed = term.installed_control
        if term.ref_joint_pos_used is None:
            return torch.zeros(env.num_envs, device=env.device)
        reference = term.ref_joint_pos_used[:, command.native_command_joint_ids]
        limits = term.finger_command_limits
        error = (
            0.5
            * (installed["finger_target_rad"] - reference).square()
            / (limits[:, 1] - limits[:, 0])
        )
        value = torch.exp(-scale * error)
        support = installed["finger_source_support"] & installed["installed_valid"][:, None]
        if side is None:
            return _supported_mean(value, support)
        # Existing split_sides divides the term weight across declared hands.
        # Retain the global support denominator, including unequal/missing support.
        own = NativeMotionTrackingRewards._side_action_mask(env, side)[term._finger_slice]
        contribution = torch.where(support & own, value, 0.0).sum(-1)
        return len(command._side_list) * contribution / support.sum(-1).clamp_min(1)

    @staticmethod
    def native_root_pose(env, command_name, position_scale_m, rotation_scale_rad, side=None):
        command = env.command_manager.get_term(command_name)
        ref = command.ref_native_root_state_w
        position = (ref[..., :3] - command.robot_wrist_trans_w).square().sum(-1)
        rotation = quat_error_magnitude(
            ref[..., 3:7], command.robot_wrist_quat_w
        ).square()
        value = torch.exp(
            -position / position_scale_m**2 - rotation / rotation_scale_rad**2
        )
        ml, ids = command.motion_lib, command._motion_flat_ids
        support = torch.stack(
            [ml.native_root_pose_support[s][ids] for s in command._side_list], -1
        )
        if side is None:
            return _supported_mean(value, support)
        own = command._side_list.index(side)
        contribution = torch.where(support[:, own], value[:, own], 0.0)
        return len(command._side_list) * contribution / support.sum(-1).clamp_min(1)

    @staticmethod
    def action_penalty(env, command_name, side=None):
        actions = env.action_manager.action
        if side is None:
            return actions.square().mean(-1)
        command = env.command_manager.get_term(command_name)
        own = NativeMotionTrackingRewards._side_action_mask(env, side)
        # Use the full action-count denominator even when hand widths differ.
        return len(command._side_list) * actions[:, own].square().sum(-1) / actions.shape[-1]

    @staticmethod
    def native_object_part_pose(
        env, command_name, position_scale_m, rotation_scale_rad
    ):
        command = env.command_manager.get_term(command_name)
        current, ref = command.sim_object_part_state_w, command.ref_object_part_state_w
        position = (ref[..., :3] - current[..., :3]).square().sum(-1)
        rotation = quat_error_magnitude(ref[..., 3:7], current[..., 3:7]).square()
        value = torch.exp(
            -position / position_scale_m**2 - rotation / rotation_scale_rad**2
        )
        support = command.motion_lib.native_object_part_pose_support[
            command._motion_flat_ids
        ]
        return _supported_mean(value, support)

    @staticmethod
    def native_object_articulation(env, command_name, scale_rad, scale_m=None):
        command = env.command_manager.get_term(command_name)
        ref = command.object_joint_reference()
        error = ref["position"] - command.native_object_joint_pos
        value = torch.empty_like(error)
        for unit, scale in (("rad", scale_rad), ("m", scale_m)):
            columns = command.object_joint_columns_by_unit[unit]
            if columns.numel() == 0:
                continue
            if unit == "m" and (scale is None or not math.isfinite(scale) or scale <= 0):
                raise ValueError("native articulation requires a finite positive scale_m for slide joints")
            value[:, columns] = torch.exp(-error[:, columns].square() / scale**2)
        return _supported_mean(value, ref["position_support"])
