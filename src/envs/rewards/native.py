"""Supported native root/part objectives; no object assistance or force proxy."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from .._common import action_term
from .motion_tracking import MotionTrackingRewards


def _supported_mean(value, support):
    return torch.where(support, value, 0.0).sum(-1) / support.sum(-1).clamp_min(1)


class NativeMotionTrackingRewards(MotionTrackingRewards):
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
    def native_object_articulation(env, command_name, scale_rad):
        command = env.command_manager.get_term(command_name)
        ref = command.object_joint_reference()
        error = ref["position"] - command.native_object_joint_pos
        return _supported_mean(
            torch.exp(-error.square() / scale_rad**2),
            ref["position_support"],
        )
