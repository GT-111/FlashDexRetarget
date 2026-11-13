"""Native task termination uses named physical roots/parts and explicit support."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from .motion_tracking import MotionTrackingTerminations


class NativeMotionTrackingTerminations(MotionTrackingTerminations):
    @staticmethod
    def native_reference_diverged(
        env,
        command_name,
        root_position_m,
        root_rotation_rad,
        part_position_m,
        part_rotation_rad,
        grace_steps,
    ):
        command = env.command_manager.get_term(command_name)
        ml, ids = command.motion_lib, command._motion_flat_ids
        roots = command.ref_native_root_state_w
        root_support = torch.stack(
            [ml.native_root_pose_support[s][ids] for s in command._side_list], -1
        )
        root_bad = (roots[..., :3] - command.robot_wrist_trans_w).norm(
            dim=-1
        ) > root_position_m
        root_bad |= (
            quat_error_magnitude(roots[..., 3:7], command.robot_wrist_quat_w)
            > root_rotation_rad
        )
        parts, ref = command.sim_object_part_state_w, command.ref_object_part_state_w
        part_bad = (parts[..., :3] - ref[..., :3]).norm(dim=-1) > part_position_m
        part_bad |= (
            quat_error_magnitude(parts[..., 3:7], ref[..., 3:7]) > part_rotation_rad
        )
        bad = (root_bad & root_support).any(-1)
        bad |= (part_bad & ml.native_object_part_pose_support[ids]).any(-1)
        return bad & (env.episode_length_buf >= grace_steps)

    @staticmethod
    def native_object_dropped(env, command_name, drop_margin_m):
        command = env.command_manager.get_term(command_name)
        body = command.motion_lib.native_object_part_body_names.index(
            command.native_layout.object.root_body_name
        )
        actual = command.sim_object_part_state_w[:, body, 2]
        reference = command.ref_object_part_state_w[:, body, 2]
        support = command.motion_lib.native_object_part_pose_support[
            command._motion_flat_ids, body
        ]
        return support & (actual < reference - drop_margin_m)

    @staticmethod
    def native_velocity_diverged(
        env, command_name, linear_m_per_s, angular_rad_per_s, joint_rad_per_s
    ):
        command = env.command_manager.get_term(command_name)
        parts = command.sim_object_part_state_w
        linear = torch.cat([command.robot_wrist_lin_vel_w, parts[..., 7:10]], 1)
        angular = torch.cat([command.robot_wrist_ang_vel_w, parts[..., 10:13]], 1)
        joints = torch.cat(
            [command.robot.data.joint_vel, command.native_object_joint_vel], -1
        )
        return (
            (linear.norm(dim=-1) > linear_m_per_s).any(-1)
            | (angular.norm(dim=-1) > angular_rad_per_s).any(-1)
            | (joints.abs() > joint_rad_per_s).any(-1)
        )

    @staticmethod
    def native_state_nonfinite(env, command_name):
        command = env.command_manager.get_term(command_name)
        fields = (
            command.robot_wrist_trans_w,
            command.robot_wrist_quat_w,
            command.robot_wrist_lin_vel_w,
            command.robot_wrist_ang_vel_w,
            command.robot.data.joint_pos,
            command.robot.data.joint_vel,
            command.sim_object_part_state_w,
            command.native_object_joint_pos,
            command.native_object_joint_vel,
        )
        return torch.stack(
            [~torch.isfinite(value).flatten(1).all(-1) for value in fields], -1
        ).any(-1)
