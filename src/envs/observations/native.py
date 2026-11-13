"""Native FREE-root / articulated-part observations in the existing obs API."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_box_minus,
    quat_inv,
    quat_mul,
)

from .motion_tracking import MotionTrackingObs


def _parts_in_wrist(command, states, support=None):
    """Part p/R/v/omega relative to each actual physical wrist body."""
    result = []
    for i in range(len(command._side_list)):
        p = command.robot_wrist_trans_w[:, i, None]
        q = command.robot_wrist_quat_w[:, i, None].expand(-1, states.shape[1], -1)
        v = command.robot_wrist_lin_vel_w[:, i, None]
        w = command.robot_wrist_ang_vel_w[:, i, None]
        r = states[..., :3] - p
        local_q = quat_mul(quat_inv(q), states[..., 3:7])
        fields = [
            quat_apply_inverse(q, r).flatten(1),
            matrix_from_quat(local_q)[..., :2].flatten(1),
            quat_apply_inverse(
                q, states[..., 7:10] - v - torch.cross(w.expand_as(r), r, dim=-1)
            ).flatten(1),
            quat_apply_inverse(q, states[..., 10:13] - w).flatten(1),
        ]
        if support is not None:
            fields = [
                torch.where(
                    mask[..., None].expand(*mask.shape, width).flatten(1), value, 0.0
                )
                for value, mask, width in zip(
                    fields, support, (3, 6, 3, 3), strict=True
                )
            ]
        result.extend(fields)
    return torch.cat(result, -1)


class NativeMotionTrackingObs(MotionTrackingObs):
    @staticmethod
    def native_root_reference_error(env, command_name):
        command = env.command_manager.get_term(command_name)
        ref = command.ref_native_root_state_w
        ml, ids = command.motion_lib, command._motion_flat_ids
        pose = torch.stack(
            [ml.native_root_pose_support[s][ids] for s in command._side_list], -1
        )
        linear = (
            torch.stack(
                [
                    ml.native_root_linear_velocity_support[s][ids]
                    for s in command._side_list
                ],
                -1,
            )
            & pose
        )
        angular = (
            torch.stack(
                [
                    ml.native_root_angular_velocity_support[s][ids]
                    for s in command._side_list
                ],
                -1,
            )
            & pose
        )
        fields = [
            (ref[..., :3] - command.robot_wrist_trans_w).flatten(1),
            quat_box_minus(ref[..., 3:7], command.robot_wrist_quat_w).flatten(1),
            (ref[..., 7:10] - command.robot_wrist_lin_vel_w).flatten(1),
            (ref[..., 10:13] - command.robot_wrist_ang_vel_w).flatten(1),
        ]
        return torch.cat(
            [
                torch.where(
                    mask[..., None].expand(*mask.shape, 3).flatten(1), value, 0.0
                )
                for value, mask in zip(
                    fields, (pose, pose, linear, angular), strict=True
                )
            ],
            -1,
        )

    @staticmethod
    def native_object_parts_wrist(env, command_name):
        command = env.command_manager.get_term(command_name)
        return _parts_in_wrist(command, command.sim_object_part_state_w)

    @staticmethod
    def native_reference_parts_wrist(env, command_name):
        command = env.command_manager.get_term(command_name)
        ml, ids = command.motion_lib, command._motion_flat_ids
        pose = ml.native_object_part_pose_support[ids]
        return _parts_in_wrist(
            command,
            command.ref_object_part_state_w,
            (
                pose,
                pose,
                pose & ml.native_object_part_linear_velocity_support[ids],
                pose & ml.native_object_part_angular_velocity_support[ids],
            ),
        )

    @staticmethod
    def native_object_joint_state(env, command_name):
        command = env.command_manager.get_term(command_name)
        return torch.cat(
            [
                command.native_object.data.joint_pos,
                command.native_object.data.joint_vel,
            ],
            -1,
        )

    @staticmethod
    def native_object_joint_reference_error(env, command_name):
        command = env.command_manager.get_term(command_name)
        ml, ids = command.motion_lib, command._motion_flat_ids
        p = ml.native_object_joint_pos[ids] - command.native_object.data.joint_pos
        v = ml.native_object_joint_vel[ids] - command.native_object.data.joint_vel
        return torch.cat(
            [
                torch.where(ml.native_object_joint_position_support[ids], p, 0.0),
                torch.where(ml.native_object_joint_velocity_support[ids], v, 0.0),
            ],
            -1,
        )

    @staticmethod
    def native_reference_support(env, command_name):
        command = env.command_manager.get_term(command_name)
        ml, ids = command.motion_lib, command._motion_flat_ids
        support = [
            ml.native_joint_position_support[ids],
            ml.native_joint_velocity_support[ids],
        ]
        for side in command._side_list:
            support.extend(
                mask[side][ids, None]
                for mask in (
                    ml.native_root_pose_support,
                    ml.native_root_linear_velocity_support,
                    ml.native_root_angular_velocity_support,
                )
            )
        support.extend(
            mask[ids]
            for mask in (
                ml.native_object_part_pose_support,
                ml.native_object_part_linear_velocity_support,
                ml.native_object_part_angular_velocity_support,
                ml.native_object_joint_position_support,
                ml.native_object_joint_velocity_support,
            )
        )
        return torch.cat(support, -1).float()

    @staticmethod
    def native_tip_part_distances(env, command_name):
        command = env.command_manager.get_term(command_name)
        # Per-part identity is retained. Unsigned/winding distance is exactly
        # the configured surface query, never a claim of paired contact force.
        tips = command.robot_tip_trans_w.flatten(1, 2)
        distance, _ = command.sdf_query_parts(tips)
        return distance.flatten(1)
