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

from .._common import action_term
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
    def native_installed_command_error(env, command_name):
        command = env.command_manager.get_term(command_name)
        installed = action_term(env).installed_control
        error = (
            installed["finger_target_rad"]
            - command.robot.data.joint_pos[:, command.native_command_joint_ids]
        )
        return torch.where(installed["installed_valid"][:, None], error, 0.0)

    @staticmethod
    def native_command_reference_error(env, command_name):
        command = env.command_manager.get_term(command_name)
        ref = command.command_reference()
        columns = command.native_command_joint_ids
        return torch.cat(
            [
                torch.where(
                    ref["position_support"],
                    ref["position"] - command.robot.data.joint_pos[:, columns],
                    0.0,
                ),
                torch.where(
                    ref["velocity_support"],
                    ref["velocity"] - command.robot.data.joint_vel[:, columns],
                    0.0,
                ),
            ],
            -1,
        )

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
                command.native_object_joint_pos,
                command.native_object_joint_vel,
            ],
            -1,
        )

    @staticmethod
    def native_object_joint_reference_error(env, command_name):
        command = env.command_manager.get_term(command_name)
        ref = command.object_joint_reference()
        p = ref["position"] - command.native_object_joint_pos
        v = ref["velocity"] - command.native_object_joint_vel
        return torch.cat(
            [
                torch.where(ref["position_support"], p, 0.0),
                torch.where(ref["velocity_support"], v, 0.0),
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
            )
        )
        joints = command.object_joint_reference()
        support.extend((joints["position_support"], joints["velocity_support"]))
        return torch.cat(support, -1).float()

    @staticmethod
    def native_tip_part_distances(env, command_name):
        command = env.command_manager.get_term(command_name)
        # Per-part identity is retained. Unsigned/winding distance is exactly
        # the configured surface query, never a claim of paired contact force.
        tips = command.robot_tip_trans_w.flatten(1, 2)
        distance, _ = command.sdf_query_parts(tips)
        return distance.flatten(1)

    @staticmethod
    def native_tip_part_normals_wrist(env, command_name):
        command = env.command_manager.get_term(command_name)
        tips = command.robot_tip_trans_w
        _, gradient = command.sdf_query_parts(tips.flatten(1, 2))
        gradient = gradient.reshape(*tips.shape[:3], gradient.shape[-2], 3)
        rotation = command.robot_wrist_quat_w[:, :, None, None].expand(
            *gradient.shape[:-1], 4
        )
        return quat_apply_inverse(rotation, gradient).flatten(1)

    @staticmethod
    def native_future_reference(env, command_name, n_future, stride):
        """Named native lookahead in the current physical wrist chart.

        Only real prepared endpoints have support. A clipped tail is zero and
        unknown; it does not acquire an invented source timestamp or velocity.
        """
        command = env.command_manager.get_term(command_name)
        ml = command.motion_lib
        offsets = torch.arange(1, n_future + 1, device=env.device) * stride
        ids = command._future_motion_flat_ids(offsets)
        valid = (
            command.motion_steps[:, None] + offsets
            <= command.motion_num_frames[:, None] - 1
        )
        current = ml.native_time_ns[command._motion_flat_ids]
        rows = []
        for k in range(n_future):
            index = ids[:, k]
            fields = []
            for side_index, side in enumerate(command._side_list):
                target = ml.native_root_state_w[side][index]
                pose = ml.native_root_pose_support[side][index]
                linear = ml.native_root_linear_velocity_support[side][index] & pose
                angular = ml.native_root_angular_velocity_support[side][index] & pose
                position = (
                    target[:, :3]
                    + env.scene.env_origins
                    - command.robot_wrist_trans_w[:, side_index]
                )
                values = (
                    position,
                    quat_box_minus(
                        target[:, 3:7], command.robot_wrist_quat_w[:, side_index]
                    ),
                    target[:, 7:10] - command.robot_wrist_lin_vel_w[:, side_index],
                    target[:, 10:13] - command.robot_wrist_ang_vel_w[:, side_index],
                )
                fields.extend(
                    torch.where(mask[:, None], value, 0.0)
                    for mask, value in zip(
                        (pose, pose, linear, angular), values, strict=True
                    )
                )
                fields.extend(mask[:, None].float() for mask in (pose, linear, angular))
            for reference, position, velocity in (
                (
                    command.command_reference(index),
                    command.robot.data.joint_pos[:, command.native_command_joint_ids],
                    command.robot.data.joint_vel[:, command.native_command_joint_ids],
                ),
                (
                    command.object_joint_reference(index),
                    command.native_object_joint_pos,
                    command.native_object_joint_vel,
                ),
            ):
                fields.extend(
                    (
                        torch.where(
                            reference["position_support"],
                            reference["position"] - position,
                            0.0,
                        ),
                        torch.where(
                            reference["velocity_support"],
                            reference["velocity"] - velocity,
                            0.0,
                        ),
                        reference["position_support"].float(),
                        reference["velocity_support"].float(),
                    )
                )
            pose = ml.native_object_part_pose_support[index]
            linear = ml.native_object_part_linear_velocity_support[index] & pose
            angular = ml.native_object_part_angular_velocity_support[index] & pose
            fields.append(
                _parts_in_wrist(
                    command,
                    command.object_part_reference(index),
                    (pose, pose, linear, angular),
                )
            )
            fields.extend(mask.float() for mask in (pose, linear, angular))
            fields.append(
                ((ml.native_time_ns[index] - current).double() * 1e-9)[:, None].float()
            )
            row = torch.cat(fields, -1)
            rows.append(torch.where(valid[:, k, None], row, 0.0))
        return torch.stack(rows, 1).flatten(1)
