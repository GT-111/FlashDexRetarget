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


def _parts_relative_to_wrist(
    states, wrist_pose, wrist_linear_velocity, wrist_angular_velocity, support=None
):
    """Part p/R/v/omega in each wrist chart, with any common position origin."""
    result = []
    for i in range(wrist_pose.shape[1]):
        p = wrist_pose[:, i, None, :3]
        q = wrist_pose[:, i, None, 3:7].expand(-1, states.shape[1], -1)
        v = wrist_linear_velocity[:, i, None]
        w = wrist_angular_velocity[:, i, None]
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


def _parts_in_wrist(command, states, support=None):
    """Part state relative to the current physical wrist bodies."""
    return _parts_relative_to_wrist(
        states,
        torch.cat((command.robot_wrist_trans_w, command.robot_wrist_quat_w), -1),
        command.robot_wrist_lin_vel_w,
        command.robot_wrist_ang_vel_w,
        support,
    )


def build_native_future_reference(
    motion_lib,
    motion_ids,
    motion_steps,
    *,
    sides,
    wrist_pose,
    wrist_linear_velocity,
    wrist_angular_velocity,
    command_position,
    command_velocity,
    object_joint_position,
    object_joint_velocity,
    command_joint_ids,
    object_joint_columns,
    n_future,
    stride,
):
    """Build live or replay lookahead from factual state and immutable references.

    Wrist positions and reference positions share the scene-local origin;
    twists retain WORLD axes. Batch size comes from the supplied state, never
    from live worlds. ``motion_steps`` is the observation's own command phase.
    Native support, column permutations and int64 query clocks stay independent.
    """
    ml = motion_lib
    offsets = torch.arange(1, n_future + 1, device=motion_steps.device) * stride
    future_steps = motion_steps[:, None] + offsets
    last = ml._motion_num_frames[motion_ids, None] - 1
    valid = future_steps <= last
    ids = ml.length_starts[motion_ids, None] + torch.minimum(future_steps, last)
    current = ml.native_time_ns[ml.length_starts[motion_ids] + motion_steps]
    rows = []
    for k in range(n_future):
        index = ids[:, k]
        fields = []
        for side_index, side in enumerate(sides):
            target = ml.native_root_state_w[side][index]
            pose = ml.native_root_pose_support[side][index]
            linear = ml.native_root_linear_velocity_support[side][index] & pose
            angular = ml.native_root_angular_velocity_support[side][index] & pose
            values = (
                target[:, :3] - wrist_pose[:, side_index, :3],
                quat_box_minus(target[:, 3:7], wrist_pose[:, side_index, 3:7]),
                target[:, 7:10] - wrist_linear_velocity[:, side_index],
                target[:, 10:13] - wrist_angular_velocity[:, side_index],
            )
            fields.extend(
                torch.where(mask[:, None], value, 0.0)
                for mask, value in zip(
                    (pose, pose, linear, angular), values, strict=True
                )
            )
            fields.extend(mask[:, None].float() for mask in (pose, linear, angular))
        for position, velocity, position_support, velocity_support, actual_q, actual_v in (
            (
                ml.robot_joint_pos[index][:, command_joint_ids],
                ml.robot_joint_vel[index][:, command_joint_ids],
                ml.native_joint_position_support[index][:, command_joint_ids],
                ml.native_joint_velocity_support[index][:, command_joint_ids],
                command_position,
                command_velocity,
            ),
            (
                ml.native_object_joint_pos[index][:, object_joint_columns],
                ml.native_object_joint_vel[index][:, object_joint_columns],
                ml.native_object_joint_position_support[index][:, object_joint_columns],
                ml.native_object_joint_velocity_support[index][:, object_joint_columns],
                object_joint_position,
                object_joint_velocity,
            ),
        ):
            fields.extend(
                (
                    torch.where(position_support, position - actual_q, 0.0),
                    torch.where(velocity_support, velocity - actual_v, 0.0),
                    position_support.float(),
                    velocity_support.float(),
                )
            )
        pose = ml.native_object_part_pose_support[index]
        linear = ml.native_object_part_linear_velocity_support[index] & pose
        angular = ml.native_object_part_angular_velocity_support[index] & pose
        fields.append(
            _parts_relative_to_wrist(
                ml.native_object_part_state_w[index],
                wrist_pose,
                wrist_linear_velocity,
                wrist_angular_velocity,
                (pose, pose, linear, angular),
            )
        )
        fields.extend(mask.float() for mask in (pose, linear, angular))
        # Subtract exact clocks before converting to seconds, including nonzero origins.
        fields.append(((ml.native_time_ns[index] - current).double() * 1e-9)[:, None].float())
        rows.append(torch.where(valid[:, k, None], torch.cat(fields, -1), 0.0))
    return torch.stack(rows, 1).flatten(1)


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
        return build_native_future_reference(
            command.motion_lib,
            command.motion_ids,
            command.motion_steps,
            sides=command._side_list,
            wrist_pose=torch.cat(
                (
                    command.robot_wrist_trans_w - env.scene.env_origins[:, None],
                    command.robot_wrist_quat_w,
                ),
                -1,
            ),
            wrist_linear_velocity=command.robot_wrist_lin_vel_w,
            wrist_angular_velocity=command.robot_wrist_ang_vel_w,
            command_position=command.robot.data.joint_pos[:, command.native_command_joint_ids],
            command_velocity=command.robot.data.joint_vel[:, command.native_command_joint_ids],
            object_joint_position=command.native_object_joint_pos,
            object_joint_velocity=command.native_object_joint_vel,
            command_joint_ids=command.native_command_joint_ids,
            object_joint_columns=command._object_joint_columns,
            n_future=n_future,
            stride=stride,
        )
