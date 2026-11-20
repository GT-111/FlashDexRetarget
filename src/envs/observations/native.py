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

from .._common import action_term, contact_interval_weights
from ..object_points import (
    native_part_reference,
    native_part_surface_cloud,
    pack_native_part_surface,
)
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


def build_native_part_point_cloud(relative_part_state, points_body):
    return pack_native_part_surface(*native_part_surface_cloud(relative_part_state, points_body))


def build_native_next_part_point_cloud_delta(
    motion_lib, motion_ids, motion_steps, *, wrist_pose, relative_part_state, points_body,
):
    """Same static samples and exact next source query in live and replay paths."""
    reference = native_part_reference(motion_lib, motion_ids, motion_steps, offset=1)
    zero_rate = wrist_pose.new_zeros(*wrist_pose.shape[:-1], 3)
    pose_support = reference["pose_support"]
    relative_reference = _parts_relative_to_wrist(reference["state"], wrist_pose, zero_rate, zero_rate,
        (pose_support, pose_support, pose_support, pose_support))
    actual, actual_support = native_part_surface_cloud(relative_part_state, points_body)
    target, target_support = native_part_surface_cloud(relative_reference, points_body)
    support = actual_support & target_support & pose_support[:, None]
    return pack_native_part_surface(target - actual, support)


def build_native_future_reference(
    command,
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
    ml = command.motion_lib
    offsets = torch.arange(1, n_future + 1, device=motion_steps.device) * stride
    future_steps = motion_steps[:, None] + offsets
    last = ml._motion_num_frames[motion_ids, None] - 1
    valid = future_steps <= last
    ids = ml.length_starts[motion_ids, None] + torch.minimum(future_steps, last)
    current = ml.native_time_ns[ml.length_starts[motion_ids] + motion_steps]
    rows = []
    for k in range(n_future):
        index = ids[:, k]
        reference = command.command_reference(index)
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
                reference["position"],
                reference["velocity"],
                reference["position_support"],
                reference["velocity_support"],
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
    def native_part_point_cloud_wrist(env, command_name, n_points=128, n_obj_verts=2048):
        """Physical-size samples in each actual wrist, with independent part masks."""
        command = env.command_manager.get_term(command_name)
        points = command.part_surface_points(n_points, n_obj_verts)
        return build_native_part_point_cloud(_parts_in_wrist(command, command.sim_object_part_state_w), points)

    @staticmethod
    def native_next_part_point_cloud_delta_wrist(env, command_name, n_points=128, n_obj_verts=2048):
        """Supported next-reference minus live samples; unavailable tails stay missing."""
        command = env.command_manager.get_term(command_name)
        return build_native_next_part_point_cloud_delta(
            command.motion_lib, command.motion_ids, command.motion_steps,
            wrist_pose=torch.cat((command.robot_wrist_trans_w - env.scene.env_origins[:, None],
                                  command.robot_wrist_quat_w), -1),
            relative_part_state=_parts_in_wrist(command, command.sim_object_part_state_w),
            points_body=command.part_surface_points(n_points, n_obj_verts))

    @staticmethod
    def native_wrist_control_state(env):
        """Prior processed wrist coordinates and pending reset, independent of source support."""
        state = action_term(env).wrist_control_state
        return torch.cat((state["filtered_coordinates"].flatten(1),
                          state["filter_pending"][:, None].float()), -1)

    @staticmethod
    def native_source_hand_part_intent(env, command_name):
        """Current decision's active values and independent knowledge masks."""
        command = env.command_manager.get_term(command_name)
        facts = command.source_hand_part_intent(command._motion_flat_ids)
        if facts is None:
            raise ValueError("source hand/part intent observation requires an explicit source carrier")
        return torch.cat((facts["active"].flatten(1).float(), facts["known"].flatten(1).float()), -1)

    @staticmethod
    @torch.no_grad()
    def native_solved_contact_feedback(env, command_name, hinge_scale_nm=.05, slip_scale_m_s=.1):
        """Previous physical interval in declared pair and scalar-joint order, with support bits."""
        command = env.command_manager.get_term(command_name)
        scene, layout = env.scene, command.native_layout
        interval = scene.hand_part_contact_interval_facts()
        valid, dt = contact_interval_weights(env, interval, interval["interval_complete"])
        complete = (valid & interval["measurement_support"]).all(-1)
        object_weights = {
            obj.entity_name: scene.native_object_weights_n[i].clamp_min(.1)
            for i, obj in enumerate(layout.objects)
        }
        fields = []

        def append(value, supported):
            fields.append(torch.where(supported[:, None], value, 0.0))
            fields.append(supported[:, None].float())

        def weighted_force(pair):
            force = pair["force_on_second_w_n"]
            return (torch.nan_to_num(force) * dt[..., None]).sum(1), torch.isfinite(force).all(dim=(1, 2))

        for name, address in interval["addresses"].items():
            pair = interval["pairs"][name]
            force, supported = weighted_force(pair)
            side = command._side_list.index(address["hand_side"])
            owner = address["part_entity_name"]
            weight = object_weights[owner]
            local = quat_apply_inverse(command.robot_wrist_quat_w[:, side], force)
            normal = pair["normal_force_n"]
            bearing = pair["force_bearing_contact_present"]
            slip = pair["force_bearing_tangent_speed_max_m_s"]
            supported &= (complete & torch.isfinite(normal).all(-1)
                          & (~bearing | torch.isfinite(slip)).all(-1))
            mean_normal = (torch.nan_to_num(normal) * dt).sum(1)
            mean_bearing = (bearing * dt).sum(1)
            max_slip = torch.nan_to_num(slip).amax(1)
            append(torch.cat((torch.tanh(local / weight),
                              torch.tanh(mean_normal / weight)[:, None], mean_bearing[:, None],
                              torch.tanh(max_slip / slip_scale_m_s)[:, None]), -1), supported)
            if "joint_effort_on_second" in pair:
                effort = pair["joint_effort_on_second"]
                columns = command._object_joint_slices[owner]
                units = command.object_joint_coordinate_metadata["position_units"][columns]
                scale = torch.stack([weight.new_tensor(hinge_scale_nm) if unit == "rad" else weight
                                     for unit in units])
                value = (torch.nan_to_num(effort) * dt[..., None]).sum(1)
                append(torch.tanh(value / scale), complete & torch.isfinite(effort).all(dim=(1, 2)))

        support_pairs = scene.support_contact_interval_pairs()
        for name, (obj, _) in layout.support_pair_bindings.items():
            pair = support_pairs.get(name)
            if pair is None:
                append(dt.new_zeros(env.num_envs, 1), torch.zeros_like(complete))
            else:
                force, supported = weighted_force(pair)
                upward = (force @ scene.gravity_up_w) / object_weights[obj.entity_name]
                append(torch.tanh(upward[:, None]), complete & supported)
        for name, (first, second) in layout.object_pair_bindings.items():
            force, supported = weighted_force(support_pairs[name])
            scale = torch.minimum(object_weights[first.entity_name], object_weights[second.entity_name])
            append(torch.tanh((force @ scene.gravity_up_w / scale)[:, None]), complete & supported)
        return torch.cat(fields, -1)

    @staticmethod
    @torch.no_grad()
    def native_source_tip_reference_error(env, command_name):
        """Current decision goal: supported source-tip error in actual wrist axes.

        Concatenated actual point errors precede their masks. This is separate
        from the consumed endpoint packet used by reward/recording.
        """
        command = env.command_manager.get_term(command_name)
        reference = command.source_point_reference(command._motion_flat_ids)
        errors, support = [], []
        for index, side in enumerate(command._side_list):
            actual = command.robot.data.site_pos_w[:, command._tip_site_ids[side]]
            mask = reference["position_support"][side] & torch.isfinite(actual).all(-1)
            delta = torch.where(mask[..., None], reference["position_w"][side] - actual, 0.0)
            quat = command.robot_wrist_quat_w[:, index, None].expand(-1, actual.shape[1], -1)
            errors.append(quat_apply_inverse(quat, delta))
            support.append(mask)
        return torch.cat((torch.cat(errors, 1).flatten(1), torch.cat(support, 1).float()), -1)

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
        return torch.cat(
            [
                torch.where(
                    ref["position_support"],
                    ref["position"] - command.project_command_state(command.robot.data.joint_pos),
                    0.0,
                ),
                torch.where(
                    ref["velocity_support"],
                    ref["velocity"] - command.project_command_state(command.robot.data.joint_vel),
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
            command,
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
            command_position=command.project_command_state(command.robot.data.joint_pos),
            command_velocity=command.project_command_state(command.robot.data.joint_vel),
            object_joint_position=command.native_object_joint_pos,
            object_joint_velocity=command.native_object_joint_vel,
            object_joint_columns=command._object_joint_columns,
            n_future=n_future,
            stride=stride,
        )
