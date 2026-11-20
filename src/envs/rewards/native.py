"""Supported native root/part objectives; no object assistance or force proxy."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_error_magnitude

from .._common import action_term, contact_interval_weights
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


def _matching_source_phase(intent, geometry):
    execution = intent["execution_valid"] & geometry["execution_valid"]
    for key in ("reference_motion_ids", "reference_motion_steps", "reference_time_ns"):
        execution &= intent[key] == geometry[key]
    return execution


def _hand_part_distances(geometry, hand, parts, source_digit_distance_m):
    metadata = geometry["metadata"]
    index = metadata["hand_sides"].index(hand)
    begin, end = metadata["hand_offsets"][index:index + 2]
    columns = [metadata["part_body_names"].index(part) for part in parts]

    def supported_distances(state):
        distance = geometry[f"{state}_part_distance_m"][:, begin:end, columns]
        support = geometry[f"{state}_part_query_support"][:, begin:end, columns]
        return torch.where(support & torch.isfinite(distance), distance.abs(), float("inf"))

    live = supported_distances("live")
    unavailable = live.new_full((live.shape[0], len(parts)), float("inf"))
    if begin == end:
        return unavailable
    if source_digit_distance_m is None:
        return live.amin(1)
    names = metadata["source_point_names"][begin:end]
    thumb = [i for i, name in enumerate(names) if name in {"thumb_tip", "thumb-tip"}]
    others = [i for i in range(len(names)) if i not in thumb]
    if not thumb or not others:
        return unavailable
    source = supported_distances("desired")
    source_groups = torch.stack((source[:, thumb].amin(1), source[:, others].amin(1)), dim=1)
    live_groups = torch.stack((live[:, thumb].amin(1), live[:, others].amin(1)), dim=1)
    required = source_groups <= source_digit_distance_m
    known = torch.isfinite(source_groups).all(1) & required.any(1)
    return torch.where(known, torch.where(required, live_groups, 0.0).amax(1), unavailable)


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
    def native_source_hand_part_proximity(
        env, command_name, distance_scale_m, side=None, source_digit_distance_m=None,
    ):
        """Source-intended surface approach; SDF proximity is never solved contact."""
        if (not math.isfinite(distance_scale_m) or distance_scale_m <= 0
                or source_digit_distance_m is not None and (not math.isfinite(source_digit_distance_m)
                                                            or source_digit_distance_m <= 0)):
            raise ValueError("source hand-part proximity requires a positive finite distance scale")
        command = env.command_manager.get_term(command_name)
        intent = command.source_hand_part_intent_facts()
        geometry = command.source_tip_geometry_facts()
        if intent is None or geometry is None:
            return torch.zeros(env.num_envs, device=env.device)
        metadata = intent["metadata"]
        execution = _matching_source_phase(intent, geometry)
        active = intent["active"] & intent["known"]
        required = active.any(-1) & execution[:, None]
        values = []
        for hand, name in enumerate(metadata["hand_sides"]):
            distance = _hand_part_distances(geometry, name, metadata["part_body_names"], source_digit_distance_m)
            closest = torch.where(active[:, hand], distance, float("inf")).amin(-1)
            values.append(torch.exp(-closest / distance_scale_m))
        return _source_intent_mean(torch.stack(values, -1), required, metadata, side)

    @staticmethod
    def native_source_hand_part_contact(
        env, command_name, force_scale_n, separation_bound_m, side=None,
        source_digit_distance_m=None, digit_contact_distance_m=0.028, unmatched_digit_scale=0.15,
    ):
        """Bounded solved-force credit across objects required in the same substep."""
        if (not math.isfinite(force_scale_n) or force_scale_n <= 0
                or not math.isfinite(separation_bound_m) or separation_bound_m < 0
                or not math.isfinite(digit_contact_distance_m) or digit_contact_distance_m <= 0
                or not math.isfinite(unmatched_digit_scale) or not 0.0 <= unmatched_digit_scale <= 1.0
                or source_digit_distance_m is not None and (not math.isfinite(source_digit_distance_m)
                                                            or source_digit_distance_m <= 0)):
            raise ValueError("source hand-part contact requires a positive force scale and nonnegative separation bound")
        command = env.command_manager.get_term(command_name)
        intent = command.source_hand_part_intent_facts()
        interval = env.scene.hand_part_contact_interval_facts()
        if intent is None or interval is None:
            return torch.zeros(env.num_envs, device=env.device)
        geometry = None
        execution = intent["execution_valid"]
        if source_digit_distance_m is not None:
            geometry = command.source_tip_geometry_facts()
            if geometry is None:
                return torch.zeros(env.num_envs, device=env.device)
            execution = _matching_source_phase(intent, geometry)
        active = intent["active"] & intent["known"]
        required = active.any(-1) & execution[:, None]
        # The manager finishes this interval before reward and resets afterwards.
        valid, weights = contact_interval_weights(env, interval, execution)
        addresses = {(row["hand_side"], row["hand_body_name"], row["part_entity_name"], row["part_body_name"]): name
                     for name, row in interval["addresses"].items()}
        hands = {hand.side: hand.root_body_name for hand in command.native_layout.hands}
        owners = {part.body_name: obj.entity_name for obj, part in command.native_layout.part_bindings}
        values = []
        metadata = intent["metadata"]
        part_columns = {}
        for index, body in enumerate(metadata["part_body_names"]):
            part_columns.setdefault(owners[body], []).append(index)
        for hand, name in enumerate(metadata["hand_sides"]):
            by_object = {owner: torch.zeros_like(weights) for owner in part_columns}
            if geometry is not None:
                digit_distance = _hand_part_distances(geometry, name, metadata["part_body_names"], source_digit_distance_m)
                digit_scale = torch.where(digit_distance <= digit_contact_distance_m, 1.0, unmatched_digit_scale)
                digit_scale = torch.where(torch.isfinite(digit_distance), digit_scale, 0.0)
            for part, body in enumerate(metadata["part_body_names"]):
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
                if geometry is not None:
                    bounded = bounded * digit_scale[:, part, None]
                owner = owners[body]
                by_object[owner] = torch.maximum(by_object[owner], torch.where(eligible, bounded, 0.0))
            # Alternate parts of one object can serve the same grasp. Independent
            # objects must be loaded together in the actual solver interval.
            credit = torch.stack([
                torch.where(active[:, hand, columns].any(-1, keepdim=True), by_object[owner], 1.0)
                for owner, columns in part_columns.items()
            ], dim=-1).amin(-1)
            values.append((credit * weights).sum(-1))
        return _source_intent_mean(torch.stack(values, -1), required, intent["metadata"], side)

    @staticmethod
    def native_source_lift_load(env, command_name, lookahead_s, minimum_speed_m_s, separation_bound_m,
                                rotation_scale_rad=.35):
        """Credit simultaneous upward load that preserves the source orientation during lift."""
        command = env.command_manager.get_term(command_name)
        intent = command.source_hand_part_intent_facts()
        interval = env.scene.hand_part_contact_interval_facts()
        if intent is None or interval is None:
            return torch.zeros(env.num_envs, device=env.device)
        up, weight = env.scene.gravity_up_w, env.scene.native_object_weights_n
        objects = command.native_layout.objects
        metadata = intent["metadata"]
        columns = {name: index for index, name in enumerate(metadata["part_body_names"])}
        root_columns = [columns[obj.root_body_name] for obj in objects]
        query = dict(motion_ids=intent["reference_motion_ids"], motion_steps=intent["reference_motion_steps"])
        now = command.part_surface_reference(**query)
        later = command.part_surface_reference(**query, offset=max(1, round(lookahead_s / env.step_dt)))
        elapsed = (later["reference_time_ns"] - now["reference_time_ns"]) * 1e-9
        rise = (later["state"][:, root_columns, :3] - now["state"][:, root_columns, :3]) @ up
        demand = ((rise > minimum_speed_m_s * elapsed[:, None]) & (elapsed[:, None] > 0)
                  & now["pose_support"][:, root_columns] & later["pose_support"][:, root_columns]
                  & (weight > 0))
        ids = command.motion_lib.length_starts[intent["reference_motion_ids"]] + intent["reference_motion_steps"]
        phase = command.task_phase_reference(ids)
        if phase is not None:
            demand &= ~phase["lift_declared"] | phase["lift"]
        active = intent["active"] & intent["known"]
        valid, weights = contact_interval_weights(env, interval, intent["execution_valid"])
        complete = (valid & interval["measurement_support"]).all(-1)
        object_load = {obj.entity_name: torch.zeros_like(weights) for obj in objects}
        object_known = {obj.entity_name: torch.ones_like(valid) for obj in objects}
        sides = {side: i for i, side in enumerate(metadata["hand_sides"])}
        hands = {hand.side: hand.root_body_name for hand in command.native_layout.hands}
        owners = {part.body_name: obj.entity_name for obj, part in command.native_layout.part_bindings}
        for name, address in interval["addresses"].items():
            side, body = address["hand_side"], address["part_body_name"]
            owner = address["part_entity_name"]
            if hands.get(side) != address["hand_body_name"] or owners.get(body) != owner:
                continue
            assigned = active[:, sides[side], columns[body], None]
            pair = interval["pairs"][name]
            force = pair["force_on_second_w_n"] @ up
            present = pair["force_bearing_contact_present"]
            distance = pair["force_bearing_distance_max_m"]
            known = torch.isfinite(force) & (~present | torch.isfinite(distance))
            object_known[owner] &= ~assigned | known
            object_load[owner] += torch.where(assigned & known & present & (distance <= separation_bound_m), force, 0.)
        required = demand & torch.stack([
            active[..., [columns[part.body_name] for part in obj.parts]].any(dim=(1, 2)) for obj in objects
        ], -1)
        load = torch.stack(list(object_load.values()), -1)
        credit = (load / weight.clamp_min(1e-12)).clamp(-1.0, 1.0)
        rotation = quat_error_magnitude(now["state"][:, root_columns, 3:7],
                                       command.sim_object_part_state_w[:, root_columns, 3:7])
        orientation = torch.exp(-(rotation / rotation_scale_rad).square())
        credit = torch.where(credit > 0., credit * orientation[:, None], credit)
        eligible = torch.stack(list(object_known.values()), -1).all(1)
        complete &= (~required | eligible).all(-1)
        simultaneous = torch.where(required[:, None], credit, 1.).amin(-1)
        return torch.where(required.any(-1) & complete, (simultaneous * weights).sum(-1), 0.)

    @staticmethod
    def native_source_joint_drive(env, command_name, hinge_effort_scale_nm=.05,
                                  min_hinge_speed_rad_s=.05, min_slide_speed_m_s=.003,
                                  separation_bound_m=.02):
        """Credit contact effort in the moving source direction and actual joint progress."""
        command = env.command_manager.get_term(command_name)
        intent = command.source_hand_part_intent_facts()
        interval = env.scene.hand_part_contact_interval_facts()
        if intent is None or interval is None or not command.object_joint_coordinate_metadata["joint_names"]:
            return torch.zeros(env.num_envs, device=env.device)
        ids = command.motion_lib.length_starts[intent["reference_motion_ids"]] + intent["reference_motion_steps"]
        ref = command.object_joint_reference(ids)
        velocity = ref["velocity"]
        units = ref["position_units"]
        owners = ref["entity_names"]
        floor = velocity.new_tensor([min_hinge_speed_rad_s if unit == "rad" else min_slide_speed_m_s
                                     for unit in units])
        direction = velocity.sign()
        phase = command.task_phase_reference(ids)
        if phase is not None:
            direction = torch.where(phase["joint_declared"], phase["joint_direction"], direction)
        required = ref["velocity_support"] & (velocity.abs() > floor) & (direction != 0)
        valid, dt = contact_interval_weights(env, interval, intent["execution_valid"])
        measured = valid & interval["measurement_support"]
        load = velocity.new_zeros(env.num_envs, interval["num_substeps"], len(units))
        known = measured[..., None].expand_as(load).clone()
        assigned = torch.zeros_like(required)
        active = intent["active"] & intent["known"]
        hands = intent["metadata"]["hand_sides"]
        parts = intent["metadata"]["part_body_names"]
        roots = {hand.side: hand.root_body_name for hand in command.native_layout.hands}
        part_owners = {part.body_name: obj.entity_name for obj, part in command.native_layout.part_bindings}
        for name, address in interval["addresses"].items():
            owner = address["part_entity_name"]
            side, part = address["hand_side"], address["part_body_name"]
            if roots.get(side) != address["hand_body_name"] or part_owners.get(part) != owner:
                continue
            coordinates = command._object_joint_slices[owner]
            if coordinates.start == coordinates.stop:
                continue
            role = active[:, hands.index(side), parts.index(part)]
            assigned[:, coordinates] |= role[:, None]
            pair = interval["pairs"][name]
            effort = pair.get("joint_effort_on_second")
            if effort is None:
                known[..., coordinates] &= ~role[:, None, None]
                continue
            present = pair["force_bearing_contact_present"]
            distance = pair["force_bearing_distance_max_m"]
            supported = torch.isfinite(effort) & (~present | torch.isfinite(distance))[..., None]
            known[..., coordinates] &= ~role[:, None, None] | supported
            eligible = role[:, None] & present & (distance <= separation_bound_m)
            load[..., coordinates] += torch.where(eligible[..., None] & supported, effort, 0.)
        object_weights = {obj.entity_name: env.scene.native_object_weights_n[i].clamp_min(.1)
                          for i, obj in enumerate(command.native_layout.objects)}
        scale = torch.stack([velocity.new_tensor(hinge_effort_scale_nm) if unit == "rad" else object_weights[owner]
                             for owner, unit in zip(owners, units, strict=True)])
        directed = ((load * direction[:, None]) / scale).clamp(0., 1.)
        movement = (command.native_object_joint_vel * direction / velocity.abs().clamp_min(floor)).clamp(0., 1.)
        simultaneous = torch.where(required[:, None], directed * movement[:, None], 1.).amin(-1)
        complete = (~required | (assigned & known.all(1))).all(-1)
        return torch.where(required.any(-1) & complete, (simultaneous * dt).sum(-1), 0.)

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
    def native_contact_overload(env, command_name, threshold_n):
        del command_name
        interval = env.scene.hand_part_contact_interval_facts()
        if interval is None or not interval["pairs"]:
            return torch.zeros(env.num_envs, device=env.device)
        force = torch.stack([pair["normal_force_n"] for pair in interval["pairs"].values()], dim=-1)
        support = interval["sample_recorded"] & interval["clock_support"] & interval["measurement_support"]
        support &= interval["interval_complete"][:, None]
        peak = torch.where(support[..., None] & torch.isfinite(force), force, 0.0).amax(dim=(1, 2))
        return ((peak - threshold_n).clamp_min(0.0) / threshold_n).square()

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
        if term.ref_command_pos_used is None:
            return torch.zeros(env.num_envs, device=env.device)
        mapping = term.actuator_command_indices
        reference = term.ref_command_pos_used[:, mapping]
        limits = term.finger_actuator_limits
        error = (
            0.5
            * (installed["finger_target_rad"] - reference).square()
            / (limits[:, 1] - limits[:, 0])
        )
        value = torch.exp(-scale * error)
        support = installed["finger_source_support"] & installed["installed_valid"][:, None]
        weight = term.actuator_command_weights
        denominator = (support * weight).sum(-1).clamp_min(1)
        if side is None:
            return torch.where(support, value * weight, 0.0).sum(-1) / denominator
        # Existing split_sides divides the term weight across declared hands.
        # A command driving several A slots contributes the same mass as a single drive.
        own = NativeMotionTrackingRewards._side_action_mask(env, side)[term._finger_slice]
        contribution = torch.where(support & own[mapping], value * weight, 0.0).sum(-1)
        return len(command._side_list) * contribution / denominator

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
