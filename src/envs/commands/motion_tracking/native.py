"""Native reference, clock, task geometry and achieved-state semantics for all engines."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from random import Random

import numpy as np
import torch
from mjlab.managers import CommandTerm
from mjlab.utils.lab_api.math import quat_apply_inverse

from envs.scene.native import NativeSceneCfg

from .motion_tracking import MotionTrackingCommand
from .motion_tracking_cfg import MotionTrackingCommandCfg

_PERTURBATION_PHASES = {"interaction", "opening", "closing", "articulate", "hold", "lift", "carry", "support_transfer"}


@dataclass
class NativeCommandControlContext:
    motion_ids: torch.Tensor
    motion_steps: torch.Tensor
    time_left: torch.Tensor
    command_counter: torch.Tensor
    reference_phase: dict[str, torch.Tensor] | None
    metrics: dict[str, torch.Tensor]
    native_step_metrics: dict[str, torch.Tensor] | None
    object_state_version: int
    assist_buffers: dict[str, torch.Tensor]


class NativeMotionTrackingCommand(MotionTrackingCommand):
    """Registered FREE-root references and actual named object articulations.

    Source MANO geometry remains source geometry. Native wrist targets use the
    separately published target-body root carrier, not a renamed MANO wrist.
    """

    def __init__(self, cfg, env):
        self.native_layout = NativeSceneCfg.from_mapping(cfg.native_scene)
        if cfg.ref_noise.enable or cfg.object.static_offset.enable:
            raise ValueError(
                "native task uses its explicit registered scene references"
            )
        super().__init__(cfg, env)
        self._bind_native_reference()
        self._init_native_object_assist()

    def capture_control_context(self) -> NativeCommandControlContext:
        """Native eval carry; no resampling, source queries or physics writes."""
        if not self._eval_mode or self._xfrc_curr_ctrl is not None:
            raise NotImplementedError("native control context requires paused eval curriculum")
        return NativeCommandControlContext(
            self.motion_ids.clone(), self.motion_steps.clone(), self.time_left.clone(),
            self.command_counter.clone(), deepcopy(getattr(self, "reference_phase", None)),
            deepcopy(self.metrics), deepcopy(getattr(self, "native_step_metrics", None)),
            int(getattr(self, "_obj_state_version", 0)),
            {name: value.clone() for name, value in vars(self).items()
             if name.startswith("_assist_") and torch.is_tensor(value)},
        )

    def restore_control_context(self, context: NativeCommandControlContext) -> None:
        if not self._eval_mode or self._xfrc_curr_ctrl is not None:
            raise NotImplementedError("native control context requires paused eval curriculum")
        for name in ("motion_ids", "motion_steps", "time_left", "command_counter"):
            getattr(self, name).copy_(getattr(context, name))
        self.metrics = deepcopy(context.metrics)
        for name in ("reference_phase", "native_step_metrics"):
            value = getattr(context, name)
            if value is None:
                self.__dict__.pop(name, None)
            else:
                setattr(self, name, deepcopy(value))
        for name, value in context.assist_buffers.items():
            getattr(self, name).copy_(value)
        self._obj_state_version = context.object_state_version
        # A rewound step/version can equal an old cache key. Derived reads must
        # be invalidated rather than mistaken for restored operative state.
        for name in ("_sim_obj_cache", "_sim_obj_cache_token_val",
                     "_source_tip_geometry_cache", "_source_tip_geometry_token"):
            self.__dict__.pop(name, None)

    @property
    def object_distance_sign(self) -> str:
        signs = {part.distance_sign for _, part in self.native_layout.part_bindings}
        return next(iter(signs)) if len(signs) == 1 else "mixed"

    @property
    def has_objects(self):
        return bool(self.native_layout.objects)

    @property
    def multi_object(self):
        # The inherited property means alternate per-world asset slots, not
        # simultaneous objects in the authored native scene.
        return False

    def _bind_native_reference(self):
        """Admit the prepared clock and bind declared columns to the actual scene."""
        ml = self.motion_lib
        if not hasattr(ml, "native_time_ns"):
            raise ValueError("native task requires an explicitly prepared native_reference carrier")
        if tuple(ml.native_object_part_body_names) != tuple(
            p.body_name for _, p in self.native_layout.part_bindings
        ):
            raise ValueError("native object reference and scene part bodies differ")
        if ml.native_object_root_body_names != self.native_layout.object_root_body_names:
            raise ValueError("native object reference names different scene entities or FREE bases")
        for hand in self.native_layout.hands:
            if ml.native_root_body_names[hand.side] != hand.root_body_name:
                raise ValueError(
                    "native reference root body does not match the selected asset"
                )
        objects = self.native_objects
        names = tuple(name for obj in objects.values() for name in obj.joint_names)
        if set(names) != set(ml.native_object_joint_names):
            raise ValueError("native object reference and actual scalar joint names differ")
        self._object_joint_columns = torch.tensor(
            [ml.native_object_joint_names.index(name) for name in names],
            device=self.device, dtype=torch.long,
        )
        coordinates = [obj.joint_coordinate_metadata() for obj in objects.values()]
        self.object_joint_coordinate_metadata = {
            key: tuple(value for entry in coordinates for value in entry[key])
            for key in coordinates[0]
        }
        self.object_joint_coordinate_metadata["entity_names"] = tuple(
            entity for entity, obj in objects.items() for _ in obj.joint_names
        )
        self._object_joint_slices = {}
        start = 0
        for entity, obj in objects.items():
            self._object_joint_slices[entity] = slice(start, start + len(obj.joint_names))
            start += len(obj.joint_names)
        self.robot_joint_coordinate_metadata = self.robot.joint_coordinate_metadata()
        actuator_joint_names = tuple(self.robot.joint_names[i] for i in self._finger_joint_ids.tolist())
        command_names, groups, mapping = self.native_layout.command_coordinates(actuator_joint_names)
        self.native_command_state_joint_names = groups
        self._command_state_columns = self.command_state_columns(self.robot.joint_names)
        self._command_state_weights = None
        if self._command_state_columns.ndim == 2:
            width = self._command_state_columns.shape[-1]
            self._command_state_weights = torch.tensor(
                [[1.0 / len(names)] * len(names) + [0.0] * (width - len(names)) for names in groups],
                device=self.device,
            )
        reference_names = getattr(ml, "native_command_names", None)
        self._reference_command_columns = None
        if reference_names is not None:
            if set(reference_names) != set(command_names):
                raise ValueError("native reference logical commands differ from the scene binding")
            self._reference_command_columns = torch.tensor(
                [reference_names.index(name) for name in command_names], device=self.device, dtype=torch.long,
            )
        elif (len(groups) != len(actuator_joint_names)
              or any(groups[c] != (joint,) for joint, c in zip(actuator_joint_names, mapping, strict=True))):
            raise ValueError("native reference requires explicit commands for a nonidentity actuator binding")

        command_columns = [self.robot.joint_names.index(names[0]) for names in groups]
        self.command_joint_coordinate_metadata = {
            key: tuple(values[i] for i in command_columns)
            for key, values in self.robot_joint_coordinate_metadata.items()
        }
        self.command_joint_coordinate_metadata["joint_names"] = command_names
        types = dict(zip(self.robot.joint_names, self.robot_joint_coordinate_metadata["joint_types"], strict=True))
        if any(len({types[name] for name in names}) != 1 for names in groups):
            raise ValueError("native command state cannot project across different physical coordinate types")

        def unit_columns(metadata):
            return {
                unit: torch.tensor(
                    [i for i, actual in enumerate(metadata["position_units"]) if actual == unit],
                    device=self.device, dtype=torch.long,
                )
                for unit in ("rad", "m")
            }

        self.object_joint_columns_by_unit = unit_columns(self.object_joint_coordinate_metadata)
        self.robot_joint_columns_by_unit = unit_columns(self.robot_joint_coordinate_metadata)
        self.command_joint_columns_by_unit = unit_columns(self.command_joint_coordinate_metadata)
        self._part_entities = tuple(objects[obj.entity_name] for obj, _ in self.native_layout.part_bindings)
        self._part_body_ids = [objects[obj.entity_name].body_names.index(p.body_name)
                               for obj, p in self.native_layout.part_bindings]
        self._object_part_columns = {
            obj.entity_name: [i for i, (owner, _) in enumerate(self.native_layout.part_bindings)
                              if owner.entity_name == obj.entity_name]
            for obj in self.native_layout.objects
        }
        self._object_root_columns = torch.tensor(
            [ml.native_object_part_body_names.index(obj.root_body_name) for obj in self.native_layout.objects],
            device=self.device, dtype=torch.long,
        )
        period_ns = float(self._env.step_dt) * 1e9
        for start, count in zip(ml.length_starts.tolist(), ml._motion_num_frames.tolist(), strict=True):
            times = ml.native_time_ns[start:start + count]
            expected = torch.round(torch.arange(count, device=times.device, dtype=torch.float64)
                                   * period_ns).to(torch.int64)
            if times.dtype != torch.int64 or bool(((times - times[0] - expected).abs() > 1).any()):
                raise ValueError("native prepared reference cadence differs from the actual control clock")

    def part_surface_points(self, n_points, n_obj_verts):
        from envs.object_points import native_part_surface_points

        return native_part_surface_points(self, self.device, n_points, n_obj_verts)

    def part_surface_geometry_metadata(self):
        from envs.object_points import native_part_surface_metadata

        return native_part_surface_metadata(self)

    def part_surface_reference(self, *, offset=0, motion_ids=None, motion_steps=None):
        from envs.object_points import native_part_reference

        return native_part_reference(self.motion_lib,
            self.motion_ids if motion_ids is None else motion_ids,
            self.motion_steps if motion_steps is None else motion_steps, offset=offset)

    def _init_site_ids(self, cfg):
        if set(self._side_list) != {hand.side for hand in self.native_layout.hands}:
            raise ValueError("native reference sides differ from the configured physical hands")
        self._palm_site_ids = {}
        self._tip_site_ids, self._contact_site_ids = {}, {}
        for side in self._side_list:
            self._tip_site_ids[side], _ = self.robot.find_sites(
                cfg.site_names["tip"][side], preserve_order=True
            )
            self._contact_site_ids[side], _ = self.robot.find_sites(
                cfg.site_names["contact"][side], preserve_order=True
            )
        self._root_body_ids = [
            self.robot.body_names.index(
                next(h.root_body_name for h in self.native_layout.hands if h.side == s)
            )
            for s in self._side_list
        ]

    def _init_joint_ids(self, cfg):
        names = tuple(
            a for h in self.native_layout.hands for a in h.command_actuator_names
        )
        ids, _ = self.robot.find_joints_by_actuator_names(names)
        self._finger_joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        self._wrist_trans_ids = self._wrist_rot_ids = torch.empty(
            0, device=self.device, dtype=torch.long
        )
        self._wrist_xy_ids = {}

    @property
    def native_command_joint_ids(self):
        """Physical actuator joint ids, including any jointly commanded drives."""
        return self._finger_joint_ids

    def command_state_columns(self, joint_names):
        width = max(map(len, self.native_command_state_joint_names))
        groups = [[joint_names.index(name) for name in names] for names in self.native_command_state_joint_names]
        rows = [ids[0] for ids in groups] if width == 1 else [ids + [ids[0]] * (width - len(ids)) for ids in groups]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def project_command_state(self, joint_state, columns=None):
        """Project measured physical state into the declared logical coordinate groups."""
        columns = self._command_state_columns if columns is None else columns
        values = joint_state[..., columns]
        weights = self._command_state_weights
        if weights is None:
            return values
        if values.dtype == torch.bool:
            return (values | (weights == 0)).all(-1)
        return (values * weights).sum(-1)

    def requested_command_reference(self, ids):
        """Admitted logical position targets and their independent support."""
        ml, columns = self.motion_lib, self._reference_command_columns
        if columns is None:
            return (self.project_command_state(ml.robot_joint_pos[ids]),
                    self.project_command_state(ml.native_joint_position_support[ids]))
        return ml.native_command_pos[ids][..., columns], ml.native_command_position_support[ids][..., columns]

    def command_reference(self, ids=None):
        """Requested logical positions and projected physical reference rates."""
        ids = self._motion_flat_ids if ids is None else ids
        ml = self.motion_lib
        position, position_support = self.requested_command_reference(ids)
        return {
            **self.command_joint_coordinate_metadata,
            "position": position,
            "velocity": self.project_command_state(ml.robot_joint_vel[ids]),
            "position_support": position_support,
            "velocity_support": self.project_command_state(ml.native_joint_velocity_support[ids]),
        }

    def object_joint_reference(self, ids=None):
        """Reference columns in actual object joint order, including a real width-zero rigid case."""
        ids = self._motion_flat_ids if ids is None else ids
        ml, columns = self.motion_lib, self._object_joint_columns
        return {
            **self.object_joint_coordinate_metadata,
            "position": ml.native_object_joint_pos[ids][..., columns],
            "velocity": ml.native_object_joint_vel[ids][..., columns],
            "position_support": ml.native_object_joint_position_support[ids][..., columns],
            "velocity_support": ml.native_object_joint_velocity_support[ids][..., columns],
        }

    def task_phase_reference(self, ids):
        """Declared interaction on the exact consumed source frame, in native scene order."""
        facts = getattr(self.motion_lib, "native_task_phases", None)
        if facts is None:
            return None
        source_objects = tuple(self.motion_lib.native_object_root_body_names)
        columns = [source_objects.index(obj.entity_name) for obj in self.native_layout.objects]
        return {
            field: values[ids][:, self._object_joint_columns if field.startswith("joint_") else columns]
            for field, values in facts.items()
        }

    @property
    def native_object_joint_pos(self):
        values = [obj.data.joint_pos for obj in self.native_objects.values() if obj.joint_names]
        return torch.cat(values, -1) if values else self.robot.data.joint_pos.new_empty(self.num_envs, 0)

    @property
    def native_object_joint_vel(self):
        values = [obj.data.joint_vel for obj in self.native_objects.values() if obj.joint_names]
        return torch.cat(values, -1) if values else self.robot.data.joint_vel.new_empty(self.num_envs, 0)

    def _reference_free_root_states(self):
        return {
            h.root_joint_name: self.motion_lib.native_root_state_w[h.side]
            .detach()
            .cpu()
            .numpy()
            for h in self.native_layout.hands
        }

    def _reference_root_body_names(self):
        return {h.side: h.root_body_name for h in self.native_layout.hands}

    def _root_field(self, field):
        return getattr(self.robot.data, field)[:, self._root_body_ids]

    @property
    def robot_wrist_trans_w(self):
        return self._root_field("body_link_pos_w")

    @property
    def robot_wrist_quat_w(self):
        return self._root_field("body_link_quat_w")

    @property
    def robot_wrist_lin_vel_w(self):
        return self._root_field("body_link_lin_vel_w")

    @property
    def robot_wrist_ang_vel_w(self):
        return self._root_field("body_link_ang_vel_w")

    @property
    def ref_native_root_state_w(self):
        state = torch.stack(
            [
                self.motion_lib.native_root_state_w[s][self._motion_flat_ids]
                for s in self._side_list
            ],
            1,
        ).clone()
        state[..., :3] += self._env.scene.env_origins[:, None]
        return state

    @property
    def native_objects(self):
        return {obj.entity_name: self._env.scene[obj.entity_name] for obj in self.native_layout.objects}

    @property
    def sim_object_part_state_w(self):
        return torch.cat(
            [
                torch.stack([getattr(obj.data, f)[:, body]
                             for obj, body in zip(self._part_entities, self._part_body_ids, strict=True)], 1)
                for f in (
                    "body_link_pos_w",
                    "body_link_quat_w",
                    "body_link_lin_vel_w",
                    "body_link_ang_vel_w",
                )
            ],
            -1,
        )

    def object_part_reference(self, ids):
        state = self.motion_lib.native_object_part_state_w[ids].clone()
        origins = self._env.scene.env_origins.reshape(
            (self.num_envs,) + (1,) * (state.ndim - 2) + (3,)
        )
        state[..., :3] += origins
        return state

    @torch.no_grad()
    def source_point_reference(self, ids):
        """Actual named source tips at supplied flat reference rows, in WORLD meters.

        Source support belongs to observed points, independently of registered
        target roots/Q. The selected source columns pair with real configured
        tip sites; absent anatomy is never filled by target FK.
        """
        ml = self.motion_lib
        masks = ml.native_source_point_position_support
        if masks is None:
            raise ValueError("native source geometry requires named source-point position support")
        layout = getattr(self, "_source_tip_point_layout", None)
        if layout is None:
            names, sites = {}, {}
            for side in self._side_list:
                columns, site_ids = ml.tip_ids[side], self._tip_site_ids[side]
                if len(columns) != len(site_ids):
                    raise ValueError("native source tip columns and configured physical tip sites differ")
                names[side] = tuple(ml.mano_joint_names[side][i] for i in columns.tolist())
                sites[side] = tuple(self.robot.site_names[i] for i in site_ids)
            layout = self._source_tip_point_layout = names, sites
        names, sites = layout
        positions, support = {}, {}
        for side in self._side_list:
            columns = ml.tip_ids[side]
            value = ml.mano_joint_pos[side][ids][..., columns, :]
            origins = self._env.scene.env_origins.reshape(
                (self.num_envs,) + (1,) * (value.ndim - 2) + (3,)
            )
            positions[side] = value + origins
            support[side] = masks[side][ids][..., columns] & torch.isfinite(value).all(-1)
        return dict(
            hand_sides=tuple(self._side_list), source_point_names=names,
            target_site_names=sites, position_w=positions, position_support=support,
            reference_time_ns=ml.native_time_ns[ids],
        )

    def source_hand_part_intent(self, ids):
        """Source-owned intent in actual hand/part order; absent labels stay unknown."""
        ml = self.motion_lib
        intent = ml.native_source_hand_part_intent
        if intent is None:
            return None
        hands = [ml.hand_sides.index(side) for side in self._side_list]
        return {
            "metadata": dict(
                hand_sides=tuple(self._side_list), part_body_names=tuple(ml.native_object_part_body_names),
                authority="declared source intent; not measured target contact, force or task success",
                source_provenance=ml.native_source_hand_part_intent_provenance,
                clock="original int64 source capture/availability/cell end; query phase retained independently",
            ),
            **{field: values[ids][..., hands, :] if field in ("active", "known") else values[ids]
               for field, values in intent.items()},
            "reference_time_ns": ml.native_time_ns[ids],
        }

    @cached_property
    def source_stationary_parts(self):
        """Require a stationary, source-supported part and owning root throughout the motion."""
        from mjlab.utils.lab_api.math import quat_error_magnitude

        ml = self.motion_lib
        stationary = []
        for start, count in zip(ml.length_starts.tolist(), ml._motion_num_frames.tolist(), strict=True):
            rows = slice(start, start + count)
            pose = ml.native_object_part_state_w[rows, :, :7]
            displacement = (pose[..., :3] - pose[0, :, :3]).norm(dim=-1)
            rotation = quat_error_magnitude(pose[..., 3:7], pose[0:1, :, 3:7].expand_as(pose[..., 3:7]))
            stable = (ml.native_object_part_pose_support[rows] & (displacement <= .005) & (rotation <= .05)).all(0)
            for columns, root in zip(self._object_part_columns.values(), self._object_root_columns, strict=True):
                stable[columns] = stable[columns] & stable[root]
            stationary.append(stable)
        return torch.stack(stationary)

    @torch.no_grad()
    def source_contact_approach_direction(
        self, ids, *, max_part_position_error_m, max_part_rotation_error_rad, max_tip_distance_m,
    ):
        """Select a supported contact direction and identify intentional drive-side acquisition."""
        from mjlab.utils.lab_api.math import quat_error_magnitude

        hands = self.native_layout.hands
        directions = self.robot.data.site_pos_w.new_zeros((self.num_envs, len(hands), 3))
        selected_part = torch.full_like(directions[..., 0], -1, dtype=torch.long)
        drives_joint = torch.zeros_like(selected_part, dtype=torch.bool)
        moving_parts = torch.zeros((self.num_envs, len(self.native_layout.part_bindings)),
                                   dtype=torch.bool, device=selected_part.device)
        intent = self.source_hand_part_intent(ids)
        if intent is None:
            return directions, selected_part, drives_joint, moving_parts
        live, target = self.sim_object_part_state_w, self.object_part_reference(ids)
        pose = (self.motion_lib.native_object_part_pose_support[ids]
                & ((live[..., :3] - target[..., :3]).norm(dim=-1) <= max_part_position_error_m)
                & (quat_error_magnitude(live[..., 3:7], target[..., 3:7]) <= max_part_rotation_error_rad))
        sites = [self._tip_site_ids[hand.side] for hand in hands]
        tips = self.robot.data.site_pos_w[:, [site for group in sites for site in group]]
        distance, gradient, query = self.sdf_query_parts(tips, part_state_w=live, return_support=True)
        active = intent["active"] & intent["known"] & pose[:, None]
        drive = self._articulated_approach(ids, tips, live, target, distance, gradient, query,
                                           max_tip_distance_m)
        if drive is not None:
            moving_parts, drive_supported, drive_delta = drive
        start = 0
        for h, (hand, group) in enumerate(zip(hands, sites, strict=True)):
            end = start + len(group)
            hand_index = intent["metadata"]["hand_sides"].index(hand.side)
            rays = gradient[:, start:end]
            supported = (query[:, start:end] & active[:, hand_index, None]
                         & torch.isfinite(rays).all(-1) & (rays.square().sum(-1) > 0.0))
            if drive is not None:
                # A signed phase cannot acquire its moving part from the opposing face.
                supported &= ~moving_parts[:, None]
            signed = distance[:, start:end].flatten(1)
            nearest, index = torch.where(supported.flatten(1), signed.abs(), float("inf")).min(-1, keepdim=True)
            valid = nearest[:, 0] <= max_tip_distance_m
            selected_part[:, h] = torch.where(valid, index[:, 0] % distance.shape[-1], -1)
            ray = rays.flatten(1, 2).gather(1, index[..., None].expand(-1, -1, 3))[:, 0]
            # Scale the last millimetre toward zero, where the signed normal can flip.
            fraction = (signed.gather(1, index) / 0.001).clamp(-1., 1.)
            directions[:, h] = torch.where(valid[:, None], -ray * fraction, 0.)
            if drive is not None:
                delta = drive_delta[:, start:end].flatten(1, 2)
                candidates = (drive_supported[:, start:end]
                              & (intent["active"] & intent["known"])[:, hand_index, None]).flatten(1)
                length = delta.norm(dim=-1)
                ranked, index = torch.where(candidates, length, float("inf")).min(-1, keepdim=True)
                accepted = torch.isfinite(ranked[:, 0])
                chosen = delta.gather(1, index[..., None].expand(-1, -1, 3))[:, 0]
                selected_part[:, h] = torch.where(accepted, index[:, 0] % distance.shape[-1], selected_part[:, h])
                drives_joint[:, h] = accepted
                directions[:, h] = torch.where(accepted[:, None],
                                               chosen / chosen.norm(dim=-1, keepdim=True).clamp_min(.001),
                                               directions[:, h])
            start = end
        return directions, selected_part, drives_joint, moving_parts

    @cached_property
    def _approach_joint_chains(self):
        """Authored joint frames and descendant parts, shared across simulator ports."""
        import mujoco

        model = mujoco.MjModel.from_xml_path(self.native_layout.scene_xml_path)
        chains = []
        joints = self.object_joint_coordinate_metadata
        for column, (name, owner) in enumerate(zip(joints["joint_names"], joints["entity_names"], strict=True)):
            joint = model.joint(name).id
            child = model.jnt_bodyid[joint]
            entity = self.native_objects[owner]
            parts = []
            for part_column in self._object_part_columns[owner]:
                body = model.body(self.native_layout.part_bindings[part_column][1].body_name).id
                while body and body != child:
                    body = model.body_parentid[body]
                if body == child:
                    parts.append(part_column)
            if parts:
                axis = self.robot.data.site_pos_w.new_tensor(model.jnt_axis[joint])
                anchor = self.robot.data.site_pos_w.new_tensor(model.jnt_pos[joint])
                chains.append((column, entity, entity.body_names.index(model.body(child).name),
                               parts, axis, anchor, model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE))
        return chains

    def _articulated_approach(self, ids, tips, live, target, distance, gradient, query, max_tip_distance_m):
        from mjlab.utils.lab_api.math import quat_apply

        phase = self.task_phase_reference(ids)
        if (phase is None or not phase["joint_direction"].shape[-1]
                or getattr(self.motion_lib, "native_source_point_position_support", None) is None):
            return None
        reference = self.source_point_reference(ids)
        source_tips = torch.cat([reference["position_w"][hand.side] for hand in self.native_layout.hands], 1)
        source_support = torch.cat([reference["position_support"][hand.side] for hand in self.native_layout.hands], 1)
        gap, source_normal, valid = self.sdf_query_parts(source_tips, part_state_w=target, return_support=True)
        ref_quat = target[:, None, :, 3:7].expand(*gap.shape, 4)
        live_quat = live[:, None, :, 3:7].expand_as(ref_quat)
        offset = source_tips[:, :, None] - gap[..., None] * source_normal - target[:, None, :, :3]
        normal = quat_apply(live_quat, quat_apply_inverse(ref_quat, source_normal))
        anchors = live[:, None, :, :3] + quat_apply(live_quat, quat_apply_inverse(ref_quat, offset))
        delta = anchors - .001 * normal - tips[:, :, None]
        moving = torch.zeros_like(live[..., 0], dtype=torch.bool)
        direction_safe = torch.ones_like(query)
        directions = phase["joint_direction"]
        for column, entity, child, parts, axis, anchor, hinged in self._approach_joint_chains:
            sign = directions[:, column]
            moving[:, parts] |= sign[:, None] != 0
            quat = entity.data.body_link_quat_w[:, child]
            axis_w = quat_apply(quat, axis.expand_as(quat[:, :3]))[:, None, None]
            velocity = axis_w
            if hinged:
                pivot = entity.data.body_link_pos_w[:, child] + quat_apply(quat, anchor.expand_as(quat[:, :3]))
                lever = anchors[:, :, parts] - pivot[:, None, None]
                velocity = torch.linalg.cross(axis_w.expand_as(lever), lever)
            effort = -(normal[:, :, parts] * velocity).sum(-1) * sign[:, None, None]
            direction_safe[:, :, parts] &= (sign[:, None, None] == 0) | (effort > 0.)
        # Nearest normals can agree around an edge even when the target pushes outward.
        aims_inward = (delta * normal).sum(-1) <= 0.
        supported = (query & valid & source_support[..., None] & moving[:, None] & direction_safe
                     & self.motion_lib.native_object_part_pose_support[ids][:, None]
                     & (gap.abs() <= max_tip_distance_m) & (distance >= -.001) & aims_inward
                     & (delta.norm(dim=-1) <= max_tip_distance_m) & ((normal * gradient).sum(-1) > 0.))
        return moving, supported, delta

    def _consumed_reference_phase(self):
        from envs._common import action_term

        installed = action_term(self._env).installed_control
        execution = installed["installed_valid"].detach().clone()
        phase = self.reference_phase
        motion_ids = torch.where(execution, installed["reference_motion_ids"], phase["motion_ids"])
        steps = torch.where(execution, installed["reference_motion_steps"], phase["motion_steps"])
        return self.motion_lib.length_starts[motion_ids] + steps, motion_ids, steps, execution

    @torch.no_grad()
    def source_hand_part_intent_facts(self):
        """Recorder/reward endpoint uses the installed packet, not an advanced cursor."""
        if self.motion_lib.native_source_hand_part_intent is None:
            return None
        ids, motion_ids, steps, execution = self._consumed_reference_phase()
        facts = self.source_hand_part_intent(ids)
        if facts is not None:
            facts.update(reference_motion_ids=motion_ids, reference_motion_steps=steps, execution_valid=execution)
        return facts

    @torch.no_grad()
    def source_tip_geometry_facts(self):
        """One stopped consumed-endpoint packet, shared by reward and recorder.

        The normal substep/reset token owns physical freshness. Command cursor
        advance alone must not switch the installed packet's source geometry.
        Decision observations use source_point_reference(current_ids) instead.
        """
        if self.motion_lib.native_source_point_position_support is None:
            return None
        token = self._sim_obj_cache_token()
        cached = getattr(self, "_source_tip_geometry_cache", None)
        if cached is not None and self._source_tip_geometry_token == token:
            return cached
        ids, motion_ids, steps, execution = self._consumed_reference_phase()
        reference = self.source_point_reference(ids)
        source, source_support, achieved = [], [], []
        source_names, target_names, point_sides, offsets = [], [], [], [0]
        for side in self._side_list:
            source.append(reference["position_w"][side])
            source_support.append(reference["position_support"][side])
            achieved.append(self.robot.data.site_pos_w[:, self._tip_site_ids[side]])
            names = reference["source_point_names"][side]
            source_names.extend(names)
            target_names.extend(reference["target_site_names"][side])
            point_sides.extend([side] * len(names))
            offsets.append(offsets[-1] + len(names))
        source, source_support = torch.cat(source, 1), torch.cat(source_support, 1)
        achieved = torch.cat(achieved, 1)
        achieved_support = torch.isfinite(achieved).all(-1)
        desired_state = self.object_part_reference(ids)
        live_state = self.sim_object_part_state_w
        desired_pose = self.motion_lib.native_object_part_pose_support[ids] & torch.isfinite(desired_state[..., :7]).all(-1)
        live_pose = torch.isfinite(live_state[..., :7]).all(-1)

        def supported_pose(state, mask):
            identity = torch.zeros_like(state)
            identity[..., 3] = 1.0
            return torch.where(mask[..., None], state, identity)

        desired_distance, _, desired_query = self.sdf_query_parts(
            torch.where(source_support[..., None], source, 0.0),
            part_state_w=supported_pose(desired_state, desired_pose), return_support=True,
        )
        live_distance, _, live_query = self.sdf_query_parts(
            torch.where(achieved_support[..., None], achieved, 0.0),
            part_state_w=supported_pose(live_state, live_pose), return_support=True,
        )
        desired_query &= source_support[..., None] & desired_pose[:, None]
        live_query &= achieved_support[..., None] & live_pose[:, None]
        facts = dict(
            metadata=dict(
                schema="native_source_tip_geometry_v1", hand_sides=tuple(self._side_list),
                hand_offsets=tuple(offsets), point_hand_sides=tuple(point_sides),
                source_point_names=tuple(source_names), target_site_names=tuple(target_names),
                part_names=tuple(f"{obj.entity_name}/{p.name}" for obj, p in self.native_layout.part_bindings),
                part_entity_names=tuple(obj.entity_name for obj, _ in self.native_layout.part_bindings),
                part_body_names=tuple(p.body_name for _, p in self.native_layout.part_bindings),
                part_distance_sign=tuple(p.distance_sign for _, p in self.native_layout.part_bindings),
                position_frame="WORLD_including_env_origin", position_unit="m", distance_unit="m",
                query_authority="finite_supported_inputs_inside_baked_grid_not_contact_or_surface_coverage",
                reference_authority="installed_packet_or_reset_pre_advance_reference_phase",
                execution_authority="native_action_installed_valid_not_backend_executed_control_readback",
            ),
            reference_motion_ids=motion_ids, reference_motion_steps=steps,
            reference_time_ns=reference["reference_time_ns"], execution_valid=execution,
            source_position_w=source, source_position_support=source_support,
            achieved_position_w=achieved, achieved_position_support=achieved_support,
            desired_part_pose_support=desired_pose, live_part_pose_support=live_pose,
            desired_part_distance_m=torch.where(desired_query, desired_distance, float("nan")),
            live_part_distance_m=torch.where(live_query, live_distance, float("nan")),
            desired_part_query_support=desired_query, live_part_query_support=live_query,
        )
        self._source_tip_geometry_cache = facts
        self._source_tip_geometry_token = token
        return facts

    @property
    def ref_object_part_state_w(self):
        return self.object_part_reference(self._motion_flat_ids)

    def _object_base_reference(self, ids):
        return self.object_part_reference(ids)[..., self._object_root_columns, :]

    @property
    def sim_obj_trans_w(self):
        return self.sim_object_part_state_w[:, self._object_root_columns, :3]

    @property
    def sim_obj_quat_w(self):
        return self.sim_object_part_state_w[:, self._object_root_columns, 3:7]

    @property
    def sim_obj_lin_vel_w(self):
        return self.sim_object_part_state_w[:, self._object_root_columns, 7:10]

    @property
    def sim_obj_ang_vel_w(self):
        return self.sim_object_part_state_w[:, self._object_root_columns, 10:13]

    @property
    def ref_obj_trans_w(self):
        return self._object_base_reference(self._motion_flat_ids)[..., :3]

    @property
    def ref_obj_quat_w(self):
        return self._object_base_reference(self._motion_flat_ids)[..., 3:7]

    @property
    def ref_obj_lin_vel_w(self):
        return self._object_base_reference(self._motion_flat_ids)[..., 7:10]

    @property
    def ref_obj_ang_vel_w(self):
        return self._object_base_reference(self._motion_flat_ids)[..., 10:13]

    @property
    def ref_obj_rotmat_w(self):
        from mjlab.utils.lab_api.math import matrix_from_quat

        return matrix_from_quat(self.ref_obj_quat_w)

    def future_obj_traj_trans_w(self, offsets):
        return self._object_base_reference(self._future_motion_flat_ids(offsets))[
            ..., :3
        ]

    def future_obj_traj_rotmat_w(self, offsets):
        from mjlab.utils.lab_api.math import matrix_from_quat

        return matrix_from_quat(
            self._object_base_reference(self._future_motion_flat_ids(offsets))[..., 3:7]
        )

    def future_obj_traj_lin_vel_w(self, offsets):
        return self._object_base_reference(self._future_motion_flat_ids(offsets))[
            ..., 7:10
        ]

    def future_obj_traj_ang_vel_w(self, offsets):
        return self._object_base_reference(self._future_motion_flat_ids(offsets))[
            ..., 10:13
        ]

    def _obj_mass_body_idx(self, obj):
        binding = next(binding for binding in self.native_layout.objects
                       if self._env.scene[binding.entity_name] is obj)
        return obj.body_names.index(binding.root_body_name)

    def _init_metrics(self):
        # Native reference-tracking metrics do not inherit raw-MANO wrist or
        # rigid-side contact statistics as if they described the registered task.
        self.metrics["native_root_position_error_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["native_part_position_error_m"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        # Use the real command lifecycle without the stock rigid contact accumulator.
        return CommandTerm.reset(self, env_ids)

    def _update_metrics(self, env_ids=None):
        from mjlab.utils.lab_api.math import quat_error_magnitude

        # This override bypasses common metrics, including its pre-advance phase.
        self._record_reference_phase(env_ids)
        ml, ids = self.motion_lib, self._motion_flat_ids
        ref = self.ref_native_root_state_w
        root_p = (ref[..., :3] - self.robot_wrist_trans_w).norm(dim=-1)
        root_r = quat_error_magnitude(ref[..., 3:7], self.robot_wrist_quat_w)
        pose_support = torch.stack(
            [ml.native_root_pose_support[s][ids] for s in self._side_list], -1
        )
        parts, part_ref = self.sim_object_part_state_w, self.ref_object_part_state_w
        object_ref, command_ref = self.object_joint_reference(), self.command_reference()
        angular = self.object_joint_columns_by_unit["rad"]
        linear = self.object_joint_columns_by_unit["m"]
        joint_error = (object_ref["position"] - self.native_object_joint_pos).abs()
        command_error = (
            command_ref["position"] - self.project_command_state(self.robot.data.joint_pos)
        ).abs()
        command_angular = self.command_joint_columns_by_unit["rad"]
        command_linear = self.command_joint_columns_by_unit["m"]
        part_p = (part_ref[..., :3] - parts[..., :3]).norm(dim=-1)
        part_support = ml.native_object_part_pose_support[ids]
        self._set_metric("native_root_position_error_m", (
            torch.where(pose_support, root_p, 0.0).sum(-1) / pose_support.sum(-1).clamp_min(1)
        ), env_ids)
        self._set_metric("native_part_position_error_m", (
            torch.where(part_support, part_p, 0.0).sum(-1) / part_support.sum(-1).clamp_min(1)
        ), env_ids)
        step_metrics = {
            "reference_time_ns": ml.native_time_ns[ids],
            "root_position_error_m": root_p,
            "root_rotation_error_rad": root_r,
            "root_pose_support": pose_support,
            "part_position_error_m": part_p,
            "part_rotation_error_rad": quat_error_magnitude(
                part_ref[..., 3:7], parts[..., 3:7]
            ),
            "part_pose_support": part_support,
            "joint_position_error_rad": joint_error[:, angular],
            "joint_position_support": object_ref["position_support"][:, angular],
            "joint_position_error_m": joint_error[:, linear],
            "joint_linear_position_support": object_ref["position_support"][:, linear],
            "command_position_error_rad": command_error[:, command_angular],
            "command_position_support": command_ref["position_support"][:, command_angular],
            "command_position_error_m": command_error[:, command_linear],
            "command_linear_position_support": command_ref["position_support"][:, command_linear],
        }
        if env_ids is None or not hasattr(self, "native_step_metrics"):
            self.native_step_metrics = step_metrics
        else:
            for name, value in step_metrics.items():
                self.native_step_metrics[name][env_ids] = value[env_ids]

    def _resample_command(self, env_ids):
        ml = self.motion_lib
        if not self._eval_mode and ml.num_trajectories > 1:
            self.motion_ids[env_ids] = torch.randint(
                ml.num_trajectories, (len(env_ids),), device=self.device
            )
        if self.cfg.sampling.mode == "start":
            start = int(self.cfg.sampling.start_frame)
            if start < 0:
                raise ValueError("native sampling.start_frame must be nonnegative")
            lengths = ml._motion_num_frames[self.motion_ids[env_ids]]
            self.motion_steps[env_ids] = torch.minimum(
                torch.full_like(lengths, start), lengths - 1
            )
        elif self.cfg.sampling.mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            raise ValueError(f"unknown native sampling mode {self.cfg.sampling.mode!r}")
        ids = self._motion_flat_ids[env_ids]
        if not bool(ml.native_joint_position_support[ids].all()):
            raise ValueError("native reset requires a fully known physical scalar pose")
        q = self.ref_joint_pos[env_ids].clone()
        v = torch.where(
            ml.native_joint_velocity_support[ids], self.ref_joint_vel[env_ids], 0.0
        )
        if self.cfg.hand.zero_init_vel:
            v.zero_()
        self.robot.write_joint_state_to_sim(q, v, env_ids=env_ids)
        for hand in self.native_layout.hands:
            if not bool(ml.native_root_pose_support[hand.side][ids].all()):
                raise ValueError(
                    "native reset requires a known registered FREE-root pose"
                )
            state = ml.native_root_state_w[hand.side][ids].clone()
            state[:, :3] += self._env.scene.env_origins[env_ids]
            state[:, 7:10] = torch.where(
                ml.native_root_linear_velocity_support[hand.side][ids, None],
                state[:, 7:10],
                0.0,
            )
            state[:, 10:13] = torch.where(
                ml.native_root_angular_velocity_support[hand.side][ids, None],
                state[:, 10:13],
                0.0,
            )
            if self.cfg.hand.zero_init_vel:
                state[:, 7:] = 0.0
            self.robot.hands[hand.side].write_root_state_to_sim(state, env_ids)
        object_ref = self.object_joint_reference(ids)
        if not bool(object_ref["position_support"].all()):
            raise ValueError("native reset requires every passive articulation pose")
        for binding in self.native_layout.objects:
            obj = self.native_objects[binding.entity_name]
            base = ml.native_object_part_body_names.index(binding.root_body_name)
            if not bool(ml.native_object_part_pose_support[ids, base].all()):
                raise ValueError(f"native reset requires object {binding.entity_name!r} FREE-base pose")
            state = ml.native_object_part_state_w[ids, base].clone()
            state[:, :3] += self._env.scene.env_origins[env_ids]
            state[:, 7:10] = torch.where(
                ml.native_object_part_linear_velocity_support[ids, base, None], state[:, 7:10], 0.0,
            )
            state[:, 10:13] = torch.where(
                ml.native_object_part_angular_velocity_support[ids, base, None], state[:, 10:13], 0.0,
            )
            columns = self._object_joint_slices[binding.entity_name]
            ov = torch.where(object_ref["velocity_support"][:, columns],
                             object_ref["velocity"][:, columns], 0.0)
            if self.cfg.object.zero_init_vel:
                state[:, 7:], ov = 0.0, torch.zeros_like(ov)
            obj.write_root_state_to_sim(state, env_ids)
            if obj.joint_names:
                obj.write_joint_state_to_sim(object_ref["position"][:, columns], ov, env_ids=env_ids)
        self.clear_object_wrenches(env_ids)
        self._obj_state_version = getattr(self, "_obj_state_version", 0) + 1

    def _init_native_object_assist(self):
        shape = (self.num_envs, len(self.native_layout.part_bindings), 3)
        self._assist_force_w = torch.zeros(shape, device=self.device)
        self._assist_torque_w = torch.zeros_like(self._assist_force_w)
        self._assist_last_force_w = torch.zeros_like(self._assist_force_w)
        self._assist_last_torque_w = torch.zeros_like(self._assist_force_w)
        self._assist_pose_support = torch.zeros(shape[:-1], dtype=torch.bool, device=self.device)
        self._assist_linear_rate_support = torch.zeros_like(self._assist_pose_support)
        self._assist_angular_rate_support = torch.zeros_like(self._assist_pose_support)
        self._assist_submission_support = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._assist_sample_time_s = torch.full((self.num_envs,), float("nan"), dtype=torch.float64, device=self.device)
        self._assist_clock_support = torch.zeros_like(self._assist_submission_support)
        self._assist_perturb_wrench = torch.zeros(
            (self.num_envs, len(self.native_layout.objects), 6), device=self.device,
        )
        self._assist_perturb_active = torch.zeros_like(self._assist_perturb_wrench[..., 0], dtype=torch.bool)
        self.hoi_evaluation_mode = None
        self._perturb_inverse_inertia_body = None
        perturb = getattr(self.cfg.object, "perturb", None)
        if perturb is not None and perturb.enabled:
            lag = float(perturb.gate_lag_s)
            if not math.isfinite(lag) or lag < 0:
                raise ValueError("native perturbation requires a finite nonnegative engagement lag")
            self._perturb_lag_ns = round(lag * 1e9)
            self._perturb_source_gate = self._source_perturbation_gate()
            if perturb.torque_scale:
                limit = perturb.max_angular_acceleration_rad_s2
                if limit is None or not math.isfinite(limit) or limit <= 0.0:
                    raise ValueError("native rotational perturbation requires a positive max_angular_acceleration_rad_s2")
                self._perturb_max_angular_acceleration_rad_s2 = float(limit)
                # The installed inertia of each struck FREE root is fixed in its body frame.
                inertia = torch.stack([
                    self._part_entities[column].body_inertial_properties().inertia_com_body_kg_m2[
                        :, self._part_body_ids[column]]
                    for column in self._object_root_columns.tolist()
                ], dim=1)
                self._perturb_inverse_inertia_body = torch.linalg.inv(inertia)

    def _source_perturbation_gate(self):
        """Arm each object after consecutive supported source engagement."""
        ml = self.motion_lib
        intent = ml.native_source_hand_part_intent
        if intent is None:
            raise ValueError("native perturbation requires source hand-part intent")
        part_active = (intent["active"] & intent["known"]).any(dim=1) & ml.native_object_part_pose_support
        active = torch.stack([
            part_active[:, self._object_part_columns[obj.entity_name]].any(dim=-1)
            for obj in self.native_layout.objects
        ], dim=-1)
        entry = active & ~torch.roll(active, 1, dims=0)
        entry[ml.length_starts] = True
        row = torch.arange(len(active), device=self.device)[:, None]
        start = torch.where(entry, row, 0).cummax(dim=0).values
        elapsed_ns = ml.native_time_ns[:, None] - ml.native_time_ns[start]
        return active & (elapsed_ns >= self._perturb_lag_ns)

    def enable_hoi_evaluation_perturbation(self, mode):
        if not self._eval_mode or self.cfg.object.pin_objects or not self.cfg.object.perturb.enabled:
            raise ValueError("HOI perturbation evaluation needs unassisted eval with authored disturbances")
        if mode not in ("nominal", "perturbed"):
            raise ValueError("HOI evaluation requires nominal or perturbed mode")
        perturb = self.cfg.object.perturb
        if (not 0. <= perturb.continue_prob <= 1.
                or not all(math.isfinite(scale) and scale >= 0. for scale in (perturb.force_scale, perturb.torque_scale))
                or not (perturb.force_scale or perturb.torque_scale)):
            raise ValueError("HOI qualification requires positive finite disturbance amplitude and a burst duration")
        self.hoi_evaluation_mode = mode
        self._assist_eval_start = self._assist_perturb_active.new_zeros((*self._assist_perturb_active.shape, 1), dtype=torch.int64)
        self._assist_eval_stop = torch.zeros_like(self._assist_eval_start)
        self._assist_eval_wrench = self._assist_perturb_wrench.unsqueeze(-2).new_zeros(
            (*self._assist_perturb_wrench.shape[:-1], 1, 6))
        self._assist_eval_tick = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    def plan_hoi_evaluation_attempts(self, env_ids, replica_indices):
        """Assign each evaluated original motion a fixed, recorded disturbance from its reset."""
        ml, perturb = self.motion_lib, self.cfg.object.perturb
        control_ns = round(self._env.step_dt * 1e9)
        physics_ns = round(self._env.physics_dt * 1e9)
        substeps, clock_remainder = divmod(control_ns, physics_ns)
        if clock_remainder or substeps < 1:
            raise ValueError("HOI perturbation requires an integer solver/control clock")
        self._assist_eval_tick.zero_()
        intent = ml.native_source_hand_part_intent
        role_active = (intent["active"] & intent["known"]).detach().cpu().numpy()
        source_frames = intent["source_frame_index"].detach().cpu().numpy()
        root_gate = (self._perturb_source_gate
                     & ml.native_object_part_pose_support[:, self._object_root_columns]).detach().cpu().numpy()
        motion_starts = ml.length_starts.tolist()
        motion_lengths = ml._motion_num_frames.tolist()
        receipts = []
        motion_ids = self.motion_ids[env_ids].tolist()
        first_steps = self.motion_steps[env_ids].tolist()
        for env_id, motion, first_step, replica in zip(env_ids.tolist(), motion_ids, first_steps, replica_indices, strict=True):
            source = ml.source_lineage[motion]
            original = source["selected_source_frame_index"]
            identity = {key: source[key] for key in ("source_id", "source_revision", "source_unit_id")}
            identity.update(prepared_name=ml.motion_files[motion].rsplit("#", 1)[-1],
                            source_frame_range_inclusive=[int(original[0]), int(original[-1])], replica_index=int(replica))
            start = motion_starts[motion]
            # Reset metrics can advance the command once before the first control solve.
            length = motion_lengths[motion]
            rows = slice(start + first_step, start + min(length, length - 1 + first_step))
            frames = source_frames[rows]
            eligible = root_gate[rows] & (np.arange(len(frames))[:, None] * control_ns >= self._perturb_lag_ns)
            processing = source.get("processing", {})
            compiled = "task_phases" in processing
            phases = processing.get("task_phases" if compiled else "source_task_phases", ())
            provenance = getattr(ml, "native_reference_provenance", None)
            objects = {} if provenance is None else provenance[motion].get("objects", {})
            part_names = [part.body_name for _, part in self.native_layout.part_bindings]
            part_support = ml.native_object_part_pose_support[rows].detach().cpu().numpy()
            bursts, missing = [], []
            for column, binding in enumerate(self.native_layout.objects):
                entity = binding.entity_name
                raw_id = objects.get(entity, {}).get("object_id", entity)
                columns = self._object_part_columns[entity]
                if compiled:
                    critical = [i for i, phase in enumerate(phases)
                                if phase.get("entity_name") == entity
                                and ("coordinate_kind" in phase or phase["kind"] in ("lift", "terminal_support"))]
                    engaged = role_active[start:start + length, :, columns].any()
                    groups = [[i] for i in critical]
                    if not groups and engaged:
                        groups = [[i for i, phase in enumerate(phases)
                                   if phase["kind"] in _PERTURBATION_PHASES
                                   and phase.get("entity_name", entity) == entity]]
                else:
                    groups = [[i for i, phase in enumerate(phases)
                               if phase["kind"] in _PERTURBATION_PHASES
                               and phase.get("object_id", raw_id) == raw_id]] if phases else [[None]]
                occupied = np.zeros(len(frames) * substeps, dtype=bool)
                for phase_indices in groups:
                    windows = []
                    for phase_index in phase_indices:
                        phase = None if phase_index is None else phases[phase_index]
                        mask = eligible[:, column].copy()
                        part_body = None if phase is None else phase.get("part_body_name")
                        target_parts = columns if part_body is None else [part_names.index(part_body)]
                        if compiled and "coordinate_kind" in phase and part_body is None:
                            continue  # A directed event needs its actual contact-bearing child binding.
                        if phase is not None:
                            first, stop = phase["source_frame_range_half_open"]
                            if phase["kind"] == "terminal_support":
                                # The payload can be released before the supported terminal interval ends.
                                receiver = phase["support_entity_name"]
                                mask = (part_support[:, self._object_part_columns[receiver]].any(axis=-1)
                                        & part_support[:, int(self._object_root_columns[column])])
                            else:
                                roles = role_active[rows][:, :, target_parts]
                                roles = roles & part_support[:, None, target_parts]
                                if (side := phase.get("hand_side")) is not None:
                                    roles = roles[:, [ml.hand_sides.index(side)]]
                                mask &= roles.any(axis=(1, 2))
                            mask &= (frames >= first) & (frames < stop)
                        steps = np.flatnonzero(np.repeat(mask, substeps) & ~occupied)
                        for run in np.split(steps, np.flatnonzero(np.diff(steps) != 1) + 1):
                            if len(run):
                                windows.append((int(run[0]), int(run[-1]) + 1, phase_index))
                    if not windows:
                        if compiled:
                            missing.append(dict(entity_name=entity, phase_indices=phase_indices))
                        continue
                    rng = Random(json.dumps([identity, entity, phase_indices], sort_keys=True, separators=(",", ":")))
                    begin, end, phase_index = rng.choice(windows)
                    capacity = end - begin
                    duration = capacity if perturb.continue_prob == 1. else min(
                        capacity, round(1. / (1. - perturb.continue_prob)))
                    begin_tick = begin + rng.randrange(capacity - duration + 1)
                    stop_tick = begin_tick + duration
                    occupied[begin_tick:stop_tick] = True
                    phase = phases[phase_index] if phase_index is not None else dict(
                        kind="source_contact_support", authority="prepared_source_hand_part_intent",
                        source_frame_range_half_open=[int(frames[begin // substeps]),
                                                      int(frames[(end - 1) // substeps]) + 1])
                    vectors = []
                    for scale in (perturb.force_scale, perturb.torque_scale):
                        direction = [rng.gauss(0., 1.) for _ in range(3)]
                        vectors.append([value * scale / math.hypot(*direction) for value in direction])
                    force, torque = vectors
                    burst = dict(entity_name=entity, root_body_name=binding.root_body_name, phase=phase,
                                 start_substep=begin_tick, stop_substep_exclusive=stop_tick,
                                 force_world_n=force, requested_torque_world_nm=torque)
                    if compiled:
                        burst["phase_index"] = phase_index
                    bursts.append(burst)
            status = ("no_usable_source_phase" if not bursts else
                      "incomplete_compiled_phase_coverage" if missing else "planned")
            receipt = dict(schema=f"native_hoi_evaluation_perturbation_v{2 if compiled else 1}", **identity,
                           control_period_ns=control_ns, physics_period_ns=physics_ns,
                           first_reference_motion_step=first_step, status=status, bursts=bursts)
            if compiled:
                receipt["missing_phases"] = missing
            receipts.append(receipt)
        counts = {}
        for env_id, receipt in zip(env_ids.tolist(), receipts, strict=True):
            for burst in receipt["bursts"]:
                key = (env_id, burst["entity_name"])
                counts[key] = counts.get(key, 0) + 1
        shape = (*self._assist_perturb_active.shape, max(counts.values(), default=1))
        self._assist_eval_start = self._assist_eval_start.new_zeros(shape)
        self._assist_eval_stop = torch.zeros_like(self._assist_eval_start)
        self._assist_eval_wrench = self._assist_eval_wrench.new_zeros((*shape, 6))
        columns = {obj.entity_name: column for column, obj in enumerate(self.native_layout.objects)}
        for env_id, receipt in zip(env_ids.tolist(), receipts, strict=True):
            counts = {}
            for burst in receipt["bursts"]:
                column = columns[burst["entity_name"]]
                slot = counts.get(column, 0)
                counts[column] = slot + 1
                self._assist_eval_start[env_id, column, slot] = burst["start_substep"]
                self._assist_eval_stop[env_id, column, slot] = burst["stop_substep_exclusive"]
                self._assist_eval_wrench[env_id, column, slot] = torch.tensor(
                    burst["force_world_n"] + burst["requested_torque_world_nm"], device=self.device)
        return receipts

    def _object_tracking_wrench(self, state, support):
        from .objects.assist import object_pose_pd_wrench

        if self.cfg.object.pin_mode != "xfrc":
            raise ValueError("native object assistance requires the explicit xfrc force mode")
        xfrc = self.cfg.object.xfrc
        reference = self.ref_object_part_state_w
        ml, ids = self.motion_lib, self._motion_flat_ids
        # Unsupported derived rates contribute no derivative error.
        reference = torch.where(support[..., None], reference, state).clone()
        linear_support = ml.native_object_part_linear_velocity_support[ids] & support
        angular_support = ml.native_object_part_angular_velocity_support[ids] & support
        reference[..., 7:10] = torch.where(linear_support[..., None], reference[..., 7:10], state[..., 7:10])
        reference[..., 10:13] = torch.where(angular_support[..., None], reference[..., 10:13], state[..., 10:13])
        if not xfrc.vel_feedforward:
            reference[..., 7:] = 0.0

        def gain(name, scalar):
            current = getattr(self, name)
            return float(scalar) if current is None else current.reshape(self.num_envs, 1, -1)

        kp_rot = gain("_xfrc_kp_rot_env", xfrc.kp.rot)
        kd_rot = gain("_xfrc_kv_rot_env", xfrc.kd.rot)
        inertia_scale = xfrc.rot_inertia_scale
        if xfrc.omega_rot > 0.0:
            kp_rot, kd_rot = float(xfrc.omega_rot) ** 2, 2.0 * float(xfrc.zeta_rot) * float(xfrc.omega_rot)
            inertia_scale = True
        inertia = torch.stack([
            obj.body_inertial_properties().inertia_com_body_kg_m2[:, body]
            for obj, body in zip(self._part_entities, self._part_body_ids, strict=True)
        ], 1)
        contact_scale = float(xfrc.contact_scale)
        if contact_scale < 1.0:
            contact_facts = getattr(self._env.scene, "hand_part_contact_facts", lambda: None)()
            if contact_facts is not None and contact_facts["pairs"]:
                addresses = {
                    (obj.entity_name, part.body_name): index
                    for index, (obj, part) in enumerate(self.native_layout.part_bindings)
                }
                in_contact = torch.zeros(state.shape[:2], dtype=torch.bool, device=self.device)
                for name, pair in contact_facts["pairs"].items():
                    address = contact_facts["addresses"][name]
                    column = addresses[address["part_entity_name"], address["part_body_name"]]
                    in_contact[:, column] |= (
                        contact_facts["measurement_support"]
                        & pair["force_bearing_contact_present"]
                        & (pair["normal_force_n"] >= 0.5)
                    )
                s_rot = torch.where(in_contact[..., None], contact_scale, 1.0)
                kp_rot = kp_rot * s_rot * s_rot
                kd_rot = kd_rot * s_rot
        force, torque, _ = object_pose_pd_wrench(
            state, reference, inertia,
            kp_pos=gain("_xfrc_kp_pos_env", xfrc.kp.pos),
            kd_pos=gain("_xfrc_kv_pos_env", xfrc.kd.pos),
            kp_rot=kp_rot, kd_rot=kd_rot, rot_inertia_scale=inertia_scale,
        )
        if self._xfrc_force_range is not None:
            force = force.clamp(-self._xfrc_force_range, self._xfrc_force_range)
            torque = torque.clamp(-self._xfrc_force_range, self._xfrc_force_range)
        return force, torque

    def _apply_object_assist(self, *, advance_perturbation=False):
        pin = self.cfg.object.pin_objects
        eval_perturbed = self.hoi_evaluation_mode == "perturbed" and self._eval_mode
        if not pin and not eval_perturbed:
            self._assist_force_w.zero_()
            self._assist_torque_w.zero_()
            for obj in self.native_objects.values():
                obj.clear_external_wrenches()
            return
        ctrl = self._xfrc_curr_ctrl
        zero_gains = ctrl is not None and ctrl.has_zero_gains()
        state = self.sim_object_part_state_w
        support = self.motion_lib.native_object_part_pose_support[self._motion_flat_ids]
        if pin and not zero_gains:
            force, torque = self._object_tracking_wrench(state, support)
        else:
            force = torch.zeros_like(self._assist_force_w)
            torque = torch.zeros_like(self._assist_torque_w)
        perturb = self.cfg.object.perturb
        if advance_perturbation and perturb.enabled and (zero_gains or eval_perturbed):
            columns = self._object_root_columns
            if eval_perturbed:
                tick = self._assist_eval_tick[:, None, None]
                scheduled = (tick >= self._assist_eval_start) & (tick < self._assist_eval_stop)
                active = scheduled.any(dim=-1)
                wrench = torch.where(scheduled[..., None], self._assist_eval_wrench, 0.).sum(dim=-2)
                self._assist_eval_tick += 1
            else:
                active = self._assist_perturb_active
                reset_elapsed_ns = self._env.episode_length_buf * round(self._env.step_dt * 1e9)
                eligible = (self._perturb_source_gate[self._motion_flat_ids] & support[:, columns]
                            & (reset_elapsed_ns[:, None] >= self._perturb_lag_ns))
                u = torch.rand(active.shape, device=self.device)
                start = eligible & ~active & (u < float(perturb.start_prob))
                active.copy_(start | (eligible & active & (u < float(perturb.continue_prob))))
                wrench = self._assist_perturb_wrench
                if start.any():
                    count = int(start.sum().item())
                    direction = torch.randn((count, 2, 3), device=self.device, dtype=force.dtype)
                    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    direction[..., 0, :] *= float(perturb.force_scale)
                    direction[..., 1, :] *= float(perturb.torque_scale)
                    wrench[start] = direction.flatten(-2)
                wrench[~active] = 0.0
            # One independent wrench per FREE object, regardless of its passive part count.
            force[:, columns] += wrench[..., :3]
            disturbance_torque = wrench[..., 3:]
            if self._perturb_inverse_inertia_body is not None:
                body_torque = quat_apply_inverse(state[:, columns, 3:7], disturbance_torque)
                body_acceleration = (self._perturb_inverse_inertia_body @ body_torque[..., None]).squeeze(-1)
                limit = self._perturb_max_angular_acceleration_rad_s2
                scale = limit / body_acceleration.norm(dim=-1, keepdim=True).clamp_min(limit)
                # The body and WORLD acceleration norms agree. Scaling the WORLD torque
                # preserves the sampled direction as the struck root changes orientation.
                disturbance_torque = torch.where(active[..., None], disturbance_torque * scale, 0.0)
            torque[:, columns] += disturbance_torque
        self._assist_force_w.copy_(torch.where(support[..., None], force, 0.0).detach())
        self._assist_torque_w.copy_(torch.where(support[..., None], torque, 0.0).detach())
        for entity, obj in self.native_objects.items():
            columns = self._object_part_columns[entity]
            obj.write_external_wrench_to_sim(
                self._assist_force_w[:, columns], self._assist_torque_w[:, columns],
                body_ids=[self._part_body_ids[i] for i in columns],
            )

    def apply_substep(self):
        self._apply_object_assist(advance_perturbation=True)
        # Bounded last-substep submitted-request evidence. This is neither a
        # measured force nor a whole-control-interval average.
        self._assist_last_force_w.copy_(self._assist_force_w)
        self._assist_last_torque_w.copy_(self._assist_torque_w)
        ml, ids = self.motion_lib, self._motion_flat_ids
        self._assist_pose_support.copy_(ml.native_object_part_pose_support[ids])
        self._assist_linear_rate_support.copy_(ml.native_object_part_linear_velocity_support[ids])
        self._assist_angular_rate_support.copy_(ml.native_object_part_angular_velocity_support[ids])
        self._assist_submission_support[:] = True
        clock = getattr(self._env.scene, "physics_time_s", None)
        if clock is not None:
            sample = clock().detach()
            self._assist_sample_time_s.copy_(sample)
            self._assist_clock_support.copy_(torch.isfinite(sample))

    def object_assist_facts(self):
        return dict(
            entity_names=tuple(obj.entity_name for obj, _ in self.native_layout.part_bindings),
            body_names=tuple(part.body_name for _, part in self.native_layout.part_bindings),
            force_w=self._assist_last_force_w, torque_w=self._assist_last_torque_w,
            submission_support=self._assist_submission_support,
            pose_support=self._assist_pose_support,
            linear_rate_support=self._assist_linear_rate_support,
            angular_rate_support=self._assist_angular_rate_support,
            sample_time_s=self._assist_sample_time_s, clock_support=self._assist_clock_support,
            authority="last_physics_substep_submitted_WORLD_body_COM_wrench_not_measured_force",
            unsupported_velocity="no derivative error from unknown rates while source rate support remains false",
            force_unit="N", torque_unit="N m", clock="actual_solver_substep_begin_when_available",
        )

    def clear_object_wrenches(self, env_ids=None):
        for obj in self.native_objects.values():
            obj.clear_external_wrenches(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        for value in (self._assist_force_w, self._assist_torque_w,
                      self._assist_last_force_w, self._assist_last_torque_w, self._assist_perturb_wrench):
            value[rows] = 0
        for value in (self._assist_submission_support, self._assist_pose_support,
                      self._assist_linear_rate_support, self._assist_angular_rate_support,
                      self._assist_clock_support, self._assist_perturb_active):
            value[rows] = False
        self._assist_sample_time_s[rows] = float("nan")
        if self.hoi_evaluation_mode is not None:
            for value in (self._assist_eval_start, self._assist_eval_stop, self._assist_eval_wrench, self._assist_eval_tick):
                value[rows] = 0

    def _init_object_sdf(self, cfg):
        from utils.sdf import bake_object_sdf_grid

        self._obj_sdf_extent = float(cfg.object.sdf.grid_extent)
        self._part_sdf = [
            bake_object_sdf_grid(
                p.mesh_path,
                1.0,
                self._obj_sdf_extent,
                int(cfg.object.sdf.grid_n),
                str(self.device),
            )
            for _, p in self.native_layout.part_bindings
        ]

    def sdf_query_parts(self, world_pts, *, part_state_w=None, env_ids=None, return_support=False):
        from mjlab.utils.lab_api.math import quat_apply, quat_mul

        state = self.sim_object_part_state_w if part_state_w is None else part_state_w
        if env_ids is not None:
            state = state[env_ids]
        distances, gradients, supports = [], [], []
        for i, (_, part) in enumerate(self.native_layout.part_bindings):
            body = state[:, i]
            offset = body.new_tensor(part.body_from_mesh_position_m).expand_as(
                body[:, :3]
            )
            pos = body[:, :3] + quat_apply(body[:, 3:7], offset)
            rotation = body.new_tensor(part.body_from_mesh_quat_wxyz).expand_as(
                body[:, 3:7]
            )
            quat = quat_mul(body[:, 3:7], rotation)[:, None].expand(
                -1, world_pts.shape[1], -1
            )
            local = quat_apply_inverse(quat, world_pts - pos[:, None])
            sample_local = local
            if return_support:
                finite = torch.isfinite(local).all(-1)
                domain = finite & (local.abs() <= self._obj_sdf_extent).all(-1)
                sample_local = torch.where(finite[..., None], local, 0.0)
            grid = (sample_local / self._obj_sdf_extent).flip(-1).reshape(1, -1, 1, 1, 3)
            sample = torch.nn.functional.grid_sample(
                self._part_sdf[i][None], grid, padding_mode="border", align_corners=True
            )
            sample = sample.reshape(4, *world_pts.shape[:2]).permute(1, 2, 0)
            distance, grad = sample[..., 0], sample[..., 1:]
            if part.distance_sign == "unsigned":
                grad = grad * torch.where(distance < 0, -1.0, 1.0)[..., None]
                distance = distance.abs()
            elif part.distance_sign != "winding":
                raise ValueError(
                    f"unknown native part distance sign {part.distance_sign!r}"
                )
            distances.append(distance)
            if return_support:
                supports.append(domain & torch.isfinite(distance))
            gradients.append(
                quat_apply(quat, torch.nn.functional.normalize(grad, dim=-1))
            )
        result = torch.stack(distances, -1), torch.stack(gradients, -2)
        return (*result, torch.stack(supports, -1)) if return_support else result

    def sdf_query(self, world_pts, side, obj_trans=None, obj_quat=None, env_ids=None):
        del side
        if obj_trans is not None or obj_quat is not None:
            raise ValueError(
                "articulated SDF requires all named part poses, not one rigid pose"
            )
        squeeze = world_pts.ndim == 2
        points = world_pts[:, None] if squeeze else world_pts
        distance, gradient = self.sdf_query_parts(points, env_ids=env_ids)
        distance, part = distance.min(-1)
        gradient = gradient.gather(
            -2, part[..., None, None].expand(*part.shape, 1, 3)
        ).squeeze(-2)
        return (distance[:, 0], gradient[:, 0]) if squeeze else (distance, gradient)


@dataclass(kw_only=True)
class NativeMotionTrackingCommandCfg(MotionTrackingCommandCfg):
    native_scene: dict

    def build(self, env):
        return NativeMotionTrackingCommand(self, env)
