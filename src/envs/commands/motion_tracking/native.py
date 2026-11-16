"""Native reference, clock, task geometry and achieved-state semantics for all engines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers import CommandTerm

from envs.scene.native import NativeSceneCfg

from .motion_tracking import MotionTrackingCommand
from .motion_tracking_cfg import MotionTrackingCommandCfg


class NativeMotionTrackingCommand(MotionTrackingCommand):
    """Registered FREE-root references and ONE named articulated object.

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

    @property
    def object_distance_sign(self) -> str:
        signs = {part.distance_sign for part in self.native_layout.object.parts}
        return next(iter(signs)) if len(signs) == 1 else "mixed"

    def _bind_native_reference(self):
        """Admit the prepared clock and bind declared columns to the actual scene."""
        ml = self.motion_lib
        if not hasattr(ml, "native_time_ns"):
            raise ValueError("native task requires an explicitly prepared native_reference carrier")
        if tuple(ml.native_object_part_body_names) != tuple(
            p.body_name for p in self.native_layout.object.parts
        ):
            raise ValueError("native object reference and scene part bodies differ")
        if ml.native_object_entity_name != self.native_layout.object.entity_name:
            raise ValueError("native object reference names a different scene entity")
        for hand in self.native_layout.hands:
            if ml.native_root_body_names[hand.side] != hand.root_body_name:
                raise ValueError(
                    "native reference root body does not match the selected asset"
                )
        names = tuple(self.native_object.joint_names)
        if set(names) != set(ml.native_object_joint_names):
            raise ValueError("native object reference and actual scalar joint names differ")
        self._object_joint_columns = torch.tensor(
            [ml.native_object_joint_names.index(name) for name in names],
            device=self.device, dtype=torch.long,
        )
        self.object_joint_coordinate_metadata = self.native_object.joint_coordinate_metadata()
        self.robot_joint_coordinate_metadata = self.robot.joint_coordinate_metadata()
        command_columns = self.native_command_joint_ids.tolist()
        self.command_joint_coordinate_metadata = {
            key: tuple(values[i] for i in command_columns)
            for key, values in self.robot_joint_coordinate_metadata.items()
        }

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
        self._part_body_ids = [self.native_object.body_names.index(p.body_name)
                               for p in self.native_layout.object.parts]
        period_ns = float(self._env.step_dt) * 1e9
        for start, count in zip(ml.length_starts.tolist(), ml._motion_num_frames.tolist(), strict=True):
            times = ml.native_time_ns[start:start + count]
            expected = torch.round(torch.arange(count, device=times.device, dtype=torch.float64)
                                   * period_ns).to(torch.int64)
            if times.dtype != torch.int64 or bool(((times - times[0] - expected).abs() > 1).any()):
                raise ValueError("native prepared reference cadence differs from the actual control clock")

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
        return self._finger_joint_ids

    def command_reference(self, ids=None):
        """Independent native commands only; passive mimic joints are not BC actions."""
        ids = self._motion_flat_ids if ids is None else ids
        ml, columns = self.motion_lib, self.native_command_joint_ids
        return {
            **self.command_joint_coordinate_metadata,
            "position": ml.robot_joint_pos[ids][..., columns],
            "velocity": ml.robot_joint_vel[ids][..., columns],
            "position_support": ml.native_joint_position_support[ids][..., columns],
            "velocity_support": ml.native_joint_velocity_support[ids][..., columns],
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

    @property
    def native_object_joint_pos(self):
        if not self.native_object.joint_names:
            return self.robot.data.joint_pos.new_empty(self.num_envs, 0)
        return self.native_object.data.joint_pos

    @property
    def native_object_joint_vel(self):
        if not self.native_object.joint_names:
            return self.robot.data.joint_vel.new_empty(self.num_envs, 0)
        return self.native_object.data.joint_vel

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
    def native_object(self):
        return self._env.scene[self.native_layout.object.entity_name]

    @property
    def sim_object_part_state_w(self):
        ids = self._part_body_ids
        data = self.native_object.data
        return torch.cat(
            [
                getattr(data, f)[:, ids]
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
        from envs._common import action_term

        installed = action_term(self._env).installed_control
        execution = installed["installed_valid"].detach().clone()
        phase = self.reference_phase
        motion_ids = torch.where(execution, installed["reference_motion_ids"], phase["motion_ids"])
        steps = torch.where(execution, installed["reference_motion_steps"], phase["motion_steps"])
        ids = self.motion_lib.length_starts[motion_ids] + steps
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
                part_names=tuple(p.name for p in self.native_layout.object.parts),
                part_body_names=tuple(p.body_name for p in self.native_layout.object.parts),
                part_distance_sign=tuple(p.distance_sign for p in self.native_layout.object.parts),
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
        index = self.motion_lib.native_object_part_body_names.index(
            self.native_layout.object.root_body_name
        )
        state = self.object_part_reference(ids)[..., index, :]
        return state.unsqueeze(-2).expand(*state.shape[:-1], len(self._side_list), 13)

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
        return obj.body_names.index(self.native_layout.object.root_body_name)

    def _init_metrics(self):
        # Native reference-tracking metrics do not inherit raw-MANO wrist or
        # rigid-side contact statistics as if they described the registered task.
        self.metrics["native_root_position_error_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["native_part_position_error_m"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        # Use the real command lifecycle without the stock rigid contact accumulator.
        return CommandTerm.reset(self, env_ids)

    def _update_metrics(self):
        from mjlab.utils.lab_api.math import quat_error_magnitude

        # This override bypasses common metrics, including its pre-advance phase.
        self.reference_phase = {
            "motion_ids": self.motion_ids.clone(),
            "motion_steps": self.motion_steps.clone(),
        }
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
            command_ref["position"] - self.robot.data.joint_pos[:, self.native_command_joint_ids]
        ).abs()
        command_angular = self.command_joint_columns_by_unit["rad"]
        command_linear = self.command_joint_columns_by_unit["m"]
        part_p = (part_ref[..., :3] - parts[..., :3]).norm(dim=-1)
        part_support = ml.native_object_part_pose_support[ids]
        self.metrics["native_root_position_error_m"] = (
            torch.where(pose_support, root_p, 0.0).sum(-1) / pose_support.sum(-1).clamp_min(1)
        )
        self.metrics["native_part_position_error_m"] = (
            torch.where(part_support, part_p, 0.0).sum(-1) / part_support.sum(-1).clamp_min(1)
        )
        self.native_step_metrics = {
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
        obj = self.native_object
        object_ref = self.object_joint_reference(ids)
        if not bool(object_ref["position_support"].all()):
            raise ValueError("native reset requires the passive articulation pose")
        base = ml.native_object_part_body_names.index(
            self.native_layout.object.root_body_name
        )
        if not bool(ml.native_object_part_pose_support[ids, base].all()):
            raise ValueError("native reset requires the object FREE-base pose")
        state = ml.native_object_part_state_w[ids, base].clone()
        state[:, :3] += self._env.scene.env_origins[env_ids]
        state[:, 7:10] = torch.where(
            ml.native_object_part_linear_velocity_support[ids, base, None],
            state[:, 7:10],
            0.0,
        )
        state[:, 10:13] = torch.where(
            ml.native_object_part_angular_velocity_support[ids, base, None],
            state[:, 10:13],
            0.0,
        )
        ov = torch.where(
            object_ref["velocity_support"],
            object_ref["velocity"],
            0.0,
        )
        if self.cfg.object.zero_init_vel:
            state[:, 7:], ov = 0.0, torch.zeros_like(ov)
        obj.write_root_state_to_sim(state, env_ids)
        if obj.joint_names:
            obj.write_joint_state_to_sim(object_ref["position"], ov, env_ids=env_ids)
        self.clear_object_wrenches(env_ids)
        self._obj_state_version = getattr(self, "_obj_state_version", 0) + 1

    def _init_native_object_assist(self):
        shape = (self.num_envs, len(self.native_layout.object.parts), 3)
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

    def _apply_object_assist(self):
        from .objects.assist import object_pose_pd_wrench

        obj = self.native_object
        if not self.cfg.object.pin_objects:
            self._assist_force_w.zero_()
            self._assist_torque_w.zero_()
            obj.clear_external_wrenches()
            return
        if self.cfg.object.pin_mode != "xfrc":
            raise ValueError("native object assistance requires the explicit xfrc force mode")
        xfrc = self.cfg.object.xfrc
        state = self.sim_object_part_state_w
        reference = self.ref_object_part_state_w
        ml, ids = self.motion_lib, self._motion_flat_ids
        support = ml.native_object_part_pose_support[ids]
        # Unknown poses never enter quaternion math. Unknown velocity targets
        # mean zero feedforward, not an observed zero source rate.
        reference = torch.where(support[..., None], reference, state).clone()
        linear_support = ml.native_object_part_linear_velocity_support[ids] & support
        angular_support = ml.native_object_part_angular_velocity_support[ids] & support
        reference[..., 7:10] = torch.where(linear_support[..., None], reference[..., 7:10], 0.0)
        reference[..., 10:13] = torch.where(angular_support[..., None], reference[..., 10:13], 0.0)
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
        inertia = obj.body_inertial_properties().inertia_com_body_kg_m2[:, self._part_body_ids]
        force, torque, _ = object_pose_pd_wrench(
            state, reference, inertia,
            kp_pos=gain("_xfrc_kp_pos_env", xfrc.kp.pos),
            kd_pos=gain("_xfrc_kv_pos_env", xfrc.kd.pos),
            kp_rot=kp_rot, kd_rot=kd_rot, rot_inertia_scale=inertia_scale,
        )
        if self._xfrc_force_range is not None:
            force = force.clamp(-self._xfrc_force_range, self._xfrc_force_range)
            torque = torque.clamp(-self._xfrc_force_range, self._xfrc_force_range)
        self._assist_force_w.copy_(torch.where(support[..., None], force, 0.0).detach())
        self._assist_torque_w.copy_(torch.where(support[..., None], torque, 0.0).detach())
        obj.write_external_wrench_to_sim(
            self._assist_force_w, self._assist_torque_w, body_ids=self._part_body_ids,
        )

    def apply_substep(self):
        self._apply_object_assist()
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
            body_names=tuple(part.body_name for part in self.native_layout.object.parts),
            force_w=self._assist_last_force_w, torque_w=self._assist_last_torque_w,
            submission_support=self._assist_submission_support,
            pose_support=self._assist_pose_support,
            linear_rate_support=self._assist_linear_rate_support,
            angular_rate_support=self._assist_angular_rate_support,
            sample_time_s=self._assist_sample_time_s, clock_support=self._assist_clock_support,
            authority="last_physics_substep_submitted_WORLD_body_COM_wrench_not_measured_force",
            unsupported_velocity="zero feedforward target; independent source rate support remains false",
            force_unit="N", torque_unit="N m", clock="actual_solver_substep_begin_when_available",
        )

    def clear_object_wrenches(self, env_ids=None):
        self.native_object.clear_external_wrenches(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        for value in (self._assist_force_w, self._assist_torque_w,
                      self._assist_last_force_w, self._assist_last_torque_w):
            value[rows] = 0
        for value in (self._assist_submission_support, self._assist_pose_support,
                      self._assist_linear_rate_support, self._assist_angular_rate_support,
                      self._assist_clock_support):
            value[rows] = False
        self._assist_sample_time_s[rows] = float("nan")

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
            for p in self.native_layout.object.parts
        ]

    def sdf_query_parts(self, world_pts, *, part_state_w=None, env_ids=None, return_support=False):
        from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

        state = self.sim_object_part_state_w if part_state_w is None else part_state_w
        if env_ids is not None:
            state = state[env_ids]
        distances, gradients, supports = [], [], []
        for i, part in enumerate(self.native_layout.object.parts):
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
