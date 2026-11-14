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
        bind = getattr(env.scene, "_bind_sensor_callbacks", None)
        if bind is not None:
            bind(lambda: self.active_obj_slot(self._side_list[0]), lambda: self)
        self._bind_native_reference()

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

    def _levels(self):
        return tuple(
            level for level in (1, 2) if self.cfg.body_mapping.get(f"level{level}")
        )

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
            "joint_position_error_rad": (
                object_ref["position"] - self.native_object_joint_pos
            ).abs(),
            "joint_position_support": object_ref["position_support"],
            "command_position_error_rad": (
                command_ref["position"] - self.robot.data.joint_pos[:, self.native_command_joint_ids]
            ).abs(),
            "command_position_support": command_ref["position_support"],
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

    def _apply_object_assist(self):
        if self.cfg.object.pin_objects:
            raise ValueError("native articulated task has no object pose assistance")

    def clear_object_wrenches(self, env_ids=None):
        self.native_object.clear_external_wrenches(env_ids)

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

    def sdf_query_parts(self, world_pts, *, part_state_w=None, env_ids=None):
        from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

        state = self.sim_object_part_state_w if part_state_w is None else part_state_w
        if env_ids is not None:
            state = state[env_ids]
        distances, gradients = [], []
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
            grid = (local / self._obj_sdf_extent).flip(-1).reshape(1, -1, 1, 1, 3)
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
            gradients.append(
                quat_apply(quat, torch.nn.functional.normalize(grad, dim=-1))
            )
        return torch.stack(distances, -1), torch.stack(gradients, -2)

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
