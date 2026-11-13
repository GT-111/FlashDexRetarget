"""Isaac motion-tracking command: neutralizes mjwarp sensor registration,
binds the Isaac sensor adapters, restricts sampling in per-env-object mode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.commands.motion_tracking import motion_tracking as _mt
from envs.commands.motion_tracking.motion_tracking import MotionTrackingCommand
from envs.commands.motion_tracking.motion_tracking_cfg import MotionTrackingCommandCfg
from envs.scene.native import NativeSceneCfg


class IsaacMotionTrackingCommand(MotionTrackingCommand):
    def __init__(self, cfg: MotionTrackingCommandCfg, env) -> None:
        orig_raw = _mt.register_raw_contact_views
        orig_mux = _mt.register_multi_object_sensor_views
        _mt.register_raw_contact_views = lambda *a, **k: None
        _mt.register_multi_object_sensor_views = lambda *a, **k: None
        try:
            super().__init__(cfg, env)
        finally:
            _mt.register_raw_contact_views = orig_raw
            _mt.register_multi_object_sensor_views = orig_mux

        bind = getattr(env.scene, "_bind_sensor_callbacks", None)
        if bind is not None:
            side = self._side_list[0]
            bind(lambda: self.active_obj_slot(side), lambda: self)

        # Per-env-object mode: resampling must stay within the env's fixed
        # object; build a padded (S, maxM) per-slot motion table once.
        self._slot_motion_table = None
        self._slot_motion_count = None
        if getattr(env.scene, "assigned_obj_slot", None) is not None:
            self._build_slot_tables()

    def _build_slot_tables(self) -> None:
        """(S, maxM) per-slot draw table: row s lists the motions of object slot s."""
        traj_slot = self.motion_lib.traj_obj_slot[self._side_list[0]]  # (M,)
        S = int(traj_slot.max().item()) + 1
        rows = [torch.where(traj_slot == s)[0] for s in range(S)]
        table = torch.zeros(
            S, max(1, max(int(r.numel()) for r in rows)),
            dtype=torch.long, device=self.device,
        )
        counts = torch.zeros(S, dtype=torch.long, device=self.device)
        for s, r in enumerate(rows):
            table[s, : r.numel()] = r
            counts[s] = r.numel()
        self._slot_motion_table = table
        self._slot_motion_count = counts

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        assigned = getattr(self._env.scene, "assigned_obj_slot", None)
        if (
            assigned is None
            or self._eval_mode
            or self.motion_lib.num_trajectories <= 1
        ):
            return super()._resample_command(env_ids)

        # Pin the draw to the env's own object, then run the base resample
        # (_eval_mode=True only gates the base's motion_ids draw).
        slot = assigned[env_ids]
        counts = self._slot_motion_count[slot].clamp(min=1)
        r = torch.rand(len(env_ids), device=self.device)
        idx = (r * counts.float()).long().clamp(max=int(counts.max().item()) - 1)
        idx = torch.minimum(idx, counts - 1)
        self.motion_ids[env_ids] = self._slot_motion_table[slot, idx]

        prev = self._eval_mode
        self._eval_mode = True
        try:
            super()._resample_command(env_ids)
        finally:
            self._eval_mode = prev

    def _obj_inertia_body(self, side: str) -> torch.Tensor:
        """Per-slot inertia stack (S, 3, 3) for the xfrc pin; the single model
        row can't describe per-env heterogeneous objects."""
        stack = getattr(self._env.scene, "obj_inertia_stack", None)
        if stack is not None and getattr(
            self._env.scene, "assigned_obj_slot", None
        ) is not None:
            if not isinstance(stack, dict):
                return stack
            if side not in stack:
                # shared-object pt: only one side was spawned, the other aliases its entity
                names = self.cfg.object.entity_names
                for s2 in stack:
                    if names.get(s2) == names.get(side):
                        return stack[s2]
            return stack[side]
        return super()._obj_inertia_body(side)


@dataclass(kw_only=True)
class IsaacMotionTrackingCommandCfg(MotionTrackingCommandCfg):
    def build(self, env) -> IsaacMotionTrackingCommand:
        return IsaacMotionTrackingCommand(self, env)


class IsaacNativeMotionTrackingCommand(IsaacMotionTrackingCommand):
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
        ml = self.motion_lib
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

    def _init_site_ids(self, cfg):
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
        ids = [
            self.native_object.body_names.index(p.body_name)
            for p in self.native_layout.object.parts
        ]
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

    def _update_metrics(self):
        from mjlab.utils.lab_api.math import quat_error_magnitude

        super()._update_metrics()
        ml, ids = self.motion_lib, self._motion_flat_ids
        ref = self.ref_native_root_state_w
        root_p = (ref[..., :3] - self.robot_wrist_trans_w).norm(dim=-1)
        root_r = quat_error_magnitude(ref[..., 3:7], self.robot_wrist_quat_w)
        pose_support = torch.stack(
            [ml.native_root_pose_support[s][ids] for s in self._side_list], -1
        )
        for i, side in enumerate(self._side_list):
            # These errors are registered native-body goals, not raw MANO wrist
            # errors. The original MANO reference properties remain unchanged.
            self.metrics[f"error_wrist_trans_{side[0]}"] = torch.where(
                pose_support[:, i], root_p[:, i], 0.0
            )
            self.metrics[f"error_wrist_rot_{side[0]}"] = torch.where(
                pose_support[:, i], root_r[:, i], 0.0
            )
            for key, values, mask in (
                (
                    "lin_vel",
                    ref[:, i, 7:10] - self.robot_wrist_lin_vel_w[:, i],
                    ml.native_root_linear_velocity_support[side][ids],
                ),
                (
                    "ang_vel",
                    ref[:, i, 10:13] - self.robot_wrist_ang_vel_w[:, i],
                    ml.native_root_angular_velocity_support[side][ids],
                ),
            ):
                self.metrics[f"error_wrist_{key}_{side[0]}"] = torch.where(
                    mask, values.abs().mean(-1), 0.0
                )
        parts, part_ref = self.sim_object_part_state_w, self.ref_object_part_state_w
        self.native_step_metrics = {
            "root_position_error_m": root_p,
            "root_rotation_error_rad": root_r,
            "root_pose_support": pose_support,
            "part_position_error_m": (part_ref[..., :3] - parts[..., :3]).norm(dim=-1),
            "part_rotation_error_rad": quat_error_magnitude(
                part_ref[..., 3:7], parts[..., 3:7]
            ),
            "part_pose_support": ml.native_object_part_pose_support[ids],
            "joint_position_error_rad": (
                ml.native_object_joint_pos[ids] - self.native_object.data.joint_pos
            ).abs(),
            "joint_position_support": ml.native_object_joint_position_support[ids],
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
        if tuple(ml.native_object_joint_names) != tuple(obj.joint_names):
            raise ValueError(
                "native object scalar reference order differs from its named articulation"
            )
        if not bool(ml.native_object_joint_position_support[ids].all()):
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
            ml.native_object_joint_velocity_support[ids],
            ml.native_object_joint_vel[ids],
            0.0,
        )
        if self.cfg.object.zero_init_vel:
            state[:, 7:], ov = 0.0, torch.zeros_like(ov)
        obj.write_root_state_to_sim(state, env_ids)
        obj.write_joint_state_to_sim(
            ml.native_object_joint_pos[ids], ov, env_ids=env_ids
        )
        self._obj_state_version = getattr(self, "_obj_state_version", 0) + 1

    def _apply_object_assist(self):
        if self.cfg.object.pin_objects:
            raise ValueError("native articulated task has no object pose assistance")

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
class IsaacNativeMotionTrackingCommandCfg(MotionTrackingCommandCfg):
    native_scene: dict

    def build(self, env):
        return IsaacNativeMotionTrackingCommand(self, env)
