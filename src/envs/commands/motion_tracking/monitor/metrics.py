"""MotionTrackingCommand mixin: tracking-error and contact metrics."""

from __future__ import annotations

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude


class MetricsMixin:
    def _init_metrics(self) -> None:
        # ── Tracking errors ────────────────────────────────────────────────────────
        side_prefixes = {"right": "r", "left": "l"}
        for side in self.motion_lib.hand_sides:
            p = side_prefixes[side]
            for key in (
                "error_wrist_trans",
                "error_wrist_rot",
                "error_wrist_lin_vel",
                "error_wrist_ang_vel",
                "error_joint_vel",
            ):
                self.metrics[f"{key}_{p}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
            for finger in self.finger_names:
                self.metrics[f"error_tip_trans_{p}_{finger}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
            for level in self._levels():
                self.metrics[f"error_level{level}_{p}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
            if self.has_objects:
                for key in (
                    "error_obj_trans",
                    "error_obj_rot",
                    "error_obj_lin_vel",
                    "error_obj_ang_vel",
                ):
                    self.metrics[f"{key}_{p}"] = torch.zeros(
                        self.num_envs, device=self.device
                    )

        # ── Contact accumulators ───────────────────────────────────────────────────
        self._contact_sum: dict[str, torch.Tensor] = {}
        self._contact_count_ref: dict[str, torch.Tensor] = {}
        self._contact_count_contact: dict[str, torch.Tensor] = {}
        if self.has_objects:
            for side in self.motion_lib.hand_sides:
                p = side_prefixes[side]
                for finger in self.finger_names:
                    k = f"{p}_{finger}"
                    self._contact_count_ref[k] = torch.zeros(
                        self.num_envs, device=self.device
                    )
                    self._contact_count_contact[k] = torch.zeros(
                        self.num_envs, device=self.device
                    )
                    for name in ("ref_dist", "pen", "force"):
                        self._contact_sum[f"contact_{name}_{k}"] = torch.zeros(
                            self.num_envs, device=self.device
                        )

        n_sides, n_fingers = len(self._side_list), len(self.finger_names)
        self.contact_miss_counter = torch.zeros(
            self.num_envs, n_sides, n_fingers, device=self.device
        )
        self.contact_miss_max = torch.zeros_like(self.contact_miss_counter)

    def _record_reference_phase(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or not hasattr(self, "reference_phase"):
            self.reference_phase = {
                "motion_ids": self.motion_ids.clone(),
                "motion_steps": self.motion_steps.clone(),
            }
        else:
            self.reference_phase["motion_ids"][env_ids] = self.motion_ids[env_ids]
            self.reference_phase["motion_steps"][env_ids] = self.motion_steps[env_ids]

    def _set_metric(self, name, value, env_ids) -> None:
        if env_ids is None:
            self.metrics[name] = value
        else:
            self.metrics[name][env_ids] = value[env_ids]

    def _update_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        # CommandTerm.compute advances the cursor after evaluating these metrics.
        self._record_reference_phase(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        side_prefixes = {"right": "r", "left": "l"}
        for si, side in enumerate(self._side_list):
            p = side_prefixes[side]

            # ── Hand tracking ──────────────────────────────────────────────────────
            self._set_metric(f"error_wrist_trans_{p}", torch.norm(
                self.mano_wrist_trans_w[:, si] - self.robot_wrist_trans_w[:, si], dim=-1
            ), env_ids)

            self._set_metric(f"error_wrist_rot_{p}", quat_error_magnitude(
                self.mano_wrist_quat_w[:, si], self.robot_wrist_quat_w[:, si]
            ), env_ids)

            tip_err = torch.norm(
                self.mano_tip_trans_w[:, si] - self.robot_tip_trans_w[:, si], dim=-1
            )  # (B, 5)
            for fi, finger in enumerate(self.finger_names):
                self._set_metric(f"error_tip_trans_{p}_{finger}", tip_err[:, fi], env_ids)

            for level in self._levels():
                mano_trans = self.mano_level_trans_w(side, level)  # (B, 5, 3)
                robot_trans = self.robot_level_trans_w(side, level)  # (B, 5, 3)
                self._set_metric(f"error_level{level}_{p}", torch.norm(
                    mano_trans - robot_trans, dim=-1
                ).mean(dim=-1), env_ids)

            self._set_metric(f"error_wrist_lin_vel_{p}", torch.mean(
                torch.abs(
                    self.mano_wrist_lin_vel_w[:, si] - self.robot_wrist_lin_vel_w[:, si]
                ),
                dim=-1,
            ), env_ids)
            self._set_metric(f"error_wrist_ang_vel_{p}", torch.mean(
                torch.abs(
                    self.mano_wrist_ang_vel_w[:, si] - self.robot_wrist_ang_vel_w[:, si]
                ),
                dim=-1,
            ), env_ids)

            # Matches joints_vel_error_exp.
            body_delta = self.mano_all_joints_lin_vel_w(
                side
            ) - self.robot_all_joints_lin_vel_w(side)  # (B, 12, 3)
            tip_mano_vel = self.mano_tip_lin_vel_w[:, si]  # (B, 5, 3)
            tip_robot_vel = self.robot_tip_lin_vel_w[:, si]  # (B, 5, 3)
            all_delta = torch.cat([body_delta, tip_mano_vel - tip_robot_vel], dim=1)
            self._set_metric(f"error_joint_vel_{p}", (
                all_delta.abs().mean(dim=-1).mean(dim=-1)
            ), env_ids)

            # ── Object tracking + contact ──────────────────────────────────────────
            # Mirrors the obj_*_error_exp rewards.
            if self.has_objects:
                self._set_metric(f"error_obj_trans_{p}", torch.norm(
                    self.ref_obj_trans_w[:, si] - self.sim_obj_trans_w[:, si], dim=-1
                ), env_ids)
                self._set_metric(f"error_obj_rot_{p}", quat_error_magnitude(
                    self.ref_obj_quat_w[:, si], self.sim_obj_quat_w[:, si]
                ), env_ids)
                self._set_metric(f"error_obj_lin_vel_{p}", torch.mean(
                    torch.abs(
                        self.ref_obj_lin_vel_w[:, si] - self.sim_obj_lin_vel_w[:, si]
                    ),
                    dim=-1,
                ), env_ids)
                self._set_metric(f"error_obj_ang_vel_{p}", torch.mean(
                    torch.abs(
                        self.ref_obj_ang_vel_w[:, si] - self.sim_obj_ang_vel_w[:, si]
                    ),
                    dim=-1,
                ), env_ids)

                # ref_flag gates ref_dist + found; ref_flag AND found gates pen + force.
                pen_sensor: ContactSensor = self._env.scene[
                    f"{p}_fingertip_penetration"
                ]
                force_sensor: ContactSensor = self._env.scene[f"{p}_fingertip_contact"]
                for fi, finger in enumerate(self.finger_names):
                    k = f"{p}_{finger}"
                    flag = self.ref_contact_flags[:, si, fi]
                    found = (pen_sensor.data.found[:, fi] > 0).to(flag.dtype)
                    contact_gate = flag * found
                    ref_dist = torch.norm(
                        self.ref_contact_trans_w[:, si, fi]
                        - self.robot_tip_trans_w[:, si, fi],
                        dim=-1,
                    )
                    pen = torch.clamp(-pen_sensor.data.dist[:, fi], min=0.0)
                    force = torch.norm(force_sensor.data.force[:, fi], dim=-1)
                    self._contact_sum[f"contact_ref_dist_{k}"][rows] += (ref_dist * flag)[rows]
                    self._contact_sum[f"contact_pen_{k}"][rows] += (pen * contact_gate)[rows]
                    self._contact_sum[f"contact_force_{k}"][rows] += (force * contact_gate)[rows]
                    self._contact_count_ref[k][rows] += flag[rows]
                    self._contact_count_contact[k][rows] += contact_gate[rows]
                    # Consecutive (flag AND not found) streak; resets otherwise.
                    miss = flag * (1.0 - found)
                    self.contact_miss_counter[rows, si, fi] = (
                        self.contact_miss_counter[rows, si, fi] + 1.0
                    ) * miss[rows]
                    self.contact_miss_max[rows, si, fi] = torch.maximum(
                        self.contact_miss_max[rows, si, fi],
                        self.contact_miss_counter[rows, si, fi],
                    )

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if not self.has_objects:
            return extras
        side_prefixes = {"right": "r", "left": "l"}
        for si, side in enumerate(self._side_list):
            p = side_prefixes[side]
            for fi, finger in enumerate(self.finger_names):
                k = f"{p}_{finger}"
                count_ref = self._contact_count_ref[k]
                count_contact = self._contact_count_contact[k]
                total_ref = count_ref[env_ids].sum().clamp(min=1.0)
                total_contact = count_contact[env_ids].sum().clamp(min=1.0)
                extras[f"contact_ref_dist_{k}"] = (
                    self._contact_sum[f"contact_ref_dist_{k}"][env_ids].sum()
                    / total_ref
                ).item()
                extras[f"contact_found_{k}"] = (
                    count_contact[env_ids].sum() / total_ref
                ).item()
                extras[f"contact_pen_{k}"] = (
                    self._contact_sum[f"contact_pen_{k}"][env_ids].sum() / total_contact
                ).item()
                extras[f"contact_force_{k}"] = (
                    self._contact_sum[f"contact_force_{k}"][env_ids].sum()
                    / total_contact
                ).item()
                extras[f"contact_miss_max_{k}"] = (
                    self.contact_miss_max[env_ids, si, fi].mean().item()
                )
                for name in ("ref_dist", "pen", "force"):
                    self._contact_sum[f"contact_{name}_{k}"][env_ids] = 0.0
                count_ref[env_ids] = 0.0
                count_contact[env_ids] = 0.0
                self.contact_miss_counter[env_ids, si, fi] = 0.0
                self.contact_miss_max[env_ids, si, fi] = 0.0
        return extras
