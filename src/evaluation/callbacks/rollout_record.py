"""Eval capture for first-success and explicit frozen-attempt exports; one flat
device->host copy per step, split back into fields at export time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RolloutRecord:
    """``[joint_pos | joint_vel | actions | root_pos | root_quat | obj_pos | obj_quat]`` per
    eval step; positions are env-local (env_origins subtracted)."""

    def __init__(
        self,
        command_name: str = "motion",
        enabled: bool = True,
    ) -> None:
        self.command_name = str(command_name)
        self.enabled = bool(enabled)
        self.rollout_steps = 1
        self._frames: list[torch.Tensor] = []
        self._roll_spec: list[tuple[str, int, tuple[int, ...]]] = []

    # ── lifecycle ────────────────────────────────────────────────────────────
    def on_eval_setup(self, env: Any) -> None:
        del env

    def on_start(self, env: Any) -> None:
        if not self.enabled:
            return
        self.cmd = env.command_manager.get_term(self.command_name)
        self.num_envs = int(env.num_envs)
        self._env_origins = env.scene.env_origins.detach().clone()
        self.num_sides = int(self.cmd.sim_obj_trans_w.shape[1])
        # Per-env clip length, as the other eval callbacks trim to.
        self.valid_steps_per_env = self.cmd.motion_lib._motion_num_frames[
            self.cmd.motion_ids
        ].to(torch.long)
        self.t = 0
        self._frames = []
        robot = self.cmd.robot
        n_act = int(env.action_manager.action.shape[-1])
        tail = () if self.num_sides == 1 else (self.num_sides,)
        self._roll_spec = [
            ("joint_pos", int(robot.data.joint_pos.shape[-1]), ()),
            ("joint_vel", int(robot.data.joint_vel.shape[-1]), ()),
            ("actions", n_act, ()),
            ("root_pos", 3, ()),
            ("root_quat", 4, ()),
            ("obj_pos", 3 * self.num_sides, tail + (3,)),
            ("obj_quat", 4 * self.num_sides, tail + (4,)),
        ]

    def on_step(self, env: Any) -> None:
        if not self.enabled:
            return
        self.t += 1
        cmd = self.cmd
        robot = cmd.robot
        origins = self._env_origins
        n = self.num_envs
        flat = torch.cat(
            [
                robot.data.joint_pos,
                robot.data.joint_vel,
                env.action_manager.action,
                robot.data.root_link_pos_w - origins,
                robot.data.root_link_quat_w,
                (cmd.sim_obj_trans_w - origins[:, None, :]).reshape(n, -1),
                cmd.sim_obj_quat_w.reshape(n, -1),
            ],
            dim=-1,
        )
        self._frames.append(flat.detach().to("cpu", torch.float32))

    # ── cross-batch reduction ────────────────────────────────────────────────
    def collect_state(self, active_mask: torch.Tensor | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"rollout": None, "valid_steps_per_env": torch.zeros(0)}
        sel = slice(None) if active_mask is None else active_mask.cpu()
        rollout = torch.stack(self._frames, dim=0)[:, sel] if self._frames else None
        return {
            "rollout": rollout,
            "valid_steps_per_env": self.valid_steps_per_env[
                slice(None) if active_mask is None else active_mask
            ].cpu(),
            "motion_ids": self.cmd.motion_ids[
                slice(None) if active_mask is None else active_mask
            ].cpu().clone(),
        }

    def reduce_state(self, states: list[dict[str, Any]]) -> dict[str, float]:
        del states
        return {}

    def get_metrics(self) -> dict[str, float]:
        return {}

    # ── export helper ────────────────────────────────────────────────────────
    def rollout_arrays(
        self, states: list[dict[str, Any]], env_index: int
    ) -> dict[str, Any] | None:
        """Recorded state for one env of the concatenated sweep, trimmed to its clip."""
        offset = 0
        for s in states:
            n = int(s["valid_steps_per_env"].shape[0])
            if env_index < offset + n:
                if s["rollout"] is None:
                    return None
                row = env_index - offset
                roll = s["rollout"]
                n_frames = min(
                    int(s["valid_steps_per_env"][row].item()), int(roll.shape[0])
                )
                if n_frames <= 0:
                    return None
                clip = roll[:n_frames, row]
                out: dict[str, Any] = {}
                at = 0
                for name, width, tail in self._roll_spec:
                    chunk = clip[:, at : at + width]
                    out[name] = (
                        chunk.reshape(n_frames, *tail).numpy() if tail else chunk.numpy()
                    )
                    at += width
                out["motion_id"] = int(s["motion_ids"][row].item())
                return out
            offset += n
        raise IndexError(f"env_index {env_index} out of range for {offset} envs")


_NATIVE_SIDES = ("left", "right")
_NATIVE_RECORDING = "dexforge-flashdex-achieved-state-v2"
_INSTALLED_ARRAYS = (
    "finger_target_rad", "finger_source_support", "root_state_w", "root_pose_support",
    "root_linear_velocity_support", "root_angular_velocity_support", "installed_valid",
    "reference_motion_ids", "reference_motion_steps", "root_force_n", "root_torque_nm",
)
_CONTACT_ARRAYS = (
    "normal_force_n", "force_bearing_contact_present",
    "force_bearing_distance_max_m", "measurement_support",
)
_CONTACT_INTERVAL_ARRAYS = (
    "normal_force_n", "force_bearing_contact_present", "force_bearing_distance_max_m",
    "measurement_support", "sample_recorded", "clock_support",
    "sample_begin_time_s", "sample_end_time_s", "interval_complete",
)


class NativeRolloutRecord(RolloutRecord):
    """Record reset and actual named hand bodies for native reference export.

    Select this callback with a literal run binding. Raw manager actions remain
    separate from any installed-control readout supplied by the actual action
    term. Full source IDs/crop mapping come from the import, never from FPS alone.
    """
    def __init__(self, *, binding: dict | str, command_name: str = "motion"):
        super().__init__(command_name=command_name)
        self.binding = json.loads(Path(binding).read_text()) if isinstance(binding, str) else dict(binding)

    def on_start(self, env):
        super().on_start(env)
        robot = self.cmd.robot
        self.joint_names = tuple(robot.joint_names)
        self.body_names = tuple(robot.body_names)
        self.hand_indices = tuple(self.body_names.index(self.binding["hands"][side]["body_name"])
                                  for side in _NATIVE_SIDES)
        for hand in self.binding["hands"].values():
            for name in hand["physical_joint_names"].values():
                self.joint_names.index(name)
        self.object_parts = []
        addresses = set()
        for part in self.binding["object_parts"].values():
            address = (part["entity_name"], part["body_name"])
            if address in addresses:
                raise ValueError("FlashDex shared object aliases cannot become two distinct physical parts")
            addresses.add(address)
            entity = env.scene[part["entity_name"]]
            self.object_parts.append((entity, entity.body_names.index(part["body_name"])))
        if tuple(self.binding["object_parts"]) != tuple(self.binding["object_part_names"]):
            raise ValueError("FlashDex named part order must match the actual scene-body bindings")
        self.step_dt = float(env.step_dt)
        if not np.isfinite(self.step_dt) or self.step_dt <= 0:
            raise ValueError("FlashDex recorder requires the actual positive environment control interval")
        self.motion_names = [name.rsplit("#", 1)[-1] for name in self.cmd.motion_lib.motion_files]
        for index, name in enumerate(self.motion_names):
            clip = self.binding["clips"][name]
            if len(clip["prepared_time_ns"]) != int(self.cmd.motion_lib._motion_num_frames[index]):
                raise ValueError("FlashDex prepared source clock differs from the actual motion library")
            lineage = self.cmd.motion_lib.source_lineage[index]
            ordinal = self.cmd.motion_lib.source_query_ordinal[index]
            elapsed = self.cmd.motion_lib.source_import_elapsed_ns[index]
            if (lineage is None or ordinal is None or elapsed is None
                    or not np.array_equal(ordinal, clip["prepared_source_ordinal"])
                    or not np.array_equal(elapsed - elapsed[0], clip["prepared_elapsed_time_ns"])):
                raise ValueError("FlashDex native binding differs from the actual imported frame/scene lineage")
            native_provenance = getattr(self.cmd.motion_lib, "native_reference_provenance", None)
            if native_provenance is not None:
                provenance = native_provenance[index]
                # Prepared native references own placement separately from their
                # canonical source mapping; neither is an achieved-state claim.
                matches = (lineage == clip and provenance.get("source") == clip
                    and provenance.get("authority") == "requested_reference_not_achieved_state"
                    and np.array_equal(provenance.get("scene_from_source_world"),
                                       self.binding["scene_from_source_world"]))
            else:
                matches = (lineage.get("capture_id") == clip["source_unit_id"]
                    and np.array_equal(lineage["selected_source_frame_index"], clip["selected_source_frame_index"])
                    and np.allclose(lineage["scene_from_source_world"], self.binding["scene_from_source_world"],
                                    atol=1e-9, rtol=0))
            if not matches:
                raise ValueError("FlashDex native binding differs from the actual imported frame/scene lineage")
        installed = [(name, env.action_manager.get_term(name)) for name in env.action_manager.active_terms
                     if hasattr(env.action_manager.get_term(name), "installed_control")]
        if len(installed) > 1:
            raise ValueError("FlashDex native export requires one combined installed-control owner")
        self.installed_term = installed[0][1] if installed else None
        self.installed_metadata = None
        if self.installed_term is not None:
            control = self.installed_term.installed_control
            for side, body in zip(control["hand_sides"], control["hand_body_names"], strict=True):
                if self.binding["hands"][side]["body_name"] != body:
                    raise ValueError("FlashDex installed root body differs from its recorded native hand")
            self.installed_metadata = {field: control[field] for field in (
                "hand_sides", "hand_body_names", "finger_actuator_names", "finger_joint_names",
                "root_control", "action_recipe")}
            self.installed_metadata.update(term_name=installed[0][0],
                coordinate_frame="env-local p/WXYZ; WORLD link twist and hand-root force/torque",
                effort_sampling="actual last physics substep; not interval-mean effort")
        self.object_joint_metadata = None
        if hasattr(self.cmd, "native_object"):
            self.object_joint_metadata = dict(entity_name=self.cmd.native_layout.object.entity_name,
                joint_names=list(self.cmd.native_object.joint_names),
                authority="observed simulator scalar state; independent from reference support")
        self._object_assist_reader = getattr(self.cmd, "object_assist_facts", None)
        self.object_assist_metadata = None
        if self._object_assist_reader is not None:
            facts = self._object_assist_reader()
            names = [part["body_name"] for part in self.binding["object_parts"].values()]
            self._object_assist_columns = [facts["body_names"].index(name) for name in names]
            self.object_assist_metadata = {key: facts[key] for key in (
                "authority", "unsupported_velocity", "force_unit", "torque_unit", "clock",
            )}
            self.object_assist_metadata.update(
                body_names=names, sampling="last submitted physics substep, not interval mean or measured force",
                reset="no submitted substep: zero request and false submission support",
            )
        self.backend_metadata = getattr(env.scene, "backend_identity", None)
        self._physics_time_reader = getattr(env.scene, "physics_time_s", None)
        self._execution_clock_reader = getattr(env.scene, "physics_execution_clock", None)
        execution = None if self._execution_clock_reader is None else self._execution_clock_reader()
        self.execution_clock_metadata = None if execution is None else dict(execution["metadata"])
        self._contact_reader = getattr(env.scene, "hand_part_contact_facts", None)
        association_reader = getattr(env.scene, "tracking_contact_metadata", None)
        self.contact_association_metadata = None if association_reader is None else association_reader()
        self.contact_metadata = None
        facts = None if self._contact_reader is None else self._contact_reader()
        if facts is not None:
            self._contact_names = []
            addresses = []
            for side in _NATIVE_SIDES:
                for part in self.binding["object_parts"].values():
                    address = dict(hand_side=side, hand_body_name=self.binding["hands"][side]["body_name"],
                                   part_entity_name=part["entity_name"], part_body_name=part["body_name"])
                    names = [n for n, a in facts["addresses"].items() if a == address]
                    if len(names) != 1:
                        raise ValueError("native contact export requires the actual hand/part body pair")
                    self._contact_names.append(names[0])
                    addresses.append(address)
            self.contact_metadata = dict(
                schema="solver_hand_part_contact_v1", pairs=addresses,
                clock=facts["clock"], normal_force_authority=facts["normal_force_authority"],
                distance_authority=facts["distance_authority"],
                absent_contact="measured zero normal force, false presence, unavailable separation",
                unsupported="NaN normal force/separation and false measurement support",
                provenance="derived target execution evidence, not original observed source contacts",
            )
        self._contact_interval_reader = getattr(env.scene, "hand_part_contact_interval_facts", None)
        interval = None if self._contact_interval_reader is None else self._contact_interval_reader()
        self.contact_interval_metadata = None
        if interval is not None:
            self.contact_interval_metadata = dict(
                schema="solver_hand_part_control_interval_v1",
                pairs=[interval["addresses"][name] for name in self._contact_names],
                num_substeps=interval["num_substeps"], clock=interval["clock"], sampling=interval["sampling"],
                normal_force_authority=interval["normal_force_authority"],
                distance_authority=interval["distance_authority"],
                pair_field_axes="recorded endpoint, configured substep, named pair",
                common_field_axes="recorded endpoint, configured substep; interval_complete is endpoint-only",
                interval_complete="all configured substeps recorded with valid contiguous solver clocks; force support remains independent",
                reset="no executed interval; NaN readings and false support/complete",
                source_required_contact_phase="UNKNOWN; no source engagement is inferred from actual force or hinge sign",
                provenance="derived target execution evidence, not original observed source contacts",
            )
            if "execution_clock" in interval:
                self.contact_interval_metadata["execution_clock"] = dict(interval["execution_clock"])
        self.native_frames = []
        phase = self.cmd.reference_phase
        self.valid_steps_per_env = (
            self.cmd.motion_lib._motion_num_frames[phase["motion_ids"]] - phase["motion_steps"]
        )
        self.rollout_steps = int((self.valid_steps_per_env - 1).clamp_min(0).max())
        # The author on_start does not capture reset. Its recorder can retain
        # this factual row without changing the policy or environment loop.
        self.on_step(env)

    def on_step(self, env):
        import torch

        super().on_step(env)
        robot = self.cmd.robot
        pos = robot.data.body_link_pos_w[:, self.hand_indices] - self._env_origins[:, None]
        quat = robot.data.body_link_quat_w[:, self.hand_indices]
        hands = torch.cat((pos, quat), dim=-1).detach().to("cpu", torch.float32)
        hand_twist = torch.cat((robot.data.body_link_lin_vel_w[:, self.hand_indices],
                                robot.data.body_link_ang_vel_w[:, self.hand_indices]), dim=-1)
        part_position = torch.stack([entity.data.body_link_pos_w[:, body] for entity, body in self.object_parts], dim=1)
        part_position -= self._env_origins[:, None]
        part_quaternion = torch.stack([entity.data.body_link_quat_w[:, body] for entity, body in self.object_parts], dim=1)
        parts = torch.cat((part_position, part_quaternion), dim=-1).detach().to("cpu", torch.float32)
        pinning = bool(self.cmd.cfg.object.pin_objects)
        curriculum = self.cmd._xfrc_curr_ctrl is not None
        phase = self.cmd.reference_phase
        frame = {
            "hands": hands, "hand_twist_w": hand_twist.detach().to("cpu", torch.float32), "parts": parts,
            # Metrics observe the achieved endpoint before compute advances the
            # next-target cursor. In particular, the final cursor is clamped.
            "motion_steps": phase["motion_steps"].detach().to("cpu", torch.int64).clone(),
            "motion_ids": phase["motion_ids"].detach().to("cpu", torch.int64).clone(),
            "command_motion_steps": self.cmd.motion_steps.detach().to("cpu", torch.int64).clone(),
            "command_motion_ids": self.cmd.motion_ids.detach().to("cpu", torch.int64).clone(),
            "pinning": pinning, "curriculum": curriculum,
        }
        if self._object_assist_reader is not None:
            facts = self._object_assist_reader()
            frame["object_assist"] = {
                key: facts[key][:, self._object_assist_columns].detach().cpu().clone()
                for key in ("force_w", "torque_w", "pose_support", "linear_rate_support", "angular_rate_support")
            }
            frame["object_assist"].update({
                key: facts[key].detach().cpu().clone()
                for key in ("submission_support", "sample_time_s", "clock_support")
            })
        if self.installed_term is not None:
            control = self.installed_term.installed_control
            frame["installed"] = {field: control[field].detach().to("cpu").clone()
                                  for field in _INSTALLED_ARRAYS}
            frame["installed"]["root_state_w"][..., :3] -= self._env_origins.cpu()[:, None]
        if self.object_joint_metadata is not None:
            frame["object_joint_pos"] = self.cmd.native_object_joint_pos.detach().to("cpu").clone()
            frame["object_joint_vel"] = self.cmd.native_object_joint_vel.detach().to("cpu").clone()
        if self._physics_time_reader is not None:
            frame["physics_time_s"] = self._physics_time_reader().detach().cpu().clone()
        if self.execution_clock_metadata is not None:
            frame["execution_steps"] = self._execution_clock_reader()["steps"].detach().cpu().clone()
        if self.contact_metadata is not None:
            facts = self._contact_reader()
            frame["contact"] = {
                key: torch.stack([facts["pairs"][n][key] for n in self._contact_names], -1).detach().cpu().clone()
                for key in _CONTACT_ARRAYS if key != "measurement_support"
            }
            frame["contact"]["measurement_support"] = facts["measurement_support"][:, None].expand(
                -1, len(self._contact_names)).detach().cpu().clone()
            frame["contact_sample_time_s"] = facts["sample_time_s"].detach().cpu().clone()
        if self.contact_interval_metadata is not None:
            interval = self._contact_interval_reader()
            frame["contact_interval"] = {
                key: torch.stack([interval["pairs"][name][key] for name in self._contact_names], -1).detach().cpu().clone()
                for key in _CONTACT_ARRAYS if key != "measurement_support"
            }
            for key in ("measurement_support", "sample_recorded", "clock_support",
                        "sample_begin_time_s", "sample_end_time_s", "interval_complete"):
                frame["contact_interval"][key] = interval[key].detach().cpu().clone()
            if "execution_clock" in self.contact_interval_metadata:
                for key in ("execution_sample_begin_step", "execution_sample_end_step",
                            "execution_clock_support", "execution_interval_complete"):
                    frame["contact_interval"][key] = interval[key].detach().cpu().clone()
        self.native_frames.append(frame)

    def collect_state(self, active_mask=None):
        state = super().collect_state(active_mask)
        select = slice(None) if active_mask is None else active_mask.cpu()
        state["native_frames"] = [{**frame, "hands": frame["hands"][select],
                                   "hand_twist_w": frame["hand_twist_w"][select],
                                       "parts": frame["parts"][select],
                                   "motion_steps": frame["motion_steps"][select],
                                   "motion_ids": frame["motion_ids"][select],
                                   "command_motion_steps": frame["command_motion_steps"][select],
                                   "command_motion_ids": frame["command_motion_ids"][select]}
                                  for frame in self.native_frames]
        for selected, frame in zip(state["native_frames"], self.native_frames, strict=True):
            if "object_assist" in frame:
                selected["object_assist"] = {field: value[select] for field, value in frame["object_assist"].items()}
            if "installed" in frame:
                selected["installed"] = {field: value[select] for field, value in frame["installed"].items()}
            if "object_joint_pos" in frame:
                selected["object_joint_pos"] = frame["object_joint_pos"][select]
                selected["object_joint_vel"] = frame["object_joint_vel"][select]
            if "contact" in frame:
                selected["contact"] = {key: value[select] for key, value in frame["contact"].items()}
                selected["contact_sample_time_s"] = frame["contact_sample_time_s"][select]
            if "physics_time_s" in frame:
                selected["physics_time_s"] = frame["physics_time_s"][select]
            if "execution_steps" in frame:
                selected["execution_steps"] = frame["execution_steps"][select]
            if "contact_interval" in frame:
                selected["contact_interval"] = {key: value[select] for key, value in frame["contact_interval"].items()}
        return state

    def rollout_arrays(self, states, env_index):
        result = super().rollout_arrays(states, env_index)
        if result is None:
            return None
        offset = 0
        for state in states:
            size = len(state["valid_steps_per_env"])
            if env_index < offset + size:
                row = env_index - offset
                count = len(result["joint_pos"])
                frames = state["native_frames"][:count]
                name = self.motion_names[int(result["motion_id"])]
                binding = {**self.binding, "clips": {name: self.binding["clips"][name]}}
                result.update(
                    native_hand_pose=np.stack([frame["hands"][row].numpy() for frame in frames]),
                    native_hand_twist_w=np.stack([frame["hand_twist_w"][row].numpy() for frame in frames]),
                    native_hand_twist_authority=np.asarray("observed WORLD link-origin linear then angular velocity"),
                    native_object_part_pose=np.stack([frame["parts"][row].numpy() for frame in frames]),
                    recorded_motion_steps=np.array([int(frame["motion_steps"][row]) for frame in frames], np.int64),
                    recorded_motion_ids=np.array([int(frame["motion_ids"][row]) for frame in frames], np.int64),
                    command_motion_steps=np.array([int(frame["command_motion_steps"][row]) for frame in frames], np.int64),
                    command_motion_ids=np.array([int(frame["command_motion_ids"][row]) for frame in frames], np.int64),
                    reference_phase_authority=np.asarray("pre_advance_command_metrics"),
                    object_pinning=np.array([frame["pinning"] for frame in frames], bool),
                    assist_curriculum=np.array([frame["curriculum"] for frame in frames], bool),
                    elapsed_time_ns=np.rint(np.arange(count) * self.step_dt * 1e9).astype(np.int64),
                    raw_action_valid=np.arange(count) > 0,
                    joint_names=np.asarray(self.joint_names),
                    recording=np.asarray(_NATIVE_RECORDING),
                    native_binding_json=np.asarray(json.dumps(binding)),
                    prepared_name=np.asarray(name),
                )
                if self.object_assist_metadata is not None:
                    result["object_assist_metadata_json"] = np.asarray(json.dumps(self.object_assist_metadata))
                    for field in frames[0]["object_assist"]:
                        result["object_assist_" + field] = np.stack([
                            frame["object_assist"][field][row].numpy() for frame in frames
                        ])
                if self.installed_metadata is not None:
                    result["installed_control_metadata_json"] = np.asarray(json.dumps(self.installed_metadata))
                    for field in _INSTALLED_ARRAYS:
                        key = "installed_control_valid" if field == "installed_valid" else "installed_" + field
                        result[key] = np.stack([
                            frame["installed"][field][row].numpy() for frame in frames])
                if self.object_joint_metadata is not None:
                    result["object_joint_metadata_json"] = np.asarray(json.dumps(self.object_joint_metadata))
                    for field in ("object_joint_pos", "object_joint_vel"):
                        result[field] = np.stack([frame[field][row].numpy() for frame in frames])
                    for field in ("object_joint_position_support", "object_joint_velocity_support"):
                        result[field] = np.ones_like(result["object_joint_pos"], dtype=bool)
                if self.backend_metadata is not None:
                    result["native_backend_metadata_json"] = np.asarray(json.dumps(self.backend_metadata))
                if self.contact_association_metadata is not None:
                    result["native_contact_association_metadata_json"] = np.asarray(
                        json.dumps(self.contact_association_metadata)
                    )
                if self._physics_time_reader is not None:
                    result["native_physics_time_s"] = np.array([float(frame["physics_time_s"][row]) for frame in frames])
                    result["native_physics_reset_time_s"] = np.asarray(float(frames[0]["physics_time_s"][row]))
                    result["native_physics_clock_metadata_json"] = np.asarray(json.dumps(dict(
                        authority="actual per-world solver floating-point time at achieved endpoint",
                        origin="first recorded reset sample; absolute solver time retained",
                        source_clock="independent original int64 source/reference clocks",
                        elapsed_time_ns="nominal control elapsed time from recorded reset",
                    )))
                if self.contact_metadata is not None:
                    result["hand_part_contact_metadata_json"] = np.asarray(json.dumps(self.contact_metadata))
                    for field in _CONTACT_ARRAYS:
                        result["hand_part_" + field] = np.stack([frame["contact"][field][row].numpy() for frame in frames])
                    result["hand_part_sample_time_s"] = np.array([float(frame["contact_sample_time_s"][row]) for frame in frames])
                if self.execution_clock_metadata is not None:
                    result["native_execution_steps"] = np.array(
                        [int(frame["execution_steps"][row]) for frame in frames], dtype=np.int64)
                    result["native_execution_clock_metadata_json"] = np.asarray(json.dumps(self.execution_clock_metadata))
                if self.contact_interval_metadata is not None:
                    result["hand_part_interval_metadata_json"] = np.asarray(json.dumps(self.contact_interval_metadata))
                    for field in _CONTACT_INTERVAL_ARRAYS:
                        result["hand_part_interval_" + field] = np.stack([
                            frame["contact_interval"][field][row].numpy() for frame in frames])
                    if "execution_clock" in self.contact_interval_metadata:
                        for field in ("execution_sample_begin_step", "execution_sample_end_step",
                                      "execution_clock_support", "execution_interval_complete"):
                            result["hand_part_interval_" + field] = np.stack([
                                frame["contact_interval"][field][row].numpy() for frame in frames])
                return result
            offset += size
        raise IndexError(env_index)
