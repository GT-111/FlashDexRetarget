from __future__ import annotations

from typing import Any, cast

import torch

from envs.commands.motion_tracking import MotionTrackingCommand


class ContactPerformance:
    """Per-finger contact ref-distance / penetration / force / found / miss-streak."""

    def __init__(self, grace_steps: int, command_name: str) -> None:
        self.grace_steps = grace_steps
        self.command_name = command_name

    def on_start(self, env: Any) -> None:
        self.cmd = cast(
            MotionTrackingCommand, env.command_manager.get_term(self.command_name)
        )
        self.device = env.device
        self.num_envs = env.num_envs
        self.side_prefixes = tuple(s[0] for s in self.cmd._side_list)
        self.num_sides = len(self.side_prefixes)
        self.finger_names = tuple(self.cmd.finger_names)
        self.num_fingers = len(self.finger_names)
        self.max_motion_frames = int(
            self.cmd.motion_lib._motion_num_frames.max().item()
        )
        self.valid_steps_per_env = (
            self.cmd.motion_num_frames - self.cmd.motion_steps
        ).clamp(min=0)
        self.motion_start_steps = self.cmd.motion_steps.clone()
        metadata_reader = getattr(env.scene, "tracking_contact_metadata", None)
        self.measurement_metadata = metadata_reader() if metadata_reader is not None else None
        self.scored_env_mask = self.valid_steps_per_env > self.grace_steps
        if not torch.any(self.scored_env_mask):
            self.scored_env_mask = torch.ones(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        self.rollout_steps = max(1, int(self.valid_steps_per_env.max().item()))
        self.eval_end = max(self.grace_steps + 1, self.rollout_steps)

        self.pen_sensors = {
            side: env.scene[f"{side[0]}_fingertip_penetration"]
            for side in self.cmd._side_list
        }
        self.force_sensors = {
            side: env.scene[f"{side[0]}_fingertip_contact"]
            for side in self.cmd._side_list
        }

        def _z() -> torch.Tensor:
            return torch.zeros(
                self.num_envs, self.num_sides, self.num_fingers, device=self.device
            )

        self.sum_ref_dist = _z()
        self.sum_pen = _z()
        self.sum_force = _z()
        self.count_ref = _z()
        self.count_contact = _z()
        self.miss_counter = _z()
        self.miss_max = _z()
        self.count_per_env = torch.zeros(self.num_envs, device=self.device)
        self.t = 0

    def on_step(self, env: Any) -> None:
        in_window = (
            (self.t < self.valid_steps_per_env)
            & (self.t >= self.grace_steps)
            & (self.t < self.eval_end)
        )
        self.t += 1
        if not torch.any(in_window):
            return
        self.count_per_env += in_window.to(self.count_per_env.dtype)

        cmd = self.cmd
        valid_mask = in_window[:, None].to(torch.float32)
        for si, side in enumerate(cmd._side_list):
            pen_sens = self.pen_sensors[side]
            force_sens = self.force_sensors[side]
            flag = cmd.ref_contact_flags[:, si] * valid_mask
            found = (pen_sens.data.found > 0).to(flag.dtype)
            gate = flag * found
            ref_dist = torch.norm(
                cmd.ref_contact_trans_w[:, si] - cmd.robot_tip_trans_w[:, si], dim=-1
            )
            pen = torch.clamp(-pen_sens.data.dist, min=0.0)
            force = torch.norm(force_sens.data.force, dim=-1)
            self.sum_ref_dist[:, si] += ref_dist * flag
            self.sum_pen[:, si] += pen * gate
            self.sum_force[:, si] += force * gate
            self.count_ref[:, si] += flag
            self.count_contact[:, si] += gate
            miss = flag * (1.0 - found)
            self.miss_counter[:, si] = (self.miss_counter[:, si] + 1.0) * miss
            self.miss_max[:, si] = torch.maximum(
                self.miss_max[:, si], self.miss_counter[:, si]
            )

    def collect_state(
        self, active_mask: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Per-env contact accumulators for the active envs, for cross-batch reduction.
        Raw ``valid_steps_per_env`` lets scored basis + fallback span the full concat."""
        sel = slice(None) if active_mask is None else active_mask
        return {
            "motion_ids": self.cmd.motion_ids[sel].clone(),
            "motion_start_steps": self.motion_start_steps[sel].clone(),
            "count_per_env": self.count_per_env[sel],
            "sum_ref_dist": self.sum_ref_dist[sel],
            "sum_pen": self.sum_pen[sel],
            "sum_force": self.sum_force[sel],
            "count_ref": self.count_ref[sel],
            "count_contact": self.count_contact[sel],
            "miss_max": self.miss_max[sel],
            "valid_steps_per_env": self.valid_steps_per_env[sel],
        }

    def current_policy_outcomes(self, states: list[dict[str, Any]]) -> dict:
        cat = {key: torch.cat([state[key] for state in states]) for key in states[0]}
        return {
            "motion_ids": cat["motion_ids"],
            "metadata": {
                "meaning": "existing reference-contact gated sensor measurements, not a task-success criterion",
                "side_prefixes": self.side_prefixes, "finger_names": self.finger_names,
                "grace_steps": self.grace_steps, "sensor_measurement_authority": self.measurement_metadata,
                "pair_force_authority": "Only the backend metadata can establish pair-force support; SDF found/proximity and net-link force cannot",
                "denominators": {"ref_distance": "count_ref", "found_fraction": "count_ref",
                                 "penetration_and_force": "count_contact"},
            },
            "columns": {
                "motion_start_step": cat["motion_start_steps"],
                "remaining_source_steps": cat["valid_steps_per_env"],
                "observed_steps": cat["count_per_env"], "scored": cat["count_per_env"] > 0,
                "raw_sum_ref_distance_m": cat["sum_ref_dist"], "raw_sum_penetration_m": cat["sum_pen"],
                "raw_sum_sensor_force_N": cat["sum_force"], "reference_contact_count": cat["count_ref"],
                "associated_contact_count": cat["count_contact"], "max_miss_streak_steps": cat["miss_max"],
            },
        }

    def reduce_state(self, states: list[dict[str, Any]]) -> dict[str, float]:
        """Per-finger contact metrics over the concatenation of per-batch states.
        Scored basis + fallback are recomputed over the concat (matches a single pass)."""
        sum_ref_dist = torch.cat([s["sum_ref_dist"] for s in states], dim=0)
        sum_pen = torch.cat([s["sum_pen"] for s in states], dim=0)
        sum_force = torch.cat([s["sum_force"] for s in states], dim=0)
        count_ref = torch.cat([s["count_ref"] for s in states], dim=0)
        count_contact = torch.cat([s["count_contact"] for s in states], dim=0)
        miss_max = torch.cat([s["miss_max"] for s in states], dim=0)
        valid_steps_per_env = torch.cat(
            [s["valid_steps_per_env"] for s in states], dim=0
        )
        num_envs = int(valid_steps_per_env.shape[0])

        scored = valid_steps_per_env > self.grace_steps
        if not torch.any(scored):
            scored = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        cnt_ref_safe = count_ref.clamp(min=1.0)
        cnt_contact_safe = count_contact.clamp(min=1.0)
        avg_ref_dist = (sum_ref_dist / cnt_ref_safe)[scored].mean(dim=0)
        avg_found = (count_contact / cnt_ref_safe)[scored].mean(dim=0)
        avg_pen = (sum_pen / cnt_contact_safe)[scored].mean(dim=0)
        avg_force = (sum_force / cnt_contact_safe)[scored].mean(dim=0)
        avg_miss_max = miss_max[scored].mean(dim=0)

        out: dict[str, float] = {}
        for si, p in enumerate(self.side_prefixes):
            for fi, finger in enumerate(self.finger_names):
                out[f"contact_ref_dist_{p}_{finger}"] = float(
                    avg_ref_dist[si, fi].item()
                )
                out[f"contact_found_{p}_{finger}"] = float(avg_found[si, fi].item())
                out[f"contact_pen_{p}_{finger}"] = float(avg_pen[si, fi].item())
                out[f"contact_force_{p}_{finger}"] = float(avg_force[si, fi].item())
                out[f"contact_miss_max_{p}_{finger}"] = float(
                    avg_miss_max[si, fi].item()
                )
        out["contact_num_scored_envs"] = int(scored.sum().item())
        return out

    def get_metrics(self) -> dict[str, float]:
        return self.reduce_state([self.collect_state()])

    def on_end(self) -> None:
        m = self.get_metrics()
        print("-" * 72)
        print("Per-finger contact performance:")
        for si, p in enumerate(self.side_prefixes):
            for fi, finger in enumerate(self.finger_names):
                print(
                    f"  {p}_{finger:<6s}  ref_dist: {m[f'contact_ref_dist_{p}_{finger}']:.6f} m  "
                    f"pen: {m[f'contact_pen_{p}_{finger}']:.6f} m  "
                    f"force: {m[f'contact_force_{p}_{finger}']:.4f} N  "
                    f"found: {m[f'contact_found_{p}_{finger}']:.4f}"
                )


class NativeContactPerformance:
    """Named pair-force sample durations, independently of source task roles.

    Known absence counts as a supported negative sample. Unavailable pair
    forces, missing clocks or reset rows never become measured zero contact.
    """

    def __init__(self, command_name: str, normal_force_threshold_n: float,
                 separation_bound_m: float, grace_steps: int = 0):
        self.command_name, self.grace_steps = command_name, int(grace_steps)
        self.normal_force_threshold_n = float(normal_force_threshold_n)
        self.separation_bound_m = float(separation_bound_m)

    def on_start(self, env):
        self.cmd = env.command_manager.get_term(self.command_name)
        layout = self.cmd.native_layout
        self.addresses = {
            f"{hand.side}/{obj.entity_name}/{part.name}": dict(hand_side=hand.side, hand_body_name=hand.root_body_name,
                part_entity_name=obj.entity_name, part_body_name=part.body_name)
            for hand in layout.hands for obj, part in layout.part_bindings
        }
        self.pair_names = tuple(self.addresses)
        self.num_substeps, self.control_dt = int(env.cfg.decimation), float(env.step_dt)
        self.motion_start_steps = self.cmd.motion_steps.clone()
        self.valid_steps = (self.cmd.motion_num_frames - self.cmd.motion_steps).clamp_min(0)
        self.rollout_steps, self.t = int(self.valid_steps.max()), 0
        self.reader = env.scene.hand_part_contact_interval_facts
        facts = self.reader()
        self.available = facts is not None
        if self.available and (facts["addresses"] != self.addresses or facts["num_substeps"] != self.num_substeps):
            raise ValueError("native contact interval differs from actual scene bindings or configured decimation")
        self.metadata = {
            "meaning": "discrete native solver pair-force measurements; no source-required contact or task-success claim",
            "pairs": self.addresses, "pair_order": self.pair_names,
            "num_substeps": self.num_substeps, "configured_physics_dt_s": float(env.physics_dt),
            "configured_control_dt_s": self.control_dt, "grace_steps": self.grace_steps,
            "normal_force_threshold_n": self.normal_force_threshold_n, "separation_bound_m": self.separation_bound_m,
            "pair_measurement_available": self.available,
            "normal_force_authority": None if facts is None else facts["normal_force_authority"],
            "distance_authority": None if facts is None else facts["distance_authority"],
            "sampling": None if facts is None else facts["sampling"],
            "duration_authority": "actual recorded solver end minus begin for supported discrete samples; not continuous force proof",
            "scored_basis": "at least one supported discrete named-pair sample; no task score",
            "complete_interval_meaning": "all configured substeps recorded with valid contiguous solver clocks; force support remains independent",
            "source_required_contact_phase": "UNKNOWN; no source engagement inferred from force, proximity or hinge sign",
            "backend": env.scene.backend_identity,
        }
        shape = (env.num_envs, len(self.pair_names))
        self.counts = {key: torch.zeros(shape, dtype=torch.int64, device=env.device)
                       for key in ("recorded_sample_count", "clock_supported_sample_count",
                                   "supported_sample_count", "qualified_sample_count")}
        self.sums = {key: torch.zeros(shape, dtype=torch.float64, device=env.device)
                     for key in ("recorded_duration_s", "supported_duration_s", "qualified_duration_s",
                                 "sample_force_duration_sum_n_s")}
        self.maximum_force = torch.zeros(shape, device=env.device)
        self.observed_controls = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.complete_intervals = torch.zeros_like(self.observed_controls)

    def on_step(self, env):
        del env
        active = (self.t < self.valid_steps) & (self.t >= self.grace_steps)
        self.t += 1
        self.observed_controls += active
        if not self.available:
            return
        facts = self.reader()
        self.complete_intervals += facts["interval_complete"] & active
        force = torch.stack([facts["pairs"][name]["normal_force_n"] for name in self.pair_names], -1)
        present = torch.stack([facts["pairs"][name]["force_bearing_contact_present"] for name in self.pair_names], -1)
        distance = torch.stack([facts["pairs"][name]["force_bearing_distance_max_m"] for name in self.pair_names], -1)
        recorded = facts["sample_recorded"] & active[:, None]
        clock = recorded & facts["clock_support"]
        known = (clock & facts["measurement_support"])[..., None] & torch.isfinite(force) & (force >= 0)
        known &= ~present | torch.isfinite(distance)
        qualified = known & present & (force >= self.normal_force_threshold_n) & (distance <= self.separation_bound_m)
        duration = torch.where(clock, facts["sample_end_time_s"] - facts["sample_begin_time_s"], 0.0)
        width = len(self.pair_names)
        self.counts["recorded_sample_count"] += recorded.sum(1)[:, None].expand(-1, width)
        self.counts["clock_supported_sample_count"] += clock.sum(1)[:, None].expand(-1, width)
        self.counts["supported_sample_count"] += known.sum(1)
        self.counts["qualified_sample_count"] += qualified.sum(1)
        self.sums["recorded_duration_s"] += duration.sum(1)[:, None].expand(-1, width)
        self.sums["supported_duration_s"] += torch.where(known, duration[..., None], 0.0).sum(1)
        self.sums["qualified_duration_s"] += torch.where(qualified, duration[..., None], 0.0).sum(1)
        self.sums["sample_force_duration_sum_n_s"] += (torch.where(known, force, 0.0) * duration[..., None]).sum(1)
        self.maximum_force = torch.maximum(self.maximum_force, torch.where(known, force, 0.0).amax(1))

    def collect_state(self, active_mask=None):
        rows = slice(None) if active_mask is None else active_mask
        return {
            "motion_ids": self.cmd.motion_ids[rows].clone(),
            "motion_start_steps": self.motion_start_steps[rows].clone(),
            "valid_steps": self.valid_steps[rows].clone(),
            "observed_controls": self.observed_controls[rows].clone(),
            "complete_intervals": self.complete_intervals[rows].clone(),
            "maximum_supported_normal_force_n": self.maximum_force[rows].clone(),
            **{key: value[rows].clone() for key, value in self.counts.items()},
            **{key: value[rows].clone() for key, value in self.sums.items()},
        }

    def current_policy_outcomes(self, states):
        cat = {key: torch.cat([state[key] for state in states]) for key in states[0]}
        observed = cat["observed_controls"]
        expected = observed[:, None] * self.num_substeps
        columns = {key: value for key, value in cat.items() if key != "motion_ids"}
        columns.update(
            scored=cat["supported_sample_count"].sum(-1) > 0,
            expected_sample_count=expected.expand_as(cat["supported_sample_count"]),
            unsupported_or_unrecorded_sample_count=expected - cat["supported_sample_count"],
            nominal_control_duration_s=observed * self.control_dt,
        )
        return {"motion_ids": cat["motion_ids"], "metadata": self.metadata, "columns": columns}

    def reduce_state(self, states):
        output = {}
        for index, name in enumerate(self.pair_names):
            for key in (*self.counts, *self.sums):
                value = torch.cat([state[key] for state in states])[:, index].sum()
                output[f"native/contact_samples/{name}/{key}"] = float(value)
            duration = output[f"native/contact_samples/{name}/supported_duration_s"]
            force_duration_sum = output[f"native/contact_samples/{name}/sample_force_duration_sum_n_s"]
            output[f"native/contact_samples/{name}/mean_supported_normal_force_n"] = (
                force_duration_sum / duration if duration else float("nan")
            )
        return output
