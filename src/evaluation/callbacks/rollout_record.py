"""Per-step eval rollout capture for the success archive (success_export.py); one flat
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
            ].cpu(),
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
_NATIVE_RECORDING = "dexforge-flashdex-achieved-state-v1"


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
                    or lineage["capture_id"] != clip["source_unit_id"]
                    or not np.array_equal(lineage["selected_source_frame_index"], clip["selected_source_frame_index"])
                    or not np.array_equal(ordinal, clip["prepared_source_ordinal"])
                    or not np.array_equal(elapsed - elapsed[0], clip["prepared_elapsed_time_ns"])
                    or not np.allclose(lineage["scene_from_source_world"], self.binding["scene_from_source_world"],
                                       atol=1e-9, rtol=0)):
                raise ValueError("FlashDex native binding differs from the actual imported frame/scene lineage")
        self.native_frames = []
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
        part_position = torch.stack([entity.data.body_link_pos_w[:, body] for entity, body in self.object_parts], dim=1)
        part_position -= self._env_origins[:, None]
        part_quaternion = torch.stack([entity.data.body_link_quat_w[:, body] for entity, body in self.object_parts], dim=1)
        parts = torch.cat((part_position, part_quaternion), dim=-1).detach().to("cpu", torch.float32)
        pinning = bool(self.cmd.cfg.object.pin_objects)
        curriculum = self.cmd._xfrc_curr_ctrl is not None
        self.native_frames.append({
            "hands": hands, "parts": parts,
            "motion_steps": self.cmd.motion_steps.detach().to("cpu", torch.int64).clone(),
            "motion_ids": self.cmd.motion_ids.detach().to("cpu", torch.int64).clone(),
            "pinning": pinning, "curriculum": curriculum,
        })

    def collect_state(self, active_mask=None):
        state = super().collect_state(active_mask)
        select = slice(None) if active_mask is None else active_mask.cpu()
        state["native_frames"] = [{**frame, "hands": frame["hands"][select],
                                       "parts": frame["parts"][select],
                                   "motion_steps": frame["motion_steps"][select],
                                   "motion_ids": frame["motion_ids"][select]}
                                  for frame in self.native_frames]
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
                    native_object_part_pose=np.stack([frame["parts"][row].numpy() for frame in frames]),
                    recorded_motion_steps=np.array([int(frame["motion_steps"][row]) for frame in frames], np.int64),
                    recorded_motion_ids=np.array([int(frame["motion_ids"][row]) for frame in frames], np.int64),
                    object_pinning=np.array([frame["pinning"] for frame in frames], bool),
                    assist_curriculum=np.array([frame["curriculum"] for frame in frames], bool),
                    elapsed_time_ns=np.rint(np.arange(count) * self.step_dt * 1e9).astype(np.int64),
                    raw_action_valid=np.arange(count) > 0,
                    joint_names=np.asarray(self.joint_names),
                    recording=np.asarray(_NATIVE_RECORDING),
                    native_binding_json=np.asarray(json.dumps(binding)),
                    prepared_name=np.asarray(name),
                )
                return result
            offset += size
        raise IndexError(env_index)
