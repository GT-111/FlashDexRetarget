"""Shared NPZ serialization for first-success and explicit frozen-attempt archives.

The success archive keeps the first passing clip per motion. Frozen evaluation
keeps every actual attempt and explicitly records unavailable or absent tapes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic rewrite, so a killed job cannot leave a truncated manifest."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _write_rollout(path: Path, arrays: dict[str, Any]) -> dict[str, Any]:
    """One NPZ schema/serializer for first-success and explicit frozen attempts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    record = {
        "frames": int(arrays["joint_pos"].shape[0]),
        "recording": str(arrays.get("recording", "author-post-step-recording")),
        "fields": sorted(arrays),
    }
    prefix = "applied_object_wrench_interval_"
    if prefix + "measurement_support" in arrays:
        support = arrays[prefix + "measurement_support"]
        force = arrays[prefix + "peak_force_n"]
        torque = arrays[prefix + "peak_torque_nm"]
        record["applied_wrench_interval"] = {
            "complete": bool(len(support) > 1 and not support[0] and support[1:].all()),
            "measured_intervals": int(support.sum()),
            "peak_force_n": float(force[support].max(initial=0.0)),
            "peak_torque_nm": float(torque[support].max(initial=0.0)),
        }
    return record


def _measure_hoi_perturbation(arrays, plan, mode):
    if plan["status"] != "planned":
        return dict(status=plan["status"], **({"missing_phases": plan["missing_phases"]} if "missing_phases" in plan else {}))
    prefix = "applied_object_wrench_interval_"
    support = arrays.get(prefix + "measurement_support")
    if support is None or not support[1:].size or not support[1:].all():
        return dict(status="no_complete_solver_wrench")
    metadata = json.loads(str(arrays[prefix + "metadata_json"]))
    parts = list(zip(metadata["entity_names"], metadata["part_body_names"], strict=True))
    wrench = arrays[prefix + "substep_wrench_w"][1:].reshape(-1, len(parts), 6)
    expected = np.zeros(wrench.shape[:-1], dtype=bool)
    if mode == "perturbed":
        for burst in plan["bursts"]:
            column = parts.index((burst["entity_name"], burst["root_body_name"]))
            begin, end = burst["start_substep"], burst["stop_substep_exclusive"]
            if not 0 <= begin < end <= len(expected):
                return dict(status="solver_wrench_schedule_mismatch")
            expected[begin:end, column] = True
    actual = np.any(wrench != 0., axis=-1)
    return dict(status="verified" if np.array_equal(actual, expected) else "solver_wrench_schedule_mismatch",
                planned_substep_parts=int(expected.sum()), measured_substep_parts=int(actual.sum()))


def export_current_attempts(
    out_dir: str | Path,
    rec_cb: Any,
    rec_states: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    *,
    evaluation_id: str,
    env_step: int,
    frozen_authority: dict[str, Any],
    motion_names: list[str | None],
    num_per_motion: int,
    motion_file: str | None,
    evaluation_report: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Persist every factual attempt; expected but unexecuted slots have no arrays.

    Replica indices are the observed per-motion occurrence within this sweep,
    independently of object-slot ordering. They never select only successful rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    actual: dict[int, list[dict[str, Any]]] = {motion_id: [] for motion_id in range(len(motion_names))}
    for row in attempt_rows:
        actual[row["motion_id"]].append(row)
    motions = []
    for motion_id, motion_name in enumerate(motion_names):
        attempts = []
        # Keep extra actual attempts too; the configured denominator cannot
        # discard a real row or turn an absent planned row into an execution.
        for replica_index in range(max(num_per_motion, len(actual[motion_id]))):
            row = actual[motion_id][replica_index] if replica_index < len(actual[motion_id]) else None
            attempt_id = f"m{motion_id:05d}-a{replica_index:05d}"
            identity = {"evaluation_id": evaluation_id, "attempt_id": attempt_id,
                        "motion_id": motion_id, "replica_index": replica_index,
                        "evaluation_row": None if row is None else row["evaluation_row"],
                        "env_step": int(env_step), "frozen_authority": frozen_authority}
            entry = {"attempt_id": attempt_id, "replica_index": replica_index,
                     "planned_replica": replica_index < num_per_motion,
                     "evaluation_row": identity["evaluation_row"],
                     "batch_index": None if row is None else row["batch_index"],
                     "env_index": None if row is None else row["env_index"],
                     "execution": "not_evaluated" if row is None else "executed"}
            recording = {"status": "unavailable", "rollout": None, "frames": None, "fields": [], "reason": None}
            arrays = None
            if row is None:
                recording["reason"] = "attempt_not_evaluated"
            elif rec_cb is None:
                recording["reason"] = "no_rollout_recorder_selected"
            elif not getattr(rec_cb, "enabled", True):
                recording["reason"] = "recorder_disabled"
            else:
                arrays = rec_cb.rollout_arrays(rec_states, row["evaluation_row"])
                if arrays is None:
                    recording["reason"] = "recorder_returned_no_factual_tape"
            if arrays is not None:
                if int(arrays["motion_id"]) != motion_id:
                    raise ValueError("frozen recording row differs from its actual evaluation motion")
                relative = f"rollouts/{attempt_id}.npz"
                # All existing native facts and dtypes pass unchanged. This
                # identity is metadata, never an inferred phase/clock or target.
                arrays = {**arrays, "evaluation_attempt_json": np.asarray(json.dumps(identity))}
                plan = row.get("hoi_evaluation_perturbation_plan")
                if plan is not None:
                    if plan["replica_index"] != replica_index:
                        raise ValueError("HOI perturbation plan differs from its original-motion replica")
                    arrays["hoi_evaluation_perturbation_plan_json"] = np.asarray(json.dumps(plan))
                recording.update(_write_rollout(out_dir / relative, arrays),
                                 status="recorded", rollout=relative, reason=None)
                if plan is not None:
                    recording["hoi_evaluation_perturbation"] = _measure_hoi_perturbation(
                        arrays, plan, frozen_authority["hoi_qualification"]["mode"])
            entry["recording"] = recording
            attempts.append(entry)
        motions.append({"motion_id": motion_id, "motion_name": motion_name, "attempts": attempts})
    manifest = out_dir / "manifest.json"
    _write_json(manifest, {
        "schema": "frozen_current_policy_attempts_v1", "evaluation_id": evaluation_id,
        "selection": "all_current_attempts; independent of first_success_ever archive",
        "env_step": int(env_step), "frozen_authority": frozen_authority,
        "authority_scope": "explicit caller-declared acting method/source, including analytic methods; exporter does not infer or verify loader history",
        "motion_file": motion_file, "selected_motion_count": len(motions),
        "configured_replicas_per_motion": int(num_per_motion),
        "evaluation_report": os.path.relpath(evaluation_report, out_dir),
        "attempt_index_authority": "per-motion observed sweep occurrence; actual batch/env/global row retained",
        "raw_fact_authority": "selected recorder arrays unchanged; source phase/clocks/native/contact authority remain in those arrays",
        "unavailable": "unexecuted or unsupported tape is explicit; no arrays are synthesized",
        "motions": motions,
    })
    return manifest, motions


def export_success(
    out_dir: str | Path,
    rec_cb: Any,
    rec_states: list[dict[str, Any]],
    ok: torch.Tensor,
    motion_ids: torch.Tensor,
    env_step: int,
    criterion: str,
    motion_names: list[str] | None = None,
    motion_file: str | None = None,
) -> int:
    """Archive the first passing env row of each motion not archived yet; returns the count.
    ``ok`` and ``motion_ids`` are per env over the concatenated eval sweep."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
            entries = {str(e["motion_id"]): e for e in prev.get("saved", [])}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[success] unreadable manifest, starting fresh: {exc}")

    written = 0
    for row in torch.nonzero(ok, as_tuple=False).flatten().tolist():
        mid = int(motion_ids[row].item())
        if str(mid) in entries:
            continue
        arrays = rec_cb.rollout_arrays(rec_states, row)
        if arrays is None:
            continue
        rel = f"rollouts/m{mid:05d}.npz"
        recording = _write_rollout(out_dir / rel, arrays)
        entries[str(mid)] = {
            "motion_id": mid,
            "name": motion_names[mid] if motion_names is not None and mid < len(motion_names) else None,
            "frames": recording["frames"],
            "rollout": rel,
            "first_saved_env_step": int(env_step),
            "recording": recording["recording"],
            "fields": recording["fields"],
        }
        written += 1
    if not written:
        return 0

    saved = sorted(entries.values(), key=lambda e: int(e["motion_id"]))
    _write_json(
        manifest_path,
        {
            "criterion": criterion,
            "motion_file": motion_file,
            "num_saved": len(saved),
            "updated_at_env_step": int(env_step),
            "coordinate_frame": "env-local (scene env_origins subtracted)",
            "selection": "first_success_ever_archive; not frozen_final_policy_evaluation",
            "action_authority": "raw action_manager.action; installed_* facts only when the actual action owner supplies them",
            "reference_phase_authority": "recorded_motion_steps in native-v2 are pre-advance metric endpoints; command_motion_steps retain the next-target cursor",
            "fields": "joint_pos, joint_vel, actions, root_pos, root_quat, obj_pos, obj_quat",
            "saved": saved,
        },
    )
    print(f"[success] archived {written} new clip(s); {len(saved)} motions in {out_dir}", flush=True)
    return written
