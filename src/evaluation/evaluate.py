from __future__ import annotations

import json
import math
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import DictConfig

from agents import Policy
from evaluation.callbacks.success_export import export_current_attempts, export_success


def validate_recording_authority(authority):
    """Require a real learned checkpoint or explicit parameterless method identity."""
    if not authority or not authority.get("source_revision"):
        raise ValueError("frozen recording requires explicit acting method and source_revision authority")
    if authority.get("checkpoint"):
        return
    policy = authority.get("policy", {})
    if (policy.get("kind") != "analytic" or policy.get("has_learned_parameters") is not False
            or not policy.get("method_id") or policy.get("checkpoint") is not None):
        raise ValueError("analytic recording requires a named parameterless policy without a checkpoint")


def evaluate(
    agent: Policy,
    eval_env: RslRlVecEnvWrapper,
    eval_cfg: DictConfig,
    device: str,
    ever_state: dict | None = None,
    success_dir: Path | None = None,
    env_step: int = 0,
    *,
    outcome_dir: Path,
    recording_dir: Path | None = None,
    frozen_authority: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Roll the deterministic policy through every eval batch, then turn the callback
    states into metrics."""
    raw_eval = eval_env.unwrapped
    if recording_dir is not None:
        validate_recording_authority(frozen_authority)
    # Keep capture only for a selected success archive or explicit frozen export.
    save_success = bool(eval_cfg.get("save_success_data", True))
    callbacks = [
        instantiate(cb) for cb in eval_cfg.callbacks
        if save_success or recording_dir is not None or "rollout_record" not in str(cb.get("_target_", ""))
    ]

    attempt_rows = [] if recording_dir is not None else None
    states = {id(cb): [] for cb in callbacks if hasattr(cb, "reduce_state")}
    stage = "rollout"
    try:
        _rollout(agent, eval_env, callbacks, device, attempt_rows=attempt_rows, states=states)
        stage = "metric_reduction"
        metrics = reduce_metrics(callbacks, states)
        metrics.update(tracking_success_metrics(
            raw_eval, eval_cfg, callbacks, states, ever_state,
            success_dir if save_success else None, env_step,
        ))
        stage = "outcome_export"
        write_current_policy_outcomes(agent, raw_eval, eval_cfg, callbacks, states, metrics,
                                      outcome_dir=outcome_dir, env_step=env_step,
                                      recording_dir=recording_dir, frozen_authority=frozen_authority,
                                      attempt_rows=attempt_rows)
    except BaseException as exc:
        if recording_dir is not None:
            # A failed forward or scorer does not erase already recorded facts.
            # Do not reduce partial metric callbacks into successful outcomes.
            recorders = [cb for cb in callbacks if hasattr(cb, "rollout_arrays")]
            try:
                write_current_policy_outcomes(
                    agent, raw_eval, eval_cfg, recorders, states, {}, outcome_dir=outcome_dir,
                    env_step=env_step, recording_dir=recording_dir, frozen_authority=frozen_authority,
                    attempt_rows=attempt_rows,
                    evaluation_error={"stage": stage, "exception": f"{type(exc).__name__}: {exc}",
                                      "scoring": "UNKNOWN; factual prefixes do not establish task success"},
                )
            except BaseException:
                traceback.print_exc()  # Keep the original failure as the raised exception.
        raise
    return {f"eval/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _json_value(value):
    if torch.is_tensor(value):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _outcome_row(columns, index):
    return {key: (_outcome_row(value, index) if isinstance(value, dict)
                  else value[index] if isinstance(value, list) else value)
            for key, value in columns.items()}


def write_current_policy_outcomes(
    agent, raw_eval, eval_cfg, callbacks, states, metrics, *, outcome_dir: Path, env_step: int,
    recording_dir: Path | None = None, frozen_authority: dict[str, Any] | None = None,
    attempt_rows: list[dict[str, int]] | None = None,
    evaluation_error: dict[str, Any] | None = None,
) -> Path:
    """Persist every current attempt separately from first/ever-success selection."""
    library = raw_eval.command_manager.get_term(eval_cfg.command_name).motion_lib
    names = _motion_names(library)
    motions = [{"motion_id": motion_id, "motion_name": names[motion_id] if names else None,
                "callbacks": {}} for motion_id in range(int(library.num_trajectories))]
    callback_metadata = {}
    for callback_index, callback in enumerate(callbacks):
        if id(callback) not in states or not hasattr(callback, "current_policy_outcomes"):
            continue
        key = f"{callback_index}:{type(callback).__module__}.{type(callback).__qualname__}"
        outcome = callback.current_policy_outcomes(states[id(callback)])
        callback_metadata[key] = _json_value(outcome["metadata"])
        columns = _json_value(outcome["columns"])
        for row_index, motion_id in enumerate(outcome["motion_ids"].detach().cpu().tolist()):
            row = _outcome_row(columns, row_index)
            row["evaluation_row"] = row_index
            for criterion in ("full", "object"):
                failed = row.get(f"{criterion}_failed")
                if failed is not None:
                    finite = row.get(f"{criterion}_finite_metrics", row.get("finite_metrics", True))
                    if not row["scored"]:
                        row[f"{criterion}_status"] = ["unscored"] * len(failed)
                    elif not finite:
                        row[f"{criterion}_status"] = ["nonfinite"] * len(failed)
                    else:
                        row[f"{criterion}_status"] = ["fail" if flag else "pass" for flag in failed]
            motions[motion_id]["callbacks"].setdefault(key, {"attempts": []})["attempts"].append(row)
        for motion in motions:
            result = motion["callbacks"].setdefault(key, {"attempts": []})
            rows = result["attempts"]
            result["attempt_count"] = len(rows)
            result["scored_attempt_count"] = sum(bool(row["scored"]) for row in rows)
            result["unscored_attempt_count"] = len(rows) - result["scored_attempt_count"]
            result["coverage"] = "not_evaluated" if not rows else "observed" if result["scored_attempt_count"] else "unscored"
            for criterion in ("full", "object"):
                statuses = [row[f"{criterion}_status"] for row in rows if f"{criterion}_status" in row]
                if statuses:
                    result[f"{criterion}_threshold_counts"] = [
                        {status: sum(row[index] == status for row in statuses)
                         for status in ("pass", "fail", "unscored", "nonfinite")}
                        for index in range(len(statuses[0]))
                    ]
    recorded_at = datetime.now(timezone.utc)
    report = {
        "selection": "current_policy_all_attempts; independent of first_success_ever archive",
        "policy_authority": getattr(agent, "acting_authority", {"kind": "current_in_memory_agent", "deterministic": True,
                             "agent_class": f"{type(agent).__module__}.{type(agent).__qualname__}",
                             "training_env_step": int(env_step),
                             "checkpoint_mapping": "Normal post-evaluation FULL/operation receipt; no checkpoint is inferred here"}),
        "recorded_UTC": recorded_at.isoformat(),
        "motion_file": getattr(library, "motion_file", None),
        "selected_motion_count": len(motions), "configured_replicas_per_motion": int(eval_cfg.num_per_motion),
        "selected_archive_criterion": str(eval_cfg.get("success_criterion", "obj")),
        "callback_metadata": callback_metadata, "motions": motions,
        "ordinary_aggregate_metrics": _json_value(metrics),
        "aggregate_note": "Original aggregate reduction/fallback and SR thresholds remain unchanged; attempt denominators are explicit",
        "physical_task_success": "UNQUALIFIED by generic tracking or SDF/contact association metrics; consume the declared downstream task contract",
    }
    if evaluation_error is not None:
        report["evaluation_error"] = evaluation_error
    outcome_dir = Path(outcome_dir)
    outcome_dir.mkdir(parents=True, exist_ok=True)
    destination = outcome_dir / f"current-policy-{int(env_step):012d}-{recorded_at:%Y%m%dT%H%M%S%fZ}.json"
    if recording_dir is not None:
        if attempt_rows is None or not frozen_authority:
            raise ValueError("frozen recording needs actual rollout attempt rows and explicit authority")
        recorder = next((cb for cb in callbacks if hasattr(cb, "rollout_arrays")), None)
        recorder_states = [] if recorder is None else states[id(recorder)]
        manifest, factual_motions = export_current_attempts(
            Path(recording_dir) / destination.stem, recorder, recorder_states, attempt_rows,
            evaluation_id=destination.stem, env_step=env_step,
            frozen_authority=_json_value(frozen_authority),
            motion_names=[motion["motion_name"] for motion in motions],
            num_per_motion=int(eval_cfg.num_per_motion), motion_file=getattr(library, "motion_file", None),
            evaluation_report=destination,
        )
        by_row = {attempt["evaluation_row"]: (motion["motion_id"], attempt)
                  for motion in factual_motions for attempt in motion["attempts"]
                  if attempt["execution"] == "executed"}
        for motion, factual in zip(motions, factual_motions, strict=True):
            motion["factual_attempts"] = factual["attempts"]
            for result in motion["callbacks"].values():
                for row in result["attempts"]:
                    motion_id, attempt = by_row[row["evaluation_row"]]
                    if motion_id != motion["motion_id"]:
                        raise ValueError("scalar outcome row differs from its actual frozen attempt")
                    row["attempt_id"] = attempt["attempt_id"]
                    row["replica_index"] = attempt["replica_index"]
        report["factual_recording_manifest"] = os.path.relpath(manifest, outcome_dir)
        report["frozen_authority"] = _json_value(frozen_authority)
    with tempfile.NamedTemporaryFile(mode="w", dir=outcome_dir, prefix=".current-policy-", delete=False) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, destination)
    return destination


# ─── rollout ───────────────────────────────────────────────────────────────────
def _rollout(
    agent: Policy, eval_env: RslRlVecEnvWrapper, callbacks: list, device: str,
    *, attempt_rows: list[dict[str, int]] | None = None,
    states: dict[int, list] | None = None,
) -> dict[int, list]:
    """Reset + deterministic inference over every eval batch;
    returns each metric callback's per-batch states, keyed by id(callback)."""
    raw_eval = eval_env.unwrapped
    setup_cb = next(cb for cb in callbacks if hasattr(cb, "set_batch"))
    metric_cbs = [cb for cb in callbacks if hasattr(cb, "reduce_state")]
    eval_k, num_motions = raw_eval.eval_k, raw_eval.eval_num_motions
    states = {id(cb): [] for cb in metric_cbs} if states is None else states
    try:
        with torch.no_grad():
            for b in range(raw_eval.eval_num_batches):
                setup_cb.set_batch(b, eval_k, num_motions)
                for cb in callbacks:
                    if hasattr(cb, "on_eval_setup"):
                        cb.on_eval_setup(raw_eval)
                obs, _ = eval_env.reset()
                for cb in callbacks:
                    cb.on_start(raw_eval)
                enc_keys = sorted(key for key in obs.keys() if key not in ("actor", "critic"))
                n_steps = max(1, max((int(getattr(cb, "rollout_steps", 0)) for cb in callbacks), default=1))
                active_mask = setup_cb.active_mask
                if attempt_rows is not None:
                    command = raw_eval.command_manager.get_term(setup_cb.command_name)
                    selected = (torch.arange(raw_eval.num_envs, device=raw_eval.device) if active_mask is None
                                else torch.nonzero(active_mask, as_tuple=False).flatten())
                    motion_ids = command.motion_ids[selected].detach().cpu().tolist()
                    for env_index, motion_id in zip(selected.detach().cpu().tolist(), motion_ids, strict=True):
                        attempt_rows.append({"evaluation_row": len(attempt_rows), "batch_index": b,
                                             "env_index": int(env_index), "motion_id": int(motion_id)})
                completed = False
                try:
                    for _ in range(n_steps):
                        actions = agent.act(_flat_actor_obs(obs, enc_keys), deterministic=True)
                        obs, _, _, _ = eval_env.step(torch.as_tensor(np.asarray(actions), device=device))
                        for cb in sorted(callbacks, key=lambda item: not hasattr(item, "rollout_arrays")):
                            cb.on_step(raw_eval)
                    completed = True
                finally:
                    for cb in sorted(metric_cbs, key=lambda item: not hasattr(item, "rollout_arrays")):
                        if completed or hasattr(cb, "rollout_arrays"):
                            states[id(cb)].append(cb.collect_state(active_mask))
    finally:
        if hasattr(eval_env, "exit_eval"):
            eval_env.exit_eval()
    return states


def _flat_actor_obs(obs: dict, enc_keys: list[str]) -> torch.Tensor:
    """Actor group + encoded (window) groups, appended exactly like FlatObsVecEnv._flatten_obs."""
    if not enc_keys:
        return obs["actor"]
    return torch.cat([obs["actor"]] + [obs[k] for k in enc_keys], dim=-1)


# ─── metrics ───────────────────────────────────────────────────────────────────
def reduce_metrics(callbacks: list, states: dict[int, list]) -> dict[str, float]:
    """reduce_state of every metric callback, then get_metrics of the rest."""
    metrics: dict[str, float] = {}
    for cb in callbacks:
        if id(cb) in states:
            metrics.update(cb.reduce_state(states[id(cb)]))
    for cb in callbacks:
        if id(cb) not in states and hasattr(cb, "get_metrics"):
            metrics.update(cb.get_metrics())
    return metrics


def tracking_success_metrics(
    raw_eval: Any,
    eval_cfg: DictConfig,
    callbacks: list,
    states: dict[int, list],
    ever_state: dict | None,
    success_dir: Path | None,
    env_step: int,
) -> dict[str, float]:
    """The success-clip archive and the cumulative ever-success curves."""
    # per_env_fail_obj alone would also match GtMocapPerformance, which has no per_env_fail.
    tp_cb = _metric_cb(callbacks, states, "per_env_fail")
    if tp_cb is None:
        return {}
    st = states[id(tp_cb)]
    fail = tp_cb.per_env_fail(st)  # (E, K), motion-major
    fail_obj = tp_cb.per_env_fail_obj(st)
    scored = tp_cb.per_env_scored(st)
    metrics: dict[str, float] = {}

    rec_cb = next(
        (cb for cb in callbacks if hasattr(cb, "rollout_arrays") and cb is not tp_cb), None
    )
    if success_dir is not None and rec_cb is not None:
        metrics.update(_archive_success(
            raw_eval, eval_cfg, tp_cb, rec_cb, states, fail, fail_obj, scored, success_dir, env_step
        ))
    if ever_state is not None:
        metrics.update(_ever_success(eval_cfg, tp_cb, fail, fail_obj, scored, ever_state))
    metrics.update(_spider_ever(raw_eval, tp_cb, st, scored, ever_state))
    return metrics


# ─── tracking success parts ────────────────────────────────────────────────────
def _archive_success(
    raw_eval: Any,
    eval_cfg: DictConfig,
    tp_cb: Any,
    rec_cb: Any,
    states: dict[int, list],
    fail: torch.Tensor,
    fail_obj: torch.Tensor,
    scored: torch.Tensor,
    success_dir: Path,
    env_step: int,
) -> dict[str, float]:
    """Archive the clips that pass SR@1.0 (success_export.py). A side artefact: it must never
    take a multi-day run down."""
    out: dict[str, float] = {}
    try:
        lib = raw_eval.command_manager.get_term(eval_cfg.command_name).motion_lib
        # "full" also demands the finger criterion; "obj" is object-only.
        crit = str(eval_cfg.get("success_criterion", "obj"))
        failed = (fail if crit == "full" else fail_obj)[:, tp_cb.k1_idx]
        export_success(
            success_dir,
            rec_cb,
            states[id(rec_cb)],
            ok=(~failed) & scored,
            motion_ids=torch.cat([s["motion_ids"] for s in states[id(rec_cb)]]),
            env_step=env_step,
            criterion=f"{crit} {getattr(tp_cb, 'success_criterion_label', 'SR@1.0')}",
            motion_names=_motion_names(lib),
            motion_file=getattr(lib, "motion_file", None),
        )
        out.update(_archive_coverage(success_dir, int(lib.num_trajectories)))
    except Exception as exc:  # noqa: BLE001 - artefact must not kill training
        print(f"[success] export failed, continuing: {exc!r}", flush=True)
        traceback.print_exc()
    return out


def _archive_coverage(success_dir: Path, total: int) -> dict[str, float]:
    """Share of motions the success archive holds (drives eval.stop_when_all_solved)."""
    manifest = Path(success_dir) / "manifest.json"
    n = len(json.loads(manifest.read_text()).get("saved", [])) if manifest.exists() else 0
    return {"archived_motions": float(n), "archived_frac": float(n) / max(1, total)}


def _ever_success(
    eval_cfg: DictConfig,
    tp_cb: Any,
    fail: torch.Tensor,
    fail_obj: torch.Tensor,
    scored: torch.Tensor,
    ever_state: dict,
) -> dict[str, float]:
    """Cumulative per-trajectory "ever succeeded": a trajectory counts once ANY replica in
    ANY eval so far succeeded (full / object-only)."""
    npm = max(1, int(eval_cfg.num_per_motion))
    n_traj = fail.shape[0] // npm
    if n_traj * npm != fail.shape[0]:
        print(f"[eval] ever-success skipped: {fail.shape[0]} envs % npm {npm} != 0")
        return {}

    def traj_any(f: torch.Tensor) -> torch.Tensor:
        return ((~f) & scored[:, None]).reshape(n_traj, npm, -1).any(dim=1)  # (M, K)

    def accumulate(key: str, succ: torch.Tensor) -> torch.Tensor:
        prev = ever_state.get(key)
        if prev is None or tuple(prev.shape) != tuple(succ.shape):
            if prev is not None:
                print(f"[eval] ever '{key}' shape {tuple(prev.shape)} != {tuple(succ.shape)}: reset", flush=True)
            prev = torch.zeros_like(succ)
        ever_state[key] = prev.to(succ.device) | succ
        return ever_state[key]

    full = accumulate("full", traj_any(fail))
    obj = accumulate("obj", traj_any(fail_obj))

    metrics: dict[str, float] = {}
    for i, kk in enumerate(tp_cb.threshold_ks):
        metrics[f"success_ever_{kk:.1f}"] = float(full[:, i].float().mean().item())
        metrics[f"success_ever_obj_{kk:.1f}"] = float(obj[:, i].float().mean().item())
    print(
        f"[eval] success_ever@1.0 full={metrics['success_ever_1.0']:.4f} "
        f"obj={metrics['success_ever_obj_1.0']:.4f} "
        f"({int(ever_state['full'][:, tp_cb.k1_idx].sum())}/{n_traj} trajs)",
        flush=True,
    )
    return metrics


def _spider_ever(
    raw_eval: Any, tp_cb: Any, st: list, scored: torch.Tensor, ever_state: dict | None
) -> dict[str, float]:
    """SPIDER ever, scattered by motion id: Isaac's per-env-object eval plan is not
    motion-major (env index != motion index), so the reshape in _ever_success does not apply."""
    sp_ok = tp_cb.per_env_spider_ok(st) if hasattr(tp_cb, "per_env_spider_ok") else None
    if ever_state is None or sp_ok is None:
        return {}
    mids = tp_cb.per_env_motion_ids(st)
    succ = torch.zeros(int(raw_eval.eval_num_motions), dtype=torch.bool, device=sp_ok.device)
    succ[mids[sp_ok & scored]] = True
    prev = ever_state.get("spider")
    if prev is None or tuple(prev.shape) != tuple(succ.shape):
        prev = torch.zeros_like(succ)
    ever_state["spider"] = prev.to(succ.device) | succ
    return {"success_ever_spider": float(ever_state["spider"].float().mean().item())}


# ─── helpers ───────────────────────────────────────────────────────────────────
def _metric_cb(callbacks: list, states: dict[int, list], attr: str) -> Any | None:
    """First metric callback (one with collected states) that exposes ``attr``."""
    return next((cb for cb in callbacks if id(cb) in states and hasattr(cb, attr)), None)


def _motion_names(lib: Any) -> list[str] | None:
    return [f.rsplit("#", 1)[-1] for f in getattr(lib, "motion_files", [])] or None
