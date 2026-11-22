"""Evaluate a frozen checkpoint or configured analytic Policy with ordinary callbacks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def _restore_hoi_zero_curriculum(command, checkpoint, env_step, source_revision, control_step,
                                 actor_optimizer_steps):
    import torch

    from agents.flashsac import REPLAY_BUFFER_FILE
    from runner.off_policy_runner import LEARNING_STATE_FILE

    if not (checkpoint / REPLAY_BUFFER_FILE).is_file():
        raise ValueError("HOI qualification requires a FULL checkpoint including replay")
    state = torch.load(checkpoint / LEARNING_STATE_FILE, map_location="cpu", weights_only=False)
    ctrl = command._xfrc_curr_ctrl
    saved = state.get("native_hoi_zero")
    if (ctrl is None or not saved or state["env_step"] != env_step
            or state.get("native_hoi_source_revision") != source_revision):
        raise ValueError("FULL checkpoint does not bind a continuous zero-hold receipt and this executable source")
    ctrl.load_state_dict(state["curriculum"], control_step)
    if ctrl.hoi_zero_receipt() != saved:
        raise ValueError("actual restored FULL curriculum disagrees with its zero-hold receipt")
    optimization = state.get("native_hoi_zero_optimization")
    if (optimization is None or optimization["actor_steps_at_checkpoint"] != actor_optimizer_steps
            or optimization["actor_steps_at_zero_start"] >= actor_optimizer_steps):
        raise ValueError("FULL checkpoint does not bind an actor optimizer update during the zero hold")
    return {**saved, "restored": True, "zero_assist_actor_optimizer_steps": (
        actor_optimizer_steps - optimization["actor_steps_at_zero_start"]
    )}


def _recorded_wrench_summary(outcome_dir):
    report, = outcome_dir.glob("current-policy-*.json")
    outcome = json.loads(report.read_text())
    manifest = (outcome_dir / outcome["factual_recording_manifest"]).resolve()
    recordings = [attempt["recording"] for motion in json.loads(manifest.read_text())["motions"]
                  for attempt in motion["attempts"]]
    records = [recording.get("applied_wrench_interval") for recording in recordings]
    measured = [record for record in records if record is not None]
    measured_intervals = sum(record["measured_intervals"] for record in measured)
    complete = bool(measured_intervals and len(measured) == len(records) and all(record["complete"] for record in measured))
    force = max((record["peak_force_n"] for record in measured if record["measured_intervals"]), default=None)
    torque = max((record["peak_torque_nm"] for record in measured if record["measured_intervals"]), default=None)
    if not complete:
        status = "MEASURED_PARTIAL_POST_SOLVE" if measured_intervals else "MEASURED_NO_SUPPORTED_STEPS"
    else:
        status = "MEASURED_ZERO_POST_SOLVE" if force == 0.0 and torque == 0.0 else "MEASURED_NONZERO_POST_SOLVE"
    qualification = outcome.get("frozen_authority", {}).get("hoi_qualification")
    schedule = {}
    if qualification is not None:
        proof = [row.get("hoi_evaluation_perturbation", {}) for row in recordings]
        schedule = {
            "verified_perturbation_attempts": sum(row.get("status") == "verified" for row in proof),
            "unusable_source_phases": sum(row.get("status") == "no_usable_source_phase" for row in proof),
        }
        if schedule["verified_perturbation_attempts"] != len(recordings):
            status = "MEASURED_INCOMPLETE_DETERMINISTIC_PERTURBATION"
    return status, {
        "manifest": str(manifest),
        "attempts": len(records),
        "complete_attempts": sum(record["complete"] for record in measured),
        "measured_intervals": measured_intervals,
        "max_force_n": force,
        "max_torque_nm": torque,
        **schedule,
    }


def _native_evaluation_backend(env):
    """Identify the installed plant and its available qualification readers."""
    scene = env.scene
    readers = ("hand_part_contact_interval_facts", "applied_object_wrench_interval_facts",
               "physics_execution_clock")
    missing = [name for name in readers if (read := getattr(scene, name, None)) is None or read() is None]
    identity = dict(scene.backend_identity)
    if identity.get("unavailable_hand_part_reports") and readers[0] not in missing:
        missing.append(readers[0])
    if identity.get("unavailable_fixed_support_reports"):
        missing.append("support_contact_interval_pairs")
    return {
        "identity": identity,
        "physics_dt_s": float(env.physics_dt),
        "control_dt_s": float(env.step_dt),
        "qualification_evidence_available": not missing,
        "missing_qualification_readers": missing,
    }


def run(args: argparse.Namespace) -> None:
    source_revision = os.environ.get("FLASH_SOURCE_REF")
    recording_dir = (None if args.recording_dir is None
                     else Path(args.recording_dir).expanduser().resolve())
    if recording_dir is not None and not source_revision:
        raise ValueError("--recording_dir requires FLASH_SOURCE_REF for the actual executable revision")
    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    config_dir = (Path(__file__).resolve().parent / args.config_path).resolve()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)

    from common.native_reference import bind_prepared_native_reference

    bind_prepared_native_reference(cfg)
    policy_cfg = cfg.eval.get("policy")
    if (args.checkpoint is None) == (policy_cfg is None):
        raise ValueError("select exactly one --checkpoint or configured eval.policy")
    checkpoint = None if args.checkpoint is None else Path(args.checkpoint).expanduser().resolve()
    qualification_mode = getattr(args, "hoi_qualification", None)
    if qualification_mode is not None:
        if checkpoint is None or recording_dir is None:
            raise ValueError("HOI qualification requires --checkpoint and --recording_dir")
        if (cfg.eval.command.sampling_mode != "start" or int(cfg.eval.command.start_frame) != 0
                or float(cfg.eval.command.noise_to_initial_level) != 0.0):
            raise ValueError("HOI qualification requires the whole reference from its exact reset")
    resume = None if checkpoint is None else json.loads((checkpoint / "resume_state.json").read_text())
    env_step = 0 if resume is None else int(resume["env_step"])
    run_dir = Path(cfg.run_dir).expanduser().resolve()
    cfg.run_dir = str(run_dir)
    OmegaConf.resolve(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"run_dir must be new or empty; existing contents are preserved: {run_dir}")
    OmegaConf.save(cfg, str(run_dir / "config.yaml"))
    authority = {
        "status": "PREPARED_NOT_EVALUATED",
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "training_env_step": None if resume is None else env_step,
        "resume_state": resume,
        "executable_source": str(Path(__file__).resolve()),
        "executable_source_revision": source_revision,
        "factual_recording_dir": None if recording_dir is None else str(recording_dir),
        "training_resume_fields": "HOI qualification restores the FULL curriculum; evaluation consumes no training replay.",
        "assistance_scope": "Saved training curriculum and evaluation pin authority are separate.",
        "wrench_measurement": "UNKNOWN: no accepted/applied object-wrench readback is exposed by the ordinary evaluation interface.",
    }
    receipt = run_dir / "evaluation-entry.json"
    receipt.write_text(json.dumps(authority, indent=2) + "\n")

    # Same config-selected runtime boot and cleanup order as the public trainer.
    from simulator.runtime import launch_runtime

    runtime, status = None, 1
    try:
        runtime = launch_runtime(cfg)
        with ExitStack() as cleanup:
            from agents import load_policy
            from common.utils import configure_torch_backends, set_seed
            from envs import create_envs
            from envs.vec_env import FlatObsVecEnv
            from evaluation.evaluate import evaluate

            set_seed(cfg.seed)
            configure_torch_backends()
            train_env, eval_env = create_envs(cfg, device=cfg.device)
            cleanup.callback(eval_env.close)
            cleanup.callback(train_env.close)
            raw_eval = eval_env.unwrapped
            command = raw_eval.command_manager.get_term(cfg.eval.command_name)
            if policy_cfg is None:
                train_command = train_env.unwrapped.command_manager.get_term(cfg.eval.command_name)
                heads_cfg = cfg.runner.agent.get("reward_heads", None)
                flat = FlatObsVecEnv.from_env(
                    train_env.unwrapped, command=train_command,
                    share_obs=bool(cfg.runner.share_actor_critic_obs),
                    reward_heads=OmegaConf.to_container(heads_cfg, resolve=True) if heads_cfg is not None else None,
                )
                # Ordinary loader setup, not an additional scored attempt.
                _, env_info = flat.reset()
                policy = load_policy(cfg.runner.agent, flat, env_info, cfg.eval.command_name, str(checkpoint))
                authority["policy"] = policy.checkpoint_authority
                authority["layout"] = {
                    "flat_observation_shape": list(flat.single_observation_space.shape),
                    "action_shape": list(flat.single_action_space.shape),
                    "env_info": {name: env_info.get(name) for name in (
                        "actor_observation_size", "encoded_obs_dims", "obs_terms", "reward_heads", "action_sides")},
                }
            else:
                policy = hydra.utils.instantiate(policy_cfg, env=raw_eval)
                authority["policy"] = policy.acting_authority
                authority["layout"] = {"action_shape": list(raw_eval.action_space.shape),
                                       "observation_consumption": "configured analytic Policy; no learned observation encoder"}
            OmegaConf.save(cfg, str(run_dir / "config.yaml"))
            if qualification_mode is not None:
                if not hasattr(command, "native_layout"):
                    raise ValueError("HOI qualification requires a prepared native scene")
                authority["runtime_backend"] = _native_evaluation_backend(raw_eval)
                zero = _restore_hoi_zero_curriculum(command, checkpoint, env_step, source_revision,
                                                    raw_eval.common_step_counter,
                                                    authority["policy"]["actor_optimizer_steps"])
                authority["hoi_qualification"] = {"mode": qualification_mode, "zero_curriculum": zero}
            authority["configured_disable_object_pin"] = bool(cfg.eval.env.disable_object_pin)
            authority["pin_before_eval"] = bool(command.cfg.object.pin_objects)
            eval_env.enter_eval()
            if qualification_mode is not None:
                command.enable_hoi_evaluation_perturbation(qualification_mode)
            authority["pin_during_eval"] = bool(command.cfg.object.pin_objects)
            authority["shared_eval_wrench_clear"] = "Ordinary enter_eval called its existing object-wrench clear; this is a submitted clear, not force measurement."
            authority["evaluation_plan"] = {
                "worlds": int(raw_eval.num_envs),
                "motion_count": int(raw_eval.eval_num_motions),
                "batches": int(raw_eval.eval_num_batches),
                "replicas_per_motion": int(cfg.eval.num_per_motion),
                "configured_command": OmegaConf.to_container(cfg.eval.command, resolve=True),
                "manager_physics_dt_s": float(raw_eval.physics_dt),
                "manager_control_dt_s": float(raw_eval.step_dt),
                "clock_authority": "Manager fields; actual SDK clock qualification remains the selected scene/runtime receipt.",
            }
            authority["status"] = "ENTERED_EVAL_NOT_COMPLETED"
            receipt.write_text(json.dumps(authority, indent=2) + "\n")
            frozen_authority = None if recording_dir is None else {
                "checkpoint": None if checkpoint is None else str(checkpoint),
                "source_revision": source_revision,
                "training_env_step": None if resume is None else env_step, "policy": authority["policy"],
            }
            if qualification_mode is not None:
                frozen_authority["hoi_qualification"] = authority["hoi_qualification"]
            metrics = evaluate(
                policy, eval_env, cfg.eval, cfg.device,
                ever_state=None, success_dir=None, env_step=env_step,
                outcome_dir=run_dir / "evaluation",
                recording_dir=recording_dir,
                frozen_authority=frozen_authority,
            )
            print(f"[evaluation] training_env_step={env_step} {metrics}", flush=True)
            if recording_dir is not None:
                authority["wrench_measurement"], authority["applied_object_wrench_summary"] = (
                    _recorded_wrench_summary(run_dir / "evaluation")
                )
            authority["status"] = "COMPLETED_CURRENT_POLICY_EVALUATION"
            authority["pin_after_eval"] = bool(command.cfg.object.pin_objects)
            authority["success_ever_or_archive_selection"] = "Not loaded or selected; ordinary current-policy outcomes retain all attempted/failed/unscored motions."
            receipt.write_text(json.dumps(authority, indent=2) + "\n")
        status = 0
    except BaseException as exc:
        authority["status"] = "FAILED_CURRENT_POLICY_EVALUATION"
        authority["exception"] = f"{type(exc).__name__}: {exc}"
        receipt.write_text(json.dumps(authority, indent=2) + "\n")
        if isinstance(exc, SystemExit):
            status = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        elif runtime is not None:
            traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if runtime is not None:
            runtime.close(exit_code=status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", help="Frozen learned policy; omit when eval.policy selects an analytic Policy")
    parser.add_argument("--recording_dir", help="Retain every attempt with the configured rollout recorder")
    parser.add_argument("--hoi_qualification", choices=("nominal", "perturbed"),
                        help="Record an exact zero-hold FULL checkpoint under the selected HOI test condition")
    parser.add_argument("--config_path", default="config")
    parser.add_argument("--config_name", default="flashdexretarget")
    parser.add_argument("--overrides", action="append", default=[])
    run(parser.parse_args())
