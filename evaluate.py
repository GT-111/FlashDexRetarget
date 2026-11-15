"""Evaluate one frozen checkpoint through the ordinary environment and callbacks."""

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
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    resume = json.loads((checkpoint / "resume_state.json").read_text())
    env_step = int(resume["env_step"])
    run_dir = Path(cfg.run_dir).expanduser().resolve()
    cfg.run_dir = str(run_dir)
    OmegaConf.resolve(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"run_dir must be new or empty; existing contents are preserved: {run_dir}")
    OmegaConf.save(cfg, str(run_dir / "config.yaml"))
    authority = {
        "status": "PREPARED_NOT_EVALUATED",
        "checkpoint": str(checkpoint),
        "training_env_step": env_step,
        "resume_state": resume,
        "executable_source": str(Path(__file__).resolve()),
        "executable_source_revision": source_revision,
        "factual_recording_dir": None if recording_dir is None else str(recording_dir),
        "training_resume_fields": "Not consumed: this entry does not construct a runner or load learning/replay state.",
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
            train_command = train_env.unwrapped.command_manager.get_term(cfg.eval.command_name)
            heads_cfg = cfg.runner.agent.get("reward_heads", None)
            flat = FlatObsVecEnv.from_env(
                train_env.unwrapped, command=train_command,
                share_obs=bool(cfg.runner.share_actor_critic_obs),
                reward_heads=OmegaConf.to_container(heads_cfg, resolve=True) if heads_cfg is not None else None,
            )
            # This is the ordinary setup reset that supplies the agent's actual
            # layout/head/action metadata, not an extra scored attempt.
            _, env_info = flat.reset()
            policy = load_policy(cfg.runner.agent, flat, env_info, cfg.eval.command_name, str(checkpoint))
            OmegaConf.save(cfg, str(run_dir / "config.yaml"))
            raw_eval = eval_env.unwrapped
            command = raw_eval.command_manager.get_term(cfg.eval.command_name)
            authority["policy"] = policy.checkpoint_authority
            authority["configured_disable_object_pin"] = bool(cfg.eval.env.disable_object_pin)
            authority["pin_before_eval"] = bool(command.cfg.object.pin_objects)
            eval_env.enter_eval()
            authority["pin_during_eval"] = bool(command.cfg.object.pin_objects)
            authority["shared_eval_wrench_clear"] = "Ordinary enter_eval called its existing object-wrench clear; this is a submitted clear, not force measurement."
            authority["layout"] = {
                "flat_observation_shape": list(flat.single_observation_space.shape),
                "action_shape": list(flat.single_action_space.shape),
                "env_info": {name: env_info.get(name) for name in (
                    "actor_observation_size", "encoded_obs_dims", "obs_terms", "reward_heads", "action_sides")},
            }
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
            metrics = evaluate(
                policy, eval_env, cfg.eval, cfg.device,
                ever_state=None, success_dir=None, env_step=env_step,
                outcome_dir=run_dir / "evaluation",
                recording_dir=recording_dir,
                frozen_authority=(None if recording_dir is None else {
                    "checkpoint": str(checkpoint), "source_revision": source_revision,
                    "training_env_step": env_step, "policy": policy.checkpoint_authority,
                }),
            )
            print(f"[evaluation] training_env_step={env_step} {metrics}", flush=True)
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recording_dir", help="Retain every attempt with the configured rollout recorder")
    parser.add_argument("--config_path", default="config")
    parser.add_argument("--config_name", default="flashdexretarget")
    parser.add_argument("--overrides", action="append", default=[])
    run(parser.parse_args())
