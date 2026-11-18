from __future__ import annotations

import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

# Make ``from envs import ...`` (and siblings under src/) resolve when the
# script is run directly. Must happen before the package imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse

import hydra
from omegaconf import OmegaConf


def run(args: argparse.Namespace) -> None:
    # ─── Hydra compose ────────────────────────────────────────────────────
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    # initialize_config_dir takes an absolute path → CWD-independent.
    config_dir = (Path(__file__).resolve().parent / args.config_path).resolve()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    run_dir = Path(cfg.run_dir).expanduser().resolve()
    cfg.run_dir = str(run_dir)
    OmegaConf.resolve(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"run_dir must be new or empty; existing contents are preserved: {run_dir}")
    OmegaConf.save(cfg, str(run_dir / "config.yaml"))

    # The selected runtime boots before Torch-heavy imports; native paths need no Kit.
    from simulator.runtime import launch_runtime

    runtime, status = None, 1
    try:
        runtime = launch_runtime(cfg)
        with ExitStack() as cleanup:
            from common.logger import close_logger, setup_logger
            from common.utils import configure_torch_backends, set_seed
            from envs import create_envs
            from runner import create_runner

            cleanup.callback(close_logger)
            set_seed(cfg.seed)
            configure_torch_backends()
            train_env, eval_env = create_envs(cfg, device=cfg.device)
            cleanup.callback(eval_env.close)
            cleanup.callback(train_env.close)
            setup_logger(cfg.logger, cfg.runner)
            runner = create_runner(cfg.runner, cfg.eval, train_env, eval_env=eval_env,
                                   headless=bool(cfg.get("headless", True)))
            runner.learn(num_learning_iterations=int(cfg.runner.max_iterations),
                         init_at_random_ep_len=True)
        status = 0
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            status = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        elif runtime is not None:
            # SDK shutdown can exit before Python prints the pending exception.
            traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if runtime is not None:
            runtime.close(exit_code=status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="config")
    parser.add_argument("--config_name", type=str, default="flashdexretarget")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--gui", action="store_true", help="isaacsim viewport mode")
    args = parser.parse_args()
    if args.gui:
        args.overrides = list(args.overrides) + ["headless=false"]
    run(args)
