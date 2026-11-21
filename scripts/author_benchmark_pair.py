"""Rehearse a matched stock FlashDexRetarget comparison without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_AUTHOR = "6f95e133aa4b7ca6a784eac93725df3de4d41509"
_BUDGET = {"xhand": 300_000_000, "sharpa": 400_000_000}
_CURRENT = Path(__file__).resolve().parents[1]


def _revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _official_clips(author: Path) -> set[str]:
    launcher = (author / "scripts/retarget/50_motion_benchmark.sh").read_text()
    rows = launcher.split('BENCHMARK="\n', 1)[1].split('\n"', 1)[0].splitlines()
    clips = {row.split()[1] for row in rows}
    if len(rows) != 50 or len(clips) != 50:
        raise ValueError("pinned author benchmark must declare 50 unique clips")
    return clips


def _validate_motion(motion: Path, clips: set[str]) -> None:
    import torch

    # The operator supplies their own trusted author-produced motion pack.
    names = torch.load(motion, map_location="cpu", weights_only=False)["motion_filename"]
    if len(names) != len(clips) or set(names) != clips:
        raise ValueError(f"packed roster differs from author benchmark; missing {sorted(clips - set(names))}, "
                         f"extra {sorted(set(names) - clips)}, total entries {len(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", required=True, type=Path, help="checkout of the pinned upstream commit")
    parser.add_argument("--motion", required=True, type=Path, help="same trusted author-packed motion.pt for both runs")
    parser.add_argument("--robot-usd", required=True, type=Path, help="same converted robot USD for both runs")
    parser.add_argument("--hand", choices=tuple(_BUDGET), required=True)
    parser.add_argument("--output", required=True, type=Path, help="parent of separate fresh run directories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compose-only", action="store_true", help="check configuration with no packed-data claim")
    args = parser.parse_args()

    author, motion, usd, output = (p.expanduser().resolve() for p in (
        args.author, args.motion, args.robot_usd, args.output,
    ))
    if _revision(author) != _AUTHOR or subprocess.check_output(
        ["git", "-C", str(author), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        parser.error("author checkout must match the pinned revision with clean tracked files")

    if not args.compose_only:
        if not usd.is_file() or not motion.is_file():
            parser.error("both the converted robot USD and the author-packed motion.pt are required")
        if subprocess.check_output(
            ["git", "-C", str(_CURRENT), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip():
            parser.error("commit the candidate's tracked changes before binding its executable revision")
        _validate_motion(motion, _official_clips(author))

    shared = [
        f"envs/robot@robot={args.hand}_bimanual",
        f"commands.motion_file={json.dumps(str(motion))}",
        f"seed={args.seed}",
        f"num_env_steps={_BUDGET[args.hand]}",
        "sim.dt=0.0041666667",
        "logger.wandb.mode=offline",
    ]
    result = {"status": "CONFIG_COMPOSED" if args.compose_only else "MOTION_ROSTER_VALIDATED", "arms": {}}
    evaluators = []
    conditions = []
    for label, root, directory_key in (("author", author, "save_path"), ("candidate", _CURRENT, "run_dir")):
        overrides = [*shared, f"{directory_key}={json.dumps(str(output / label))}"]
        if label == "candidate":
            overrides.append(f"sim.robot_asset.usd_path={json.dumps(str(usd))}")
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            cfg = compose(config_name="flashdexretarget", overrides=overrides)
        evaluators.append(OmegaConf.to_container(cfg.eval, resolve=False))
        conditions.append((cfg.seed, cfg.num_env_steps, cfg.num_envs, cfg.sim.dt, cfg.decimation))
        result["arms"][label] = {
            "revision": _revision(root),
            "cwd": str(root),
            "environment": {"FDR_ROBOT_USD": str(usd)},
            "argv": [sys.executable, "train.py", "--config_name", "flashdexretarget", *(
                arg for override in overrides for arg in ("--overrides", override)
            )],
            "seed": cfg.seed,
            "num_env_steps": cfg.num_env_steps,
            "num_envs": cfg.num_envs,
            "physics_dt": cfg.sim.dt,
            "decimation": cfg.decimation,
        }
    if evaluators[0] != evaluators[1]:
        raise ValueError("stock author and candidate evaluation configurations differ")
    if conditions[0] != conditions[1]:
        raise ValueError("stock author and candidate seed, budget, environment count or clock differ")
    result["evaluation"] = evaluators[0]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
