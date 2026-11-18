from __future__ import annotations

from pathlib import Path

import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf


def setup_logger(logger_cfg: DictConfig, runner_cfg: DictConfig) -> None:
    """Use the entry's admitted run directory, initialize the backend, and mutate ``runner_cfg``:
    logger-related keys + ``log_dir`` are added for the runner builder (``train_cfg``)."""
    log_dir = Path(runner_cfg.run_dir)
    OmegaConf.update(runner_cfg, "log_dir", str(log_dir), force_add=True)

    if logger_cfg.type == "wandb":
        _init_wandb(logger_cfg.wandb, str(log_dir))
        _inject_wandb_runner_keys(logger_cfg, runner_cfg)
    else:
        raise NotImplementedError(f"logger.type={logger_cfg.type!r} not supported")


def close_logger() -> None:
    if wandb.run is not None:
        wandb.finish()


def _init_wandb(wb: DictConfig, log_dir: str) -> None:
    if wandb.run is not None:
        return

    resume_arg: str | None = None
    resume_id: str | None = None
    if wb.resume:
        if wb.get("resumeid") is None:
            raise ValueError(
                "logger.wandb.resume=true but logger.wandb.resumeid is null. "
                "Set resumeid=<wandb run id> in logger/wandb.yaml or via "
                "`logger.wandb.resumeid=<id>` override to resume that run."
            )
        resume_arg = "must"
        resume_id = wb.resumeid

    tags = _resolve_tags(wb.tags)

    wandb.init(
        project=wb.project_name,
        entity=wb.entity,
        name=wb.run_name,
        group=wb.get("group_name"),
        job_type=wb.get("job_type"),
        mode=wb.get("mode", "online"),
        dir=log_dir,
        tags=tags or None,
        resume=resume_arg,
        id=resume_id,
    )

    if wb.resume:
        _allow_config_overwrite_on_resume()


def _allow_config_overwrite_on_resume() -> None:
    """Let rsl_rl re-store train_cfg/env_cfg on a resumed wandb run: default
    ``allow_val_change=True`` at the ``Config.update`` level for this process."""
    from wandb.sdk import wandb_config as _wc

    if getattr(_wc.Config.update, "_resume_patched", False):
        return
    _orig_update = _wc.Config.update

    def _update(self, d, allow_val_change=True, _orig=_orig_update):
        return _orig(self, d, allow_val_change=allow_val_change)

    _update._resume_patched = True
    _wc.Config.update = _update


def _inject_wandb_runner_keys(logger_cfg: DictConfig, runner_cfg: DictConfig) -> None:
    """Set the wandb-related fields rsl_rl reads from train_cfg."""
    wb = logger_cfg.wandb
    values = {
        "logger": logger_cfg.type,  # "wandb"
        "wandb_project": wb.project_name,
        "wandb_tags": _resolve_tags(wb.tags),
        "run_name": wb.run_name,
        "resume": bool(wb.resume),
        "upload_model": bool(wb.get("upload_model", True)),
    }
    for k, v in values.items():
        OmegaConf.update(runner_cfg, k, v, force_add=True)


def _resolve_tags(tags_cfg) -> list[str]:
    """Coerce a yaml ``tags`` value (``null`` or list) to a plain list."""
    if tags_cfg is None:
        return []
    if isinstance(tags_cfg, ListConfig):
        return list(OmegaConf.to_container(tags_cfg, resolve=True))
    return list(tags_cfg)


def log_metrics(metrics: dict[str, float], step: int) -> None:
    """Log every supplied scalar to wandb and stdout without rounding."""
    if not metrics:
        return
    wandb.log(metrics, step=step)
    print(f"[train] step={step} {metrics}", flush=True)
