"""Learning agents: ``runner.agent.class_name`` picks the implementation (e.g. agents.flashsac.FlashSACAgent)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import Agent, Policy

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from envs.vec_env import FlatObsVecEnv

__all__ = ["Agent", "Policy", "build_agent", "load_policy"]


def _agent_class(cfg: DictConfig) -> type[Agent]:
    module_name, _, class_name = str(cfg.class_name).rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Agent)):
        raise TypeError(f"runner.agent.class_name={cfg.class_name!r} is not an agents.Agent")
    return cls


def build_agent(cfg: DictConfig, env: FlatObsVecEnv, env_info: dict[str, Any], command_name: str) -> Agent:
    """Import ``cfg.class_name`` ("module.Class") and build it from the rest of ``cfg``."""
    return _agent_class(cfg).build(cfg, env, env_info, command_name)


def load_policy(
    cfg: DictConfig, env: FlatObsVecEnv, env_info: dict[str, Any],
    command_name: str, checkpoint: str,
) -> Policy:
    """Use the same registered agent class to load its acting owner for evaluation."""
    return _agent_class(cfg).load_policy(cfg, env, env_info, command_name, checkpoint)
