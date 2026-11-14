"""Env builders: pure per-manager ``cfg -> manager dict`` functions (rewards/obs/commands/...);
``create_envs`` assembles the selected train/eval envs from a Hydra cfg."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The vendored mjlab-compat subset must be importable before the builder imports.
_VENDOR = str(Path(__file__).resolve().parents[1] / "simulator" / "isaacsim" / "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from .actions import build_actions  # noqa: E402
from .commands import build_commands  # noqa: E402
from .curriculum import build_curriculum  # noqa: E402
from .events import build_events  # noqa: E402
from .observations import build_observations  # noqa: E402
from .rewards import build_rewards  # noqa: E402
from .terminations import build_terminations  # noqa: E402

__all__ = [
    "build_actions", "build_commands", "build_curriculum", "build_events",
    "build_observations", "build_rewards", "build_terminations", "build_manager_cfgs", "create_envs",
]


def create_envs(cfg: Any, device: str = "cuda"):
    """Build the selected simulator without importing other engines."""
    from hydra.utils import get_method

    return get_method(cfg.sim._target_)(cfg, device=device)


def build_manager_cfgs(cfg) -> dict:
    """One task recipe, reused by every scene/simulator implementation."""
    entity_name, command_name = cfg.robot.entity_name, cfg.commands.name
    return dict(
        observations=build_observations(cfg.obs, command_name=command_name, entity_name=entity_name),
        rewards=build_rewards(cfg.rewards, command_name=command_name, entity_name=entity_name),
        commands=build_commands(cfg.commands),
        actions=build_actions(cfg.actions),
        terminations=build_terminations(cfg.terminations, command_name=command_name, entity_name=entity_name),
        curriculum=build_curriculum(cfg.curriculum, command_name=command_name, entity_name=entity_name),
        events=build_events(cfg.events),
    )
