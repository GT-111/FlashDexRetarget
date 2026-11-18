"""Isaac-only app lifetime; native MJCF engines do not import this module."""

from __future__ import annotations

import os
import sys


class IsaacRuntime:
    def __init__(self, cfg):
        from isaaclab.app import AppLauncher

        self.launcher = AppLauncher({
            "headless": bool(cfg.get("headless", True)),
            "device": str(cfg.device),
            "multi_gpu": False,
            "kit_args": str(cfg.sim.kit_args),
        })

    def close(self, exit_code: int):
        # Isaac Sim 5.1 segfaults in app.close(); the entry has cleaned up and
        # reported any original exception before this SDK-specific process exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def launch(cfg):
    # License acceptance is supplied by the owning operator's launch environment.
    return IsaacRuntime(cfg)
