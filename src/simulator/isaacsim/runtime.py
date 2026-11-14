"""Isaac-only app lifetime; native MJCF engines do not import this module."""

from __future__ import annotations

import os
import sys


class IsaacRuntime:
    def __init__(self, cfg):
        from isaaclab.app import AppLauncher

        self.launcher = AppLauncher({"headless": bool(cfg.get("headless", True))})

    def close(self):
        # Isaac Sim 5.1 segfaults in app.close(); the normal runner has flushed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def launch(cfg):
    # License acceptance is supplied by the owning operator's launch environment.
    return IsaacRuntime(cfg)
