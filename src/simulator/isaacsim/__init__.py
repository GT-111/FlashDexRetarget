"""IsaacSim backend: mjlab-compatible adapters so the src/envs manager terms
run unchanged. bootstrap() puts the vendored mjlab-compat on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = str(Path(__file__).resolve().parent / "_vendor")

# Available before launching Kit so qualification cannot spend the training
# budget on a plant whose required evidence ports have yet to be implemented.
HOI_QUALIFICATION_UNAVAILABLE_READERS = (
    "hand_part_contact_interval_facts",
    "applied_object_wrench_interval_facts",
    "physics_execution_clock",
)


def bootstrap() -> None:
    """Make the vendored mjlab-compat subset importable (idempotent)."""
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)


bootstrap()


def create_envs_isaac(cfg, device: str = "cuda"):
    """Deferred import: needs the Isaac Sim app to be running."""
    from .create import create_envs_isaac as _impl

    return _impl(cfg, device=device)
