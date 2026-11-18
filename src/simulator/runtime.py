"""Config-selected runtime launch before simulator-specific heavy imports."""

import os
from pathlib import Path


def prepare_runtime_directories():
    """Create explicitly requested temporary/compiler paths before engine boot."""
    for name in (
        "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "WARP_CACHE_PATH",
    ):
        value = os.environ.get(name)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)


def launch_runtime(cfg):
    prepare_runtime_directories()
    from hydra.utils import get_method

    launcher = cfg.sim.runtime_launcher
    return get_method(launcher)(cfg) if launcher is not None else None
