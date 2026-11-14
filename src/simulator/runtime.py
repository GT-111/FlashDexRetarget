"""Config-selected runtime launch before simulator-specific heavy imports."""


def launch_runtime(cfg):
    from hydra.utils import get_method

    launcher = cfg.sim.runtime_launcher
    return get_method(launcher)(cfg) if launcher is not None else None
