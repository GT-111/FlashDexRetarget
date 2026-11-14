from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(action_space) -> dict:
    """Learning draws and the Gym sampler have independent generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "action_space": action_space.np_random.bit_generator.state,
    }


def restore_rng_state(state: dict, action_space) -> None:
    """Restore after constructors, physical reset and owner-state loading."""
    if len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("learning RNG state requires the same visible CUDA device layout")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    for device, rng in enumerate(state["torch_cuda"]):
        torch.cuda.set_rng_state(rng, device)
    action_space.np_random.bit_generator.state = state["action_space"]


def configure_torch_backends() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
