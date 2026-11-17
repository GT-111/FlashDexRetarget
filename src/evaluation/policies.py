"""Configured nonparametric policies for the ordinary evaluation interface."""

from __future__ import annotations

import numpy as np
import torch


class NativeReferenceFollower:
    """Follow the admitted native reference through the existing action owner.

    Fingers use its actual independent command bounds. FREE roots request zero
    correction, leaving reference pose/rate installation to NativeResidualAction.
    No checkpoint, learned parameter, target-Q relabeling or object aid is used.
    """

    def __init__(self, *, env, command_name: str, action_name: str, method_id: str):
        self.env = env
        self.command = env.command_manager.get_term(command_name)
        self.action = env.action_manager.get_term(action_name)
        if env.action_manager.total_action_dim != self.action.action_dim:
            raise ValueError("reference follower must own the complete configured native action")
        self.scale = float(self.action.cfg.action_scale)
        self.offset = float(self.action.cfg.action_offset)
        if not np.isfinite(self.scale) or self.scale == 0 or not np.isfinite(self.offset):
            raise ValueError("reference follower requires an invertible finite action affine")
        if self.action.cfg.finger_residual_rad is not None:
            raise ValueError("reference follower requires the configured absolute finger action")
        if self.action.cfg.cutoff_hz.get("finger") is not None:
            raise ValueError("reference follower requires unfiltered reference finger requests")
        self.finger_columns = tuple(self.action.action_names.index(name)
                                    for name in self.action.finger_actuator_names)
        limits = self.action.finger_command_limits
        self.lower, self.upper = limits.unbind(-1)
        if not torch.isfinite(limits).all() or not (self.upper > self.lower).all():
            raise ValueError("reference follower requires the actual finite command bounds")
        self.acting_authority = {
            "kind": "analytic", "method_id": str(method_id),
            "policy_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "has_learned_parameters": False, "checkpoint": None,
            "finger_action": "inverse configured affine of actual command-bound normalized reference",
            "wrist_action": "zero correction; ordinary native reference pose and independently supported rates",
            "unsupported_commands": "ordinary action-owner installed-target hold",
            "command_name": str(command_name), "action_name": str(action_name),
            "action_names": list(self.action.action_names),
            "finger_joint_names": list(self.action.finger_joint_names),
            "deterministic": True,
        }

    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        reference = self.command.command_reference()
        names = tuple(reference["joint_names"])
        columns = [names.index(name) for name in self.action.finger_joint_names]
        position = reference["position"][..., columns]
        support = reference["position_support"][..., columns]
        normalized = 2 * (position - self.lower) / (self.upper - self.lower) - 1
        processed = position.new_zeros(self.env.num_envs, self.action.action_dim)
        processed[:, self.finger_columns] = torch.where(support, normalized, 0.)
        return ((processed - self.offset) / self.scale).detach().cpu().numpy()
