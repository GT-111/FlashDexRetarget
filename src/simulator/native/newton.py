"""Flash native task over the installed DexForge public Newton engine."""

from __future__ import annotations

from importlib import metadata

import numpy as np
import torch

from .plant import NativePlant


class NewtonPlant(NativePlant):
    def __init__(self, *, scene_xml_path, num_envs, sim_dt, device, contact_pairs,
                 collision_generation=None, cuda_graph=True):
        from dexforge_tracking.envs.backend import create_backend
        from dexforge_tracking.envs.physics import (
            DeviceTrackingBackend,
            NativeSceneSpec,
        )

        self.backend = create_backend(
            "newton", NativeSceneSpec(scene_xml_path, collision_generation=collision_generation),
            num_envs, sim_dt, device=device, cuda_graph=cuda_graph,
        )
        if not isinstance(self.backend, DeviceTrackingBackend):
            raise ValueError("Flash Newton requires the real device backend")
        self.backend.materialize_device()
        self.backend.assert_torch_device_identity(device)
        self._init_buffers(self.backend.model, num_envs, device, contact_pairs)
        self.has_applied_wrench_readback = hasattr(self.backend, "get_applied_body_wrench_w_torch")
        self._state = self.backend.public_state_torch()
        self._world_ids = torch.arange(num_envs, device=device)
        self._body_ids = torch.arange(1, self.model.nbody, device=device)
        self.effort = torch.zeros(num_envs, self.model.nv, device=device)
        self.identity = {
            "backend": "newton", "solver": "SolverMuJoCo",
            "contact_generation": collision_generation,
            "native_engine": "installed_dexforge_tracking",
            "engine_versions": {
                package: metadata.version(package)
                for package in ("newton", "mujoco", "mujoco-warp", "warp-lang")
            },
            "scene_xml_path": scene_xml_path, "sim_dt_s": float(sim_dt),
        }
        self._bind_execution_clock(float(sim_dt))

    def _bind_execution_clock(self, configured_timestep_s):
        """Count accepted Newton calls, separately from accumulated solver time.

        Newton's public compiled native model retains the configured scalar dt;
        its owning backend passes that fixed value to each SolverMuJoCo step.
        This is not a claim about an exact sum of binary device timesteps.
        """
        actual = float(self.backend.model.opt.timestep)
        if (not np.isfinite(configured_timestep_s) or configured_timestep_s <= 0
                or actual != configured_timestep_s):
            raise ValueError("Newton execution clock requires the installed fixed native timestep")
        self._execution_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.contact_sample_execution_step = torch.full_like(self._execution_steps, -1)
        self._execution_clock_metadata = dict(
            schema="accepted_newton_fixed_step_clock_v1",
            authority="accepted public Newton backend step_torch calls on the owning stream",
            configured_timestep_s=configured_timestep_s,
            installed_native_timestep_s=actual, configured_native_timestep_verified=True,
            time_ns="round((accepted_steps - recorded_reset_steps) * configured_timestep_s * 1e9); not a sum of binary device dt",
            raw_solver_time="unchanged public backend physics time; separate accumulated floating engine state",
        )
        self.identity["execution_clock"] = self._execution_clock_metadata

    def execution_clock(self):
        return dict(steps=self._execution_steps, metadata=self._execution_clock_metadata)

    def state(self):
        value = self._state
        return value.qpos, value.qvel

    def reset(self, env_ids):
        q = torch.as_tensor(self.backend.get_default_qpos(), device=self.device, dtype=self.control.dtype)
        self.set_state(env_ids, q.expand(len(env_ids), -1), self.effort.new_zeros(len(env_ids), self.model.nv))
        self.control[env_ids] = 0
        self._execution_steps[env_ids] = 0

    def set_state(self, env_ids, qpos, qvel):
        self._state = self.backend.set_state_torch(env_ids, qpos=qpos, qvel=qvel)
        self.effort[env_ids] = 0
        self.external_body_wrench_w[env_ids] = 0
        self.applied_body_wrench_w[env_ids] = 0
        self.applied_body_wrench_support[env_ids] = False
        self.invalidate_contacts(env_ids)

    def advance(self):
        self.prepare_contact_substep()
        sample_time = self.physics_time()
        sample_step = self._execution_steps.clone()
        self.backend.set_body_wrench_w_torch(
            self._world_ids, body_ids=self._body_ids,
            forces=self.external_body_wrench_w[:, 1:, :3],
            torques=self.external_body_wrench_w[:, 1:, 3:],
        )
        result = self.backend.step_torch(self.control, nsteps=1)
        if getattr(self, "has_applied_wrench_readback", False):
            try:
                readback = self.backend.get_applied_body_wrench_w_torch(self._body_ids)
                if isinstance(readback, tuple) and len(readback) == 3:
                    f_app, t_app, v_app = readback
                    self.applied_body_wrench_w[:, 0] = 0
                    self.applied_body_wrench_w[:, 1:, :3] = f_app
                    self.applied_body_wrench_w[:, 1:, 3:] = t_app
                    self.applied_body_wrench_support.copy_(v_app)
            except NotImplementedError:
                self.has_applied_wrench_readback = False
        self._execution_steps.add_(1)
        self._state = result.public_state
        if result.applied_joint_effort is None:
            raise RuntimeError("Newton did not expose actual solved native effort")
        self.effort.copy_(result.applied_joint_effort)
        for name, (first, second) in self.contact_pairs.items():
            values = self.backend.get_body_pair_contact_facts(
                np.asarray(first), np.asarray(second), include_world_force=True)
            for key, value in values.items():
                self.contact_facts[name][key].copy_(value)
        self.contact_support[:] = True
        self.contact_sample_time_s.copy_(sample_time)
        self.contact_sample_execution_step.copy_(sample_step)
        self.record_contact_substep()

    def body_state(self, body_ids):
        return self.backend.get_body_state_w_torch(np.asarray(body_ids))

    def physics_time(self):
        return self.backend.get_physics_time_torch()

    def forward(self):
        # Public reset/step already own completed kinematics; no extra solve or reset.
        return None

    def capture_physics(self):
        return self.backend.capture_full_physics_state(), self.effort.clone()

    def restore_physics(self, state):
        physics, effort = state
        self._state = self.backend.restore_full_physics_state_torch(
            physics, source_rows=torch.arange(self.num_envs, device=self.device),
        )
        self.effort.copy_(effort)
