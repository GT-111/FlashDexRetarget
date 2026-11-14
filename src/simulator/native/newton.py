"""Flash native task over the installed DexForge public Newton engine."""

from __future__ import annotations

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
        self._state = self.backend.public_state_torch()
        self.effort = torch.zeros(num_envs, self.model.nv, device=device)
        self.identity = {
            "backend": "newton", "solver": "SolverMuJoCo",
            "contact_generation": collision_generation,
            "native_engine": "installed_dexforge_tracking",
            "scene_xml_path": scene_xml_path, "sim_dt_s": float(sim_dt),
        }

    def state(self):
        value = self._state
        return value.qpos, value.qvel

    def reset(self, env_ids):
        q = torch.as_tensor(self.backend.get_default_qpos(), device=self.device, dtype=self.control.dtype)
        self.set_state(env_ids, q.expand(len(env_ids), -1), self.effort.new_zeros(len(env_ids), self.model.nv))
        self.control[env_ids] = 0

    def set_state(self, env_ids, qpos, qvel):
        self._state = self.backend.set_state_torch(env_ids, qpos=qpos, qvel=qvel)
        self.effort[env_ids] = 0
        self.invalidate_contacts(env_ids)

    def advance(self):
        sample_time = self.physics_time()
        result = self.backend.step_torch(self.control, nsteps=1)
        self._state = result.public_state
        if result.applied_joint_effort is None:
            raise RuntimeError("Newton did not expose actual solved native effort")
        self.effort.copy_(result.applied_joint_effort)
        for name, (first, second) in self.contact_pairs.items():
            values = self.backend.get_body_pair_contact_facts(np.asarray(first), np.asarray(second))
            for key, value in values.items():
                self.contact_facts[name][key].copy_(value)
        self.contact_support[:] = True
        self.contact_sample_time_s.copy_(sample_time)

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
