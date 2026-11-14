"""Direct authored-MJCF MuJoCo Warp plant, independently of Isaac or Newton import."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from .plant import NativePlant, body_origin_velocity


class MjWarpPlant(NativePlant):
    def __init__(self, *, scene_xml_path, num_envs, sim_dt, device, contact_pairs,
                 nconmax, njmax):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        wp.init()
        self._wp, self._mjw = wp, mjw
        selected = torch.device(device)
        self._torch_stream = torch.cuda.current_stream(selected) if selected.type == "cuda" else None
        # Retain the external Warp wrapper for the actual Torch stream's lifetime.
        self._warp_stream = wp.stream_from_torch(self._torch_stream) if self._torch_stream is not None else None
        model = mujoco.MjModel.from_xml_path(scene_xml_path)
        model.opt.timestep = float(sim_dt)
        self._init_buffers(model, num_envs, device, contact_pairs)
        self.identity = {"backend": "mjwarp", "solver": "direct_mujoco_warp",
                         "contact_generation": "authored_mujoco", "nconmax": nconmax, "njmax": njmax,
                         "scene_xml_path": scene_xml_path, "sim_dt_s": float(sim_dt)}
        with self._scope():
            self._model = mjw.put_model(model)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self._data = mjw.put_data(model, data, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            self._qpos, self._qvel = wp.to_torch(self._data.qpos), wp.to_torch(self._data.qvel)
            self._solved_effort = wp.to_torch(self._data.qfrc_actuator)
            self.effort = torch.zeros_like(self._solved_effort)
            self._ctrl = wp.to_torch(self._data.ctrl)
            self._reset_mask = wp.zeros(num_envs, dtype=wp.bool, device=device)
            self._contact_ids = wp.array(np.arange(self._data.naconmax, dtype=np.int32), device=device)
            self._contact_forces = wp.zeros(self._data.naconmax, dtype=wp.spatial_vector, device=device)
            width = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
            self._integration = wp.empty((num_envs, width), dtype=wp.float32, device=device)
            self._body_root = torch.as_tensor(model.body_rootid, device=device, dtype=torch.long)
            self._geom_body = torch.as_tensor(model.geom_bodyid, device=device, dtype=torch.long)
            mjw.forward(self._model, self._data)

    @contextmanager
    def _scope(self):
        with self._wp.ScopedDevice(self.device):
            if self._warp_stream is None:
                yield
            else:
                if torch.cuda.current_stream(self.device).cuda_stream != self._torch_stream.cuda_stream:
                    raise RuntimeError("native MJWarp uses its declared owning Torch stream")
                with self._wp.ScopedStream(self._warp_stream):
                    yield

    def state(self):
        return self._qpos, self._qvel

    def reset(self, env_ids):
        with self._scope():
            mask = self._wp.to_torch(self._reset_mask)
            mask[:] = False
            mask[env_ids] = True
            self._mjw.reset_data(self._model, self._data, reset=self._reset_mask)
            self.control[env_ids] = 0
            self.effort[env_ids] = 0
            self.invalidate_contacts(env_ids)

    def set_state(self, env_ids, qpos, qvel):
        with self._scope():
            self._qpos[env_ids] = qpos
            self._qvel[env_ids] = qvel
            self.effort[env_ids] = 0
            self.invalidate_contacts(env_ids)
            self.forward()

    def advance(self):
        with self._scope():
            sample_time = self.physics_time()
            self._ctrl.copy_(self.control)
            self._mjw.step(self._model, self._data)
            self.effort.copy_(self._solved_effort)
            torch._assert_async((self._wp.to_torch(self._data.overflow) == 0).all(), "MJWarp contact/constraint capacity exceeded")
            # A subsequent forward refreshes endpoint geometry; retain the actual solve first.
            self._read_solved_contacts()
            self.contact_sample_time_s.copy_(sample_time)
            self.forward()

    def _read_solved_contacts(self):
        self._mjw.contact_force(self._model, self._data, self._contact_ids, False, self._contact_forces)
        wp, data = self._wp, self._data
        geom = wp.to_torch(data.contact.geom).long()
        active = (wp.to_torch(self._contact_ids) < wp.to_torch(data.nacon)[0]) & geom.ge(0).all(-1)
        body = self._geom_body[geom.clamp(0, self.model.ngeom - 1)]
        world = wp.to_torch(data.contact.worldid).long()
        normal = wp.to_torch(self._contact_forces)[:, 0].clamp_min(0)
        distance = wp.to_torch(data.contact.dist).abs()
        for name, (first, second) in self.contact_pairs.items():
            a = torch.as_tensor(first, device=self.device)
            b = torch.as_tensor(second, device=self.device)
            selected = active & ((torch.isin(body[:, 0], a) & torch.isin(body[:, 1], b))
                                 | (torch.isin(body[:, 0], b) & torch.isin(body[:, 1], a)))
            rows = world.clamp(0, self.num_envs - 1)
            facts = self.contact_facts[name]
            facts["normal_force_n"].zero_().scatter_add_(0, rows, torch.where(selected, normal, 0))
            bearing = selected & (normal > 0)
            present = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            present.scatter_add_(0, rows, bearing.to(torch.int32))
            facts["force_bearing_contact_present"].copy_(present > 0)
            facts["force_bearing_distance_max_m"].zero_().scatter_reduce_(0, rows, torch.where(bearing, distance, 0), reduce="amax")
            facts["force_bearing_distance_max_m"].masked_fill_(present == 0, float("nan"))
        self.contact_support[:] = True

    def body_state(self, body_ids):
        with self._scope():
            wp, data = self._wp, self._data
            p = wp.to_torch(data.xpos)[:, body_ids]
            q = wp.to_torch(data.xquat)[:, body_ids]
            spatial = wp.to_torch(data.cvel)[:, body_ids]
            center = wp.to_torch(data.subtree_com)[:, self._body_root[body_ids]]
            linear, angular = body_origin_velocity(spatial, p, center)
            return p, q, linear, angular

    def physics_time(self):
        with self._scope():
            return self._wp.to_torch(self._data.time).clone()

    def forward(self):
        with self._scope():
            self._mjw.fwd_position(self._model, self._data)
            self._mjw.fwd_velocity(self._model, self._data)

    def capture_physics(self):
        with self._scope():
            self._mjw.get_state(self._model, self._data, self._integration, self._mjw.State.INTEGRATION)
            return self._wp.to_torch(self._integration).clone(), self.effort.clone()

    def restore_physics(self, state):
        with self._scope():
            integration, effort = state
            self._mjw.set_state(self._model, self._data, self._wp.from_torch(integration.contiguous()), self._mjw.State.INTEGRATION)
            self.forward()
            self.effort.copy_(effort)
