"""Native scene/entity views shared by direct MJWarp and the public Newton adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from simulator.mjcf_meta import site_table

from .entity import (
    ArticulationData,
    FloatingRootView,
    NamedArticulationView,
    NativeRobotAdapter,
    _Cached,
    _FreshnessClock,
)
from .plant import free_root_velocity_to_native, root_drive_indices, root_origin_wrench


class NativeArticulation(FloatingRootView, NamedArticulationView):
    """A named view into ONE coupled plant, never a separate hand/object simulation."""

    is_fixed_base = False

    def __init__(self, binding, meta, sim, *, hand=False):
        self.binding, self._meta, self._sim = binding, meta, sim
        self.device, self._clock = sim.device, sim._clock
        self._cache = _Cached(self._clock)
        self.body_names, self.joint_names = list(meta.body_names), list(meta.joint_names)
        self.num_joints = len(self.joint_names)
        model = sim.plant.model
        self._body_ids = np.asarray([model.body(n).id for n in self.body_names])
        self.indexing = SimpleNamespace(body_ids=torch.as_tensor(self._body_ids, device=self.device))
        self._qpos_ids = torch.as_tensor(meta.joint_qpos_address, device=self.device, dtype=torch.long)
        self._qvel_ids = torch.as_tensor(meta.joint_dof_address, device=self.device, dtype=torch.long)
        self._actuators = {self.joint_names.index(j): model.actuator(a).id for a, j in meta.actuator_joint.items()}
        if len(self._actuators) != len(meta.actuator_names):
            raise ValueError("native position view requires one authored actuator per independent joint")
        self.root_body_name = binding.root_body_name
        self._root = next(r for r in meta.free_roots if r.name == binding.root_joint_name)
        if len(meta.free_roots) != 1 or self._root.body_name != self.root_body_name:
            raise ValueError("native articulation must retain its single declared FREE root")
        self._root_drives = root_drive_indices(model, binding.root_joint_name) if hand else np.empty(0, dtype=int)
        if not hand:
            import mujoco
            attached = np.isin(model.actuator_trntype, (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT))
            bodies = model.jnt_bodyid[model.actuator_trnid[attached, 0]]
            if np.isin(bodies, self._body_ids).any():
                raise ValueError("native object roots and scalar joints must remain passive")
        self._mass = torch.as_tensor(meta.body_mass, device=self.device, dtype=torch.float32)
        self._ipos = torch.as_tensor(meta.body_ipos, device=self.device, dtype=torch.float32)
        self._inertia = torch.as_tensor(meta.body_inertia, device=self.device, dtype=torch.float32)
        self._inertia_quat = torch.as_tensor(meta.body_iquat, device=self.device, dtype=torch.float32)
        count = sim.plant.num_envs
        self._default_joint_pos = torch.as_tensor(meta.joint_default, device=self.device, dtype=torch.float32).expand(count, -1)
        self._default_joint_vel = torch.zeros_like(self._default_joint_pos)
        self._joint_pos_target = self._default_joint_pos.clone()
        self._soft_limits = torch.as_tensor(meta.joint_range, device=self.device, dtype=torch.float32).expand(count, -1, -1)
        self._default_root_state = torch.zeros(count, 13, device=self.device)
        self._default_root_state[:, :7] = torch.as_tensor(self._root.default_pose, device=self.device)
        self.site_names, self._site_parent, self._site_lpos, self._site_lquat = site_table(
            meta, {n: i for i, n in enumerate(self.body_names)}, self.device,
        )
        self.data = ArticulationData(self)
        self.data.indexing = self.indexing
        import mujoco
        self.cfg = SimpleNamespace(spec_fn=lambda: mujoco.MjSpec.from_file(meta.xml_path))

    def _joint_pos(self):
        return self._sim.plant.state()[0][:, self._qpos_ids]

    def _joint_vel(self):
        return self._sim.plant.state()[1][:, self._qvel_ids]

    def _joint_effort(self):
        return self._sim.plant.effort[:, self._qvel_ids]

    def _body_pose(self):
        return self._cache.get("body_state", lambda: self._sim.plant.body_state(self._body_ids))[:2]

    def _body_vel(self):
        return self._cache.get("body_state", lambda: self._sim.plant.body_state(self._body_ids))[2:]

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        if joint_ids is not None:
            raise ValueError("native reset writes the complete named scalar state")
        rows = self._sim.rows(env_ids)
        q, v = self._sim.pending_state(rows)
        q[:, self._qpos_ids] = position
        v[:, self._qvel_ids] = velocity
        self._joint_pos_target[rows] = position
        active = list(self._actuators)
        self.set_joint_position_target(position[:, active], joint_ids=active, env_ids=rows)

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        rows = self._sim.rows(env_ids)
        ids = range(self.num_joints) if joint_ids is None else list(joint_ids)
        for column, joint in enumerate(ids):
            joint = int(joint)
            if joint not in self._actuators:
                raise ValueError("passive mimic/object joint cannot receive a position target")
            self._sim.plant.control[rows, self._actuators[joint]] = target[:, column]
            self._joint_pos_target[rows, joint] = target[:, column]

    def write_root_state_to_sim(self, root_state, env_ids=None):
        rows = self._sim.rows(env_ids)
        q, v = self._sim.pending_state(rows)
        a, d = self._root.qpos_address, self._root.dof_address
        q[:, a:a + 7] = root_state[:, :7]
        v[:, d:d + 6] = free_root_velocity_to_native(root_state)

    def _write_root_wrench(self, force, torque):
        if not len(self._root_drives):
            raise ValueError("passive object has no root wrench authority")
        _, rotation = self._body_pose()
        value = root_origin_wrench(force, torque, rotation[:, 0], self._ipos[0])
        self._sim.plant.control[:, self._root_drives] = value

    def clear_external_wrenches(self, env_ids=None):
        if len(self._root_drives):
            rows = self._sim.rows(env_ids)
            self._sim.plant.control[rows[:, None], self._root_drives] = 0

    def reset(self, env_ids=None):
        rows = self._sim.rows(env_ids)
        self._joint_pos_target[rows] = self._default_joint_pos[rows]
        self.clear_external_wrenches(rows)
        self._clock.bump()


class NativeSim:
    def __init__(self, plant):
        self.plant, self.device = plant, plant.device
        self.mj_model = plant.model
        self._clock = _FreshnessClock()
        self._pending = None

    def rows(self, env_ids):
        return torch.arange(self.plant.num_envs, device=self.device) if env_ids is None else env_ids

    def pending_state(self, rows):
        if self._pending is None:
            q, v = self.plant.state()
            self._pending = (rows.clone(), q[rows].clone(), v[rows].clone())
        if not torch.equal(rows, self._pending[0]):
            raise ValueError("one native reset transaction must retain its selected world rows")
        return self._pending[1:]

    def flush_state(self):
        if self._pending is not None:
            self.plant.set_state(*self._pending)
            self._pending = None
            self._clock.bump()

    def expand_model_fields(self, fields):
        if fields:
            raise ValueError("this native producer recipe does not implement domain-randomized model fields")

    def reset(self, env_ids):
        self.plant.reset(env_ids)
        self._pending = None
        self._clock.bump()

    def step(self):
        if self._pending is not None:
            raise RuntimeError("commit authored reset before native advance")
        self.plant.advance()
        self._clock.bump()

    def forward(self):
        self.plant.forward()
        self._clock.bump()

    def sense(self):
        # Native task geometry uses real part poses/SDF. Pair forces are separate facts.
        return None


class NativeScene:
    def __init__(self, layout, scene_meta, sim, robot_name):
        self.sim, self._clock, self.native_layout = sim, sim._clock, layout
        self.num_envs = sim.plant.num_envs
        self.env_origins = torch.zeros(self.num_envs, 3, device=sim.device)
        self.entities = {
            b.entity_name: NativeArticulation(b, scene_meta.subtree(b.root_body_name), sim,
                                              hand=b in layout.hands)
            for b in layout.articulations
        }
        self.entities[robot_name] = NativeRobotAdapter(
            {h.side: self.entities[h.entity_name] for h in layout.hands},
            scene_meta=scene_meta, clock=self._clock, device=sim.device,
        )
        self.sensors = {}
        self.backend_identity = sim.plant.identity
        self.root_wrench_application = {
            "frame": "WORLD", "point": "FREE_body_origin",
            "conversion": "link_COM_torque_plus_rotated_COM_offset_cross_force",
            "recorded_wrench_point": "root_link_COM_before_origin_shift",
        }
        self.object_parts = {p.name: {"entity_name": layout.object.entity_name, "body_name": p.body_name}
                             for p in layout.object.parts}

    def __getitem__(self, name):
        return self.entities[name]

    def __contains__(self, name):
        return name in self.entities

    def keys(self):
        return self.entities.keys()

    def write_data_to_sim(self):
        self.sim.flush_state()

    def update(self, dt):
        del dt
        self._clock.bump()

    def reset(self, env_ids=None):
        for binding in self.native_layout.articulations:
            self.entities[binding.entity_name].reset(env_ids)

    def physics_time_s(self):
        """Actual endpoint solver clock, independently of source/control phases."""
        return self.sim.plant.physics_time()

    def hand_part_contact_facts(self):
        return {
            "pairs": self.sim.plant.contact_facts,
            "addresses": {
                f"{h.side}/{p.name}": {
                    "hand_side": h.side, "hand_body_name": h.root_body_name,
                    "part_entity_name": self.native_layout.object.entity_name,
                    "part_body_name": p.body_name,
                }
                for h in self.native_layout.hands for p in self.native_layout.object.parts
            },
            "measurement_support": self.sim.plant.contact_support,
            "sample_time_s": self.sim.plant.contact_sample_time_s,
            "clock": "actual_solver_time_at_last_integrated_substep_begin",
            "normal_force_authority": "actual_solver_body_pair_positive_normal_force",
            "distance_authority": "force_bearing_actual_collider_max_absolute_separation",
        }
