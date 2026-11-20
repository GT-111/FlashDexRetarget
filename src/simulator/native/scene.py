"""Native scene/entity views shared by direct MJWarp and the public Newton adapter."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from simulator.mjcf_meta import site_table

from .entity import (
    ArticulationData,
    BodyInertialProperties,
    FloatingRootView,
    NamedArticulationView,
    NativeRobotAdapter,
    _Cached,
    _FreshnessClock,
)
from .plant import (
    NativePlantSnapshot,
    free_root_velocity_to_native,
    root_drive_indices,
    root_origin_wrench,
)


@dataclass
class NativeSimControlContext:
    """Same-plant integration plus an uncommitted authored reset, if present."""

    plant: NativePlantSnapshot
    pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


@dataclass
class NativeSceneControlContext:
    simulation: NativeSimControlContext
    joint_position_targets: dict[str, torch.Tensor]


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
        count = sim.plant.num_envs
        mass = torch.as_tensor(model.body_mass[self._body_ids], device=self.device, dtype=torch.float32)
        com = torch.as_tensor(model.body_ipos[self._body_ids], device=self.device, dtype=torch.float32)
        principal = torch.as_tensor(model.body_inertia[self._body_ids], device=self.device, dtype=torch.float32)
        rotation = matrix_from_quat(torch.as_tensor(
            model.body_iquat[self._body_ids], device=self.device, dtype=torch.float32,
        ))
        inertia = rotation @ torch.diag_embed(principal) @ rotation.transpose(-1, -2)
        self._body_properties = BodyInertialProperties(
            tuple(self.body_names), mass.expand(count, -1), com.expand(count, -1, -1),
            inertia.expand(count, -1, -1, -1),
        )
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

    def body_inertial_properties(self):
        """Fixed compiled-model properties; this recipe does not mutate dynamics."""
        return self._body_properties

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
        value = root_origin_wrench(
            force, torque, rotation[:, 0], self._body_properties.com_position_body_m[:, 0],
        )
        self._sim.plant.control[:, self._root_drives] = value

    def write_external_wrench_to_sim(self, forces, torques, env_ids=None, body_ids=None):
        """Replace detached WORLD body-COM requests, independently of native motors."""
        rows = self._sim.rows(env_ids)
        ids = range(len(self.body_names)) if body_ids is None else body_ids
        native = torch.as_tensor(self._body_ids[list(ids)], device=self.device, dtype=torch.long)
        self._sim.plant.external_body_wrench_w[rows[:, None], native[None]] = torch.cat(
            (forces.detach(), torques.detach()), -1,
        )

    def clear_external_wrenches(self, env_ids=None):
        rows = self._sim.rows(env_ids)
        self._sim.plant.external_body_wrench_w[rows[:, None], self._body_ids] = 0
        if len(self._root_drives):
            self._sim.plant.control[rows[:, None], self._root_drives] = 0

    def reset(self, env_ids=None):
        rows = self._sim.rows(env_ids)
        self._joint_pos_target[rows] = self._default_joint_pos[rows]
        self.clear_external_wrenches(rows)
        self._clock.bump()


class NativeSim:
    supports_dynamic_gravity = False

    def __init__(self, plant):
        self.plant, self.device = plant, plant.device
        self.mj_model = plant.model
        self._gravity_world_m_per_s2 = tuple(float(value) for value in plant.model.opt.gravity)
        self._clock = _FreshnessClock()
        self._pending = None

    def capture_control_context(self) -> NativeSimControlContext:
        """Capture outside a control interval; assets and model remain borrowed.

        Plant restore deliberately discards executed interval evidence. This
        context is used at the initial reset boundary, not to replay an interval.
        """
        if self.plant.contact_interval is not None and self.plant.contact_interval._active:
            raise RuntimeError("native control context requires a closed control interval")
        return NativeSimControlContext(
            self.plant.snapshot(),
            None if self._pending is None else tuple(value.clone() for value in self._pending),
        )

    def restore_control_context(self, context: NativeSimControlContext) -> None:
        if self.plant.contact_interval is not None and self.plant.contact_interval._active:
            raise RuntimeError("native control context requires a closed control interval")
        self.plant.restore(context.plant)
        self._pending = None if context.pending is None else tuple(value.clone() for value in context.pending)
        self._clock.bump()  # Invalidate named kinematics; the token is not physics time.

    def begin_control_step(self):
        if self.plant.contact_interval is not None:
            self.plant.contact_interval.begin()

    def end_control_step(self):
        if self.plant.contact_interval is not None:
            self.plant.contact_interval.finish()

    def gravity_world_m_per_s2(self):
        """Constructor gravity of the executing, nonrandomized native recipe."""
        return self._gravity_world_m_per_s2

    def set_gravity_world_m_per_s2(self, gravity):
        actual = self.gravity_world_m_per_s2()
        if tuple(float(value) for value in gravity) != actual:
            raise NotImplementedError("native dynamic gravity is not implemented; keep constructor gravity")
        return actual

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
        model = sim.plant.model
        gravity = torch.as_tensor(model.opt.gravity, device=sim.device, dtype=torch.float32)
        gravity_magnitude = gravity.norm()
        self.gravity_up_w = -gravity / gravity_magnitude.clamp_min(1e-12)
        root_ids = [model.body(obj.root_body_name).id for obj in layout.objects]
        mass = torch.as_tensor(model.body_subtreemass[root_ids], device=sim.device, dtype=torch.float32)
        self.native_object_weights_n = mass * gravity_magnitude
        self.root_wrench_application = {
            "frame": "WORLD", "point": "FREE_body_origin",
            "conversion": "link_COM_torque_plus_rotated_COM_offset_cross_force",
            "recorded_wrench_point": "root_link_COM_before_origin_shift",
        }
        self.object_parts = {
            f"{obj.entity_name}/{p.name}": {"entity_name": obj.entity_name, "body_name": p.body_name}
            for obj, p in layout.part_bindings
        }
        self._object_pair_contact_addresses = {
            key: dict(first_entity_name=first.entity_name, first_root_body_name=first.root_body_name,
                      second_entity_name=second.entity_name, second_root_body_name=second.root_body_name)
            for key, (first, second) in layout.object_pair_bindings.items()
        }
        self._object_part_body_ids = torch.tensor(
            [sim.plant.model.body(part.body_name).id for _, part in layout.part_bindings],
            device=sim.device, dtype=torch.long,
        )

    def capture_control_context(self) -> NativeSceneControlContext:
        return NativeSceneControlContext(
            self.sim.capture_control_context(),
            {binding.entity_name: self.entities[binding.entity_name]._joint_pos_target.clone()
             for binding in self.native_layout.articulations},
        )

    def restore_control_context(self, context: NativeSceneControlContext) -> None:
        self.sim.restore_control_context(context.simulation)
        for name, target in context.joint_position_targets.items():
            # Plant control is already restored; reinstalling would overwrite it.
            self.entities[name]._joint_pos_target.copy_(target)

    def bind_tracking_command(self, command):
        # Native task geometry reads admitted references directly; actual pair
        # facts belong to the coupled plant, independently of reference intent.
        del command

    def tracking_contact_metadata(self):
        # No Isaac net-link/SDF force association is installed on this scene.
        return None

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

    def physics_execution_clock(self):
        """Backend-admitted accepted-step facts, independently of raw solver time."""
        return self.sim.plant.execution_clock()

    def hand_part_contact_facts(self):
        addresses = {
            f"{hand.side}/{obj.entity_name}/{part.name}": {
                "hand_side": hand.side, "hand_body_name": hand.root_body_name,
                "part_entity_name": obj.entity_name, "part_body_name": part.body_name,
            }
            for hand in self.native_layout.hands for obj, part in self.native_layout.part_bindings
        }
        return {
            "pairs": {key: pair for key, pair in self.sim.plant.contact_facts.items() if key in addresses},
            "addresses": addresses,
            "measurement_support": self.sim.plant.contact_support,
            "sample_time_s": self.sim.plant.contact_sample_time_s,
            "clock": "actual_solver_time_at_last_integrated_substep_begin",
            "normal_force_authority": "actual_solver_body_pair_positive_normal_force",
            "distance_authority": "force_bearing_actual_collider_max_absolute_separation",
        }

    def hand_part_contact_interval_facts(self):
        if self.sim.plant.contact_interval is None:
            return None
        last = self.hand_part_contact_facts()
        interval = self.sim.plant.contact_interval.facts()
        return {
            **interval, "pairs": {key: interval["pairs"][key] for key in last["pairs"]},
            "addresses": last["addresses"],
            "normal_force_authority": last["normal_force_authority"],
            "distance_authority": last["distance_authority"],
        }

    def object_pair_contact_interval_facts(self):
        """Named pairs of distinct complete object trees on the shared solver clock."""
        interval = self.sim.plant.contact_interval
        addresses = self._object_pair_contact_addresses
        if interval is None or not addresses:
            return None
        facts = interval.facts()
        if "execution_interval_complete" not in facts:
            return None
        force = torch.stack([facts["pairs"][key]["normal_force_n"] for key in addresses], dim=-1)
        force_on_second = torch.stack([facts["pairs"][key]["force_on_second_w_n"] for key in addresses], dim=-2)
        complete = (facts["interval_complete"] & facts["execution_interval_complete"]
                    & facts["measurement_support"].all(-1))
        support = complete[:, None] & torch.isfinite(force).all(dim=1) & torch.isfinite(force_on_second).all(-1).all(1)
        force = torch.where(support[:, None, :], force, torch.nan)
        force_on_second = torch.where(support[:, None, :, None], force_on_second, torch.nan)
        return dict(pairs=list(addresses.values()), num_substeps=facts["num_substeps"],
                    normal_force_n=force, force_on_second_w_n=force_on_second,
                    peak_normal_force_n=force.amax(dim=1), measurement_support=support)

    def support_contact_interval_pairs(self):
        """Fixed-world and independent object contacts from the same hand/part solve interval."""
        pairs = self.sim.plant.contact_interval.pairs
        names = (*self.native_layout.support_pair_bindings, *self.native_layout.object_pair_bindings)
        return {name: pairs[name] for name in names if name in pairs}

    def applied_object_wrench_interval_facts(self):
        interval = self.sim.plant.contact_interval
        if interval is None or not self.object_parts:
            return None
        readback = interval.applied_wrench_facts()
        if readback is None:
            return None
        peaks = readback["peaks"].index_select(1, self._object_part_body_ids)
        return {
            "entity_names": tuple(obj.entity_name for obj, _ in self.native_layout.part_bindings),
            "part_body_names": tuple(part.body_name for _, part in self.native_layout.part_bindings),
            "peak_force_n": peaks[..., 0],
            "peak_torque_nm": peaks[..., 1],
            "substep_wrench_w": readback["substep_wrench_w"].index_select(2, self._object_part_body_ids),
            "substep_support": readback["substep_support"],
            "measurement_support": readback["measurement_support"],
            "num_substeps": readback["num_substeps"],
            "authority": readback["authority"],
        }
